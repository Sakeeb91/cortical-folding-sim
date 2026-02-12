"""Run a reproducible forward-parameter sweep and log paper-ready metrics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cortical_folding.losses import gyrification_index
from cortical_folding.mesh import build_topology, compute_face_areas, compute_mean_curvature
from cortical_folding.solver import SimParams, make_initial_state, simulate
from cortical_folding.synthetic import (
    create_icosphere,
    create_regional_growth,
    create_skull,
    create_uniform_growth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        default="results/forward_sweep.csv",
        help="Path to output CSV with one row per run.",
    )
    parser.add_argument(
        "--output-summary",
        default="results/forward_sweep_summary.json",
        help="Path to output summary JSON.",
    )
    parser.add_argument("--subdivisions", type=int, default=3, help="Icosphere subdivisions.")
    parser.add_argument("--radius", type=float, default=1.0, help="Initial sphere radius.")
    parser.add_argument("--skull-radius", type=float, default=1.5, help="Skull radius.")
    parser.add_argument("--n-steps", type=int, default=200, help="Simulation timesteps.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation dt.")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for debugging/quick execution.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller grid for fast iteration.",
    )
    return parser.parse_args()


def build_grid(quick: bool) -> list[dict]:
    """Return run configurations for forward sweep."""
    if quick:
        return [
            {
                "growth_mode": "uniform",
                "uniform_rate": 0.30,
                "high_rate": 0.30,
                "low_rate": 0.30,
                "Kc": 2.0,
                "Kb": 3.0,
                "carrying_cap_factor": 3.5,
                "damping": 0.9,
                "tau": 500.0,
            },
            {
                "growth_mode": "uniform",
                "uniform_rate": 0.50,
                "high_rate": 0.50,
                "low_rate": 0.50,
                "Kc": 2.0,
                "Kb": 3.0,
                "carrying_cap_factor": 4.0,
                "damping": 0.9,
                "tau": 500.0,
            },
            {
                "growth_mode": "regional",
                "uniform_rate": 0.0,
                "high_rate": 0.80,
                "low_rate": 0.10,
                "Kc": 2.0,
                "Kb": 3.0,
                "carrying_cap_factor": 4.0,
                "damping": 0.9,
                "tau": 500.0,
            },
            {
                "growth_mode": "regional",
                "uniform_rate": 0.0,
                "high_rate": 0.90,
                "low_rate": 0.15,
                "Kc": 1.8,
                "Kb": 2.5,
                "carrying_cap_factor": 4.0,
                "damping": 0.9,
                "tau": 500.0,
            },
        ]

    grid: list[dict] = []
    for kc in [1.6, 2.0]:
        for kb in [2.5, 3.5]:
            for cap in [3.5, 4.0]:
                grid.append(
                    {
                        "growth_mode": "uniform",
                        "uniform_rate": 0.40,
                        "high_rate": 0.40,
                        "low_rate": 0.40,
                        "Kc": kc,
                        "Kb": kb,
                        "carrying_cap_factor": cap,
                        "damping": 0.9,
                        "tau": 500.0,
                    }
                )
                grid.append(
                    {
                        "growth_mode": "regional",
                        "uniform_rate": 0.0,
                        "high_rate": 0.80,
                        "low_rate": 0.10,
                        "Kc": kc,
                        "Kb": kb,
                        "carrying_cap_factor": cap,
                        "damping": 0.9,
                        "tau": 500.0,
                    }
                )
    return grid


def run_single(
    run_id: int,
    cfg: dict,
    verts: jnp.ndarray,
    topo,
    skull_center: jnp.ndarray,
    skull_radius: float,
    n_steps: int,
    dt: float,
) -> dict:
    """Execute one config and return a flat metrics row."""
    if cfg["growth_mode"] == "uniform":
        growth = create_uniform_growth(topo.faces.shape[0], rate=cfg["uniform_rate"])
    else:
        growth = create_regional_growth(
            verts,
            topo.faces,
            high_rate=cfg["high_rate"],
            low_rate=cfg["low_rate"],
            axis=2,
            threshold=0.0,
        )

    params = SimParams(
        Kc=cfg["Kc"],
        Kb=cfg["Kb"],
        damping=cfg["damping"],
        skull_center=skull_center,
        skull_radius=skull_radius,
        skull_stiffness=100.0,
        carrying_cap_factor=cfg["carrying_cap_factor"],
        tau=cfg["tau"],
        dt=dt,
    )

    initial_state = make_initial_state(verts, topo)
    initial_area = float(jnp.sum(compute_face_areas(verts, topo)))

    t0 = time.perf_counter()
    final_state, _ = simulate(initial_state, topo, growth, params, n_steps=n_steps)
    jax.block_until_ready(final_state.vertices)
    runtime_s = time.perf_counter() - t0

    final_verts_np = np.asarray(final_state.vertices)
    stable = bool(np.isfinite(final_verts_np).all())

    row = {
        "run_id": run_id,
        "growth_mode": cfg["growth_mode"],
        "uniform_rate": cfg["uniform_rate"],
        "high_rate": cfg["high_rate"],
        "low_rate": cfg["low_rate"],
        "Kc": cfg["Kc"],
        "Kb": cfg["Kb"],
        "damping": cfg["damping"],
        "carrying_cap_factor": cfg["carrying_cap_factor"],
        "tau": cfg["tau"],
        "n_steps": n_steps,
        "dt": dt,
        "stable": int(stable),
        "runtime_s": runtime_s,
    }

    if not stable:
        row.update(
            {
                "final_area": float("nan"),
                "area_ratio": float("nan"),
                "gi": float("nan"),
                "mean_curv_mean": float("nan"),
                "mean_curv_std": float("nan"),
                "mean_curv_abs_max": float("nan"),
                "disp_mean": float("nan"),
                "disp_p95": float("nan"),
                "outside_skull_frac": float("nan"),
            }
        )
        return row

    final_area = float(jnp.sum(compute_face_areas(final_state.vertices, topo)))
    gi = float(gyrification_index(final_state.vertices, topo, skull_radius))
    curv = np.asarray(compute_mean_curvature(final_state.vertices, topo))
    disp = np.linalg.norm(final_verts_np - np.asarray(verts), axis=1)
    radii = np.linalg.norm(final_verts_np - np.asarray(skull_center), axis=1)

    row.update(
        {
            "final_area": final_area,
            "area_ratio": final_area / max(initial_area, 1e-12),
            "gi": gi,
            "mean_curv_mean": float(np.mean(curv)),
            "mean_curv_std": float(np.std(curv)),
            "mean_curv_abs_max": float(np.max(np.abs(curv))),
            "disp_mean": float(np.mean(disp)),
            "disp_p95": float(np.percentile(disp, 95)),
            "outside_skull_frac": float(np.mean(radii > skull_radius)),
        }
    )
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stable_rows = [r for r in rows if r["stable"] == 1]
    unstable_rows = [r for r in rows if r["stable"] == 0]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(rows),
        "n_stable": len(stable_rows),
        "n_unstable": len(unstable_rows),
        "stability_rate": (len(stable_rows) / len(rows)) if rows else 0.0,
    }

    if stable_rows:
        gi_values = [r["gi"] for r in stable_rows]
        summary.update(
            {
                "gi_mean": float(np.mean(gi_values)),
                "gi_std": float(np.std(gi_values)),
                "best_gi_run_id": int(max(stable_rows, key=lambda r: r["gi"])["run_id"]),
            }
        )

    with path.open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()

    verts, faces = create_icosphere(subdivisions=args.subdivisions, radius=args.radius)
    topo = build_topology(verts, faces)
    skull_center, skull_radius = create_skull(radius=args.skull_radius)

    grid = build_grid(args.quick)
    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    print(
        f"Running forward sweep with {len(grid)} configs "
        f"(subdivisions={args.subdivisions}, steps={args.n_steps})"
    )

    rows = []
    for idx, cfg in enumerate(grid, start=1):
        row = run_single(
            run_id=idx,
            cfg=cfg,
            verts=verts,
            topo=topo,
            skull_center=skull_center,
            skull_radius=skull_radius,
            n_steps=args.n_steps,
            dt=args.dt,
        )
        rows.append(row)
        print(
            f"[{idx:02d}/{len(grid):02d}] "
            f"mode={row['growth_mode']:<8} "
            f"Kc={row['Kc']:.2f} Kb={row['Kb']:.2f} "
            f"GI={row['gi']:.3f} stable={row['stable']}"
        )

    csv_path = Path(args.output_csv)
    summary_path = Path(args.output_summary)
    write_csv(rows, csv_path)
    write_summary(rows, summary_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
