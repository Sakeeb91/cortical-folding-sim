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

from cortical_folding.benchmarking import (
    config_hash,
    current_git_commit,
    is_gi_plausible,
    load_grid_config,
)
from cortical_folding.losses import gyrification_index
from cortical_folding.mesh import build_topology, compute_face_areas, compute_mean_curvature
from cortical_folding.solver import SimParams, compute_force_components, make_initial_state, simulate
from cortical_folding.synthetic import (
    create_anisotropy_field,
    create_icosphere,
    create_regional_growth,
    create_skull,
    create_uniform_growth,
)


def cfg_get(cfg: dict, key: str, default):
    """Safe config value fetch with defaults for backward compatibility."""
    return cfg[key] if key in cfg else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default="configs/forward_sweep_baseline.json",
        help="Path to frozen JSON sweep config grid.",
    )
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
    parser.add_argument(
        "--output-manifest",
        default="results/forward_sweep_manifest.json",
        help="Path to output run manifest JSON.",
    )
    parser.add_argument("--subdivisions", type=int, default=3, help="Icosphere subdivisions.")
    parser.add_argument("--radius", type=float, default=1.0, help="Initial sphere radius.")
    parser.add_argument("--skull-radius", type=float, default=1.5, help="Skull radius.")
    parser.add_argument("--n-steps", type=int, default=200, help="Simulation timesteps.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation dt.")
    parser.add_argument("--seed", type=int, default=42, help="Run seed metadata.")
    parser.add_argument(
        "--gi-plausible-min",
        type=float,
        default=0.8,
        help="Lower bound for GI plausibility flag.",
    )
    parser.add_argument(
        "--gi-plausible-max",
        type=float,
        default=3.5,
        help="Upper bound for GI plausibility flag.",
    )
    parser.add_argument(
        "--fail-fast-disp-max",
        type=float,
        default=1.0,
        help="Mark run unstable if displacement p95 exceeds this threshold.",
    )
    parser.add_argument(
        "--fail-fast-penetration-max",
        type=float,
        default=0.5,
        help="Mark run unstable if max skull penetration exceeds this threshold.",
    )
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


def build_quick_grid() -> list[dict]:
    """Small debug grid for rapid local checks."""
    return [
        {
            "label": "quick_uniform",
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
            "label": "quick_uniform_high",
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
            "label": "quick_regional_iso",
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
            "label": "quick_regional_aniso",
            "growth_mode": "regional",
            "uniform_rate": 0.0,
            "high_rate": 0.90,
            "low_rate": 0.15,
            "Kc": 1.8,
            "Kb": 2.5,
            "carrying_cap_factor": 4.0,
            "damping": 0.9,
            "tau": 500.0,
            "anisotropy_mode": "regional",
            "anisotropy_strength": 0.35,
            "anisotropy_high": 1.0,
            "anisotropy_low": 0.0,
        },
    ]


def collision_mode_label(cfg: dict) -> str:
    """Canonical collision mode label for reporting."""
    if not cfg_get(cfg, "enable_self_collision", False):
        return "disabled"
    if cfg_get(cfg, "self_collision_use_spatial_hash", False):
        return "spatial_hash"
    return "sampled"


def compute_nonadjacent_overlap_stats(
    verts_np: np.ndarray,
    edges_np: np.ndarray,
    min_dist: float,
) -> dict[str, float]:
    """Measure non-adjacent pair overlap depth statistics."""
    n_verts = verts_np.shape[0]
    adjacency = np.zeros((n_verts, n_verts), dtype=bool)
    adjacency[edges_np[:, 0], edges_np[:, 1]] = True
    adjacency[edges_np[:, 1], edges_np[:, 0]] = True

    tri_mask = np.triu(np.ones((n_verts, n_verts), dtype=bool), k=1)
    valid_mask = tri_mask & (~adjacency)
    if not np.any(valid_mask):
        return {
            "collision_overlap_mean": 0.0,
            "collision_overlap_p95": 0.0,
            "collision_overlap_max": 0.0,
            "collision_overlap_count": 0.0,
            "collision_overlap_frac": 0.0,
        }

    diff = verts_np[:, None, :] - verts_np[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    pair_dist = dist[valid_mask]
    overlap = np.maximum(min_dist - pair_dist, 0.0)
    overlap_count = int(np.sum(overlap > 0.0))
    total_pairs = int(overlap.shape[0])
    return {
        "collision_overlap_mean": float(np.mean(overlap)),
        "collision_overlap_p95": float(np.percentile(overlap, 95)),
        "collision_overlap_max": float(np.max(overlap)),
        "collision_overlap_count": float(overlap_count),
        "collision_overlap_frac": float(overlap_count / max(total_pairs, 1)),
    }


def run_single(
    run_id: int,
    cfg: dict,
    verts: jnp.ndarray,
    topo,
    skull_center: jnp.ndarray,
    skull_radius: float,
    n_steps: int,
    dt: float,
    seed: int,
    gi_plausible_min: float,
    gi_plausible_max: float,
    fail_fast_disp_max: float,
    fail_fast_penetration_max: float,
    sweep_config_hash: str,
    git_commit: str,
) -> dict:
    """Execute one config and return a flat metrics row."""
    if cfg_get(cfg, "growth_mode", "uniform") == "uniform":
        growth = create_uniform_growth(topo.faces.shape[0], rate=cfg_get(cfg, "uniform_rate", 0.5))
    else:
        growth = create_regional_growth(
            verts,
            topo.faces,
            high_rate=cfg_get(cfg, "high_rate", 0.8),
            low_rate=cfg_get(cfg, "low_rate", 0.1),
            axis=2,
            threshold=0.0,
        )
    anisotropy_mode = cfg_get(cfg, "anisotropy_mode", "none")
    anisotropy_axis = jnp.array(cfg_get(cfg, "anisotropy_axis", [0.0, 0.0, 1.0]))
    face_anisotropy = create_anisotropy_field(
        mode=anisotropy_mode,
        vertices=verts,
        faces=topo.faces,
        high_value=cfg_get(cfg, "anisotropy_high", 1.0),
        low_value=cfg_get(cfg, "anisotropy_low", 0.0),
        axis=2,
        threshold=0.0,
    )

    params = SimParams(
        Kc=cfg_get(cfg, "Kc", 2.0),
        Kb=cfg_get(cfg, "Kb", 3.0),
        damping=cfg_get(cfg, "damping", 0.9),
        skull_center=skull_center,
        skull_radius=skull_radius,
        skull_stiffness=100.0,
        carrying_cap_factor=cfg_get(cfg, "carrying_cap_factor", 4.0),
        tau=cfg_get(cfg, "tau", 500.0),
        dt=dt,
        enable_self_collision=cfg_get(cfg, "enable_self_collision", False),
        self_collision_min_dist=cfg_get(cfg, "self_collision_min_dist", 0.02),
        self_collision_stiffness=cfg_get(cfg, "self_collision_stiffness", 50.0),
        self_collision_n_sample=cfg_get(cfg, "self_collision_n_sample", 256),
        self_collision_use_spatial_hash=cfg_get(cfg, "self_collision_use_spatial_hash", False),
        self_collision_hash_cell_size=cfg_get(cfg, "self_collision_hash_cell_size", 0.02),
        self_collision_hash_neighbor_window=cfg_get(
            cfg, "self_collision_hash_neighbor_window", 8
        ),
        self_collision_deterministic_fallback=cfg_get(
            cfg, "self_collision_deterministic_fallback", True
        ),
        self_collision_fallback_n_sample=cfg_get(cfg, "self_collision_fallback_n_sample", 256),
        anisotropy_strength=cfg_get(cfg, "anisotropy_strength", 0.0),
        anisotropy_axis=anisotropy_axis,
    )

    initial_state = make_initial_state(verts, topo)
    initial_area = float(jnp.sum(compute_face_areas(verts, topo)))

    t0 = time.perf_counter()
    final_state, _ = simulate(
        initial_state, topo, growth, params, face_anisotropy=face_anisotropy, n_steps=n_steps
    )
    jax.block_until_ready(final_state.vertices)
    runtime_s = time.perf_counter() - t0

    final_verts_np = np.asarray(final_state.vertices)
    stable = bool(np.isfinite(final_verts_np).all())
    fail_reason = "none"
    if not stable:
        fail_reason = "non_finite_vertices"

    row = {
        "run_id": run_id,
        "label": cfg_get(cfg, "label", f"run_{run_id}"),
        "growth_mode": cfg_get(cfg, "growth_mode", "uniform"),
        "uniform_rate": cfg_get(cfg, "uniform_rate", 0.0),
        "high_rate": cfg_get(cfg, "high_rate", 0.0),
        "low_rate": cfg_get(cfg, "low_rate", 0.0),
        "Kc": cfg_get(cfg, "Kc", 2.0),
        "Kb": cfg_get(cfg, "Kb", 3.0),
        "damping": cfg_get(cfg, "damping", 0.9),
        "carrying_cap_factor": cfg_get(cfg, "carrying_cap_factor", 4.0),
        "tau": cfg_get(cfg, "tau", 500.0),
        "anisotropy_mode": anisotropy_mode,
        "anisotropy_strength": cfg_get(cfg, "anisotropy_strength", 0.0),
        "anisotropy_high": cfg_get(cfg, "anisotropy_high", 0.0),
        "anisotropy_low": cfg_get(cfg, "anisotropy_low", 0.0),
        "collision_mode": collision_mode_label(cfg),
        "enable_self_collision": int(cfg_get(cfg, "enable_self_collision", False)),
        "self_collision_use_spatial_hash": int(
            cfg_get(cfg, "self_collision_use_spatial_hash", False)
        ),
        "self_collision_min_dist": cfg_get(cfg, "self_collision_min_dist", 0.02),
        "self_collision_stiffness": cfg_get(cfg, "self_collision_stiffness", 50.0),
        "self_collision_n_sample": cfg_get(cfg, "self_collision_n_sample", 256),
        "self_collision_hash_cell_size": cfg_get(cfg, "self_collision_hash_cell_size", 0.02),
        "self_collision_hash_neighbor_window": cfg_get(
            cfg, "self_collision_hash_neighbor_window", 8
        ),
        "self_collision_deterministic_fallback": int(
            cfg_get(cfg, "self_collision_deterministic_fallback", True)
        ),
        "n_steps": n_steps,
        "dt": dt,
        "seed": seed,
        "run_config_hash": config_hash(cfg),
        "sweep_config_hash": sweep_config_hash,
        "git_commit": git_commit,
        "fail_reason": fail_reason,
        "stable": int(stable),
        "runtime_s": runtime_s,
    }

    if not stable:
        row.update(
            {
                "final_area": float("nan"),
                "area_ratio": float("nan"),
                "gi": float("nan"),
                "gi_plausible": 0,
                "mean_curv_mean": float("nan"),
                "mean_curv_std": float("nan"),
                "mean_curv_abs_max": float("nan"),
                "mean_curv_p50": float("nan"),
                "mean_curv_p90": float("nan"),
                "mean_curv_p99": float("nan"),
                "disp_mean": float("nan"),
                "disp_p95": float("nan"),
                "outside_skull_frac": float("nan"),
                "skull_penetration_mean": float("nan"),
                "skull_penetration_p95": float("nan"),
                "skull_penetration_max": float("nan"),
                "collision_force_l2": float("nan"),
                "total_force_l2": float("nan"),
                "collision_force_share": float("nan"),
                "collision_overlap_mean": float("nan"),
                "collision_overlap_p95": float("nan"),
                "collision_overlap_max": float("nan"),
                "collision_overlap_count": float("nan"),
                "collision_overlap_frac": float("nan"),
            }
        )
        return row

    final_area = float(jnp.sum(compute_face_areas(final_state.vertices, topo)))
    gi = float(gyrification_index(final_state.vertices, topo, skull_radius))
    curv = np.asarray(compute_mean_curvature(final_state.vertices, topo))
    disp = np.linalg.norm(final_verts_np - np.asarray(verts), axis=1)
    radii = np.linalg.norm(final_verts_np - np.asarray(skull_center), axis=1)
    penetration = np.maximum(radii - skull_radius, 0.0)
    force_components = compute_force_components(final_state, topo, params)
    collision_force_l2 = float(np.linalg.norm(np.asarray(force_components.collision)))
    total_force_l2 = float(np.linalg.norm(np.asarray(force_components.total)))
    collision_force_share = collision_force_l2 / max(total_force_l2, 1e-12)
    overlap_stats = compute_nonadjacent_overlap_stats(
        final_verts_np,
        np.asarray(topo.edges),
        float(params.self_collision_min_dist),
    )

    row.update(
        {
            "final_area": final_area,
            "area_ratio": final_area / max(initial_area, 1e-12),
            "gi": gi,
            "gi_plausible": int(is_gi_plausible(gi, gi_plausible_min, gi_plausible_max)),
            "mean_curv_mean": float(np.mean(curv)),
            "mean_curv_std": float(np.std(curv)),
            "mean_curv_abs_max": float(np.max(np.abs(curv))),
            "mean_curv_p50": float(np.percentile(curv, 50)),
            "mean_curv_p90": float(np.percentile(curv, 90)),
            "mean_curv_p99": float(np.percentile(curv, 99)),
            "disp_mean": float(np.mean(disp)),
            "disp_p95": float(np.percentile(disp, 95)),
            "outside_skull_frac": float(np.mean(radii > skull_radius)),
            "skull_penetration_mean": float(np.mean(penetration)),
            "skull_penetration_p95": float(np.percentile(penetration, 95)),
            "skull_penetration_max": float(np.max(penetration)),
            "collision_force_l2": collision_force_l2,
            "total_force_l2": total_force_l2,
            "collision_force_share": collision_force_share,
            **overlap_stats,
        }
    )
    if row["disp_p95"] > fail_fast_disp_max:
        row["stable"] = 0
        row["fail_reason"] = "dispersion_explosion"
    if row["skull_penetration_max"] > fail_fast_penetration_max:
        row["stable"] = 0
        row["fail_reason"] = "penetration_explosion"
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


def write_summary(rows: list[dict], path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stable_rows = [r for r in rows if r["stable"] == 1]
    unstable_rows = [r for r in rows if r["stable"] == 0]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "n_runs": len(rows),
        "n_anisotropic_runs": int(sum(1 for r in rows if float(r["anisotropy_strength"]) > 0)),
        "n_isotropic_runs": int(sum(1 for r in rows if float(r["anisotropy_strength"]) == 0)),
        "n_collision_enabled_runs": int(sum(1 for r in rows if int(r["enable_self_collision"]) == 1)),
        "n_spatial_hash_collision_runs": int(
            sum(1 for r in rows if int(r["self_collision_use_spatial_hash"]) == 1)
        ),
        "n_stable": len(stable_rows),
        "n_unstable": len(unstable_rows),
        "stability_rate": (len(stable_rows) / len(rows)) if rows else 0.0,
        "fail_reason_counts": {
            k: int(v)
            for k, v in {
                "none": sum(1 for r in rows if r["fail_reason"] == "none"),
                "non_finite_vertices": sum(
                    1 for r in rows if r["fail_reason"] == "non_finite_vertices"
                ),
                "dispersion_explosion": sum(
                    1 for r in rows if r["fail_reason"] == "dispersion_explosion"
                ),
                "penetration_explosion": sum(
                    1 for r in rows if r["fail_reason"] == "penetration_explosion"
                ),
            }.items()
            if v > 0
        },
    }

    if stable_rows:
        gi_values = [r["gi"] for r in stable_rows]
        plausible_count = int(sum(r["gi_plausible"] for r in stable_rows))
        summary.update(
            {
                "gi_mean": float(np.mean(gi_values)),
                "gi_std": float(np.std(gi_values)),
                "best_gi_run_id": int(max(stable_rows, key=lambda r: r["gi"])["run_id"]),
                "gi_plausible_count": plausible_count,
                "gi_plausible_rate": float(plausible_count / len(stable_rows)),
            }
        )

    with path.open("w") as f:
        json.dump(summary, f, indent=2)


def write_manifest(path: Path, metadata: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "n_runs": len(rows),
        "run_config_hashes": [r["run_config_hash"] for r in rows],
        "output_files": {
            "csv": metadata["output_csv"],
            "summary": metadata["output_summary"],
        },
    }
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()

    verts, faces = create_icosphere(subdivisions=args.subdivisions, radius=args.radius)
    topo = build_topology(verts, faces)
    skull_center, skull_radius = create_skull(radius=args.skull_radius)

    grid = build_quick_grid() if args.quick else load_grid_config(args.config_path)
    if args.max_runs is not None:
        grid = grid[: args.max_runs]
    sweep_config_hash = config_hash(grid)
    git_commit = current_git_commit()

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
            seed=args.seed,
            gi_plausible_min=args.gi_plausible_min,
            gi_plausible_max=args.gi_plausible_max,
            fail_fast_disp_max=args.fail_fast_disp_max,
            fail_fast_penetration_max=args.fail_fast_penetration_max,
            sweep_config_hash=sweep_config_hash,
            git_commit=git_commit,
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
    manifest_path = Path(args.output_manifest)
    metadata = {
        "seed": args.seed,
        "git_commit": git_commit,
        "sweep_config_hash": sweep_config_hash,
        "config_path": args.config_path,
        "gi_plausible_min": args.gi_plausible_min,
        "gi_plausible_max": args.gi_plausible_max,
        "fail_fast_disp_max": args.fail_fast_disp_max,
        "fail_fast_penetration_max": args.fail_fast_penetration_max,
        "output_csv": str(csv_path),
        "output_summary": str(summary_path),
    }
    write_csv(rows, csv_path)
    write_summary(rows, summary_path, metadata)
    write_manifest(manifest_path, metadata, rows)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
