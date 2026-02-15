"""Generate deterministic Week 7 baseline-vs-improved comparison animations."""

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

from cortical_folding.benchmarking import config_hash, current_git_commit, load_grid_config
from cortical_folding.losses import gyrification_index
from cortical_folding.mesh import build_topology
from cortical_folding.solver import SimParams, make_initial_state, simulate
from cortical_folding.synthetic import (
    create_anisotropy_field,
    create_icosphere,
    create_regional_growth,
    create_uniform_growth,
)
from cortical_folding.viz import save_comparison_animation


def cfg_get(cfg: dict, key: str, default):
    return cfg[key] if key in cfg else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default="configs/week5_layered_ablation.json",
        help="Config grid containing baseline and improved labels.",
    )
    parser.add_argument(
        "--source-csv",
        default="results/week5_layered_ablation.csv",
        help="CSV artifact used for source run ID mapping.",
    )
    parser.add_argument(
        "--source-summary",
        default="results/week5_layered_ablation_summary.json",
        help="Summary artifact used for source sweep hash mapping.",
    )
    parser.add_argument(
        "--baseline-label",
        default="layered_off_reference",
        help="Baseline config label.",
    )
    parser.add_argument(
        "--improved-label",
        default="layered_on_reference",
        help="Improved config label.",
    )
    parser.add_argument("--subdivisions", type=int, default=3, help="Icosphere subdivisions.")
    parser.add_argument("--radius", type=float, default=1.0, help="Initial mesh radius.")
    parser.add_argument("--skull-radius", type=float, default=1.5, help="Skull radius for metrics.")
    parser.add_argument("--n-steps", type=int, default=140, help="Simulation timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed metadata for outputs.")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second.")
    parser.add_argument("--stride", type=int, default=6, help="Frame stride for output videos.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=90,
        help="Animation output DPI.",
    )
    parser.add_argument("--rotate", action="store_true", help="Rotate camera over time.")
    parser.add_argument(
        "--output-gif",
        default="docs/assets/week7_baseline_vs_improved.gif",
        help="Comparison GIF output.",
    )
    parser.add_argument(
        "--output-mp4",
        default="docs/assets/week7_baseline_vs_improved.mp4",
        help="Comparison MP4 output.",
    )
    parser.add_argument(
        "--output-summary",
        default="results/week7_animation_comparison_summary.json",
        help="Output JSON summary for Week 7 animation comparison.",
    )
    parser.add_argument(
        "--output-metadata",
        default="docs/assets/week7_baseline_vs_improved.meta.json",
        help="Output sidecar metadata path.",
    )
    parser.add_argument(
        "--gi-plausible-min",
        type=float,
        default=0.8,
        help="Lower GI plausibility threshold.",
    )
    parser.add_argument(
        "--gi-plausible-max",
        type=float,
        default=3.5,
        help="Upper GI plausibility threshold.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write metadata/summary contract without executing simulations.",
    )
    return parser.parse_args()


def load_source_row_map(path: str | Path) -> dict[str, dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row.get("label", ""): row for row in rows}


def load_source_summary(path: str | Path) -> dict:
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    with summary_path.open() as f:
        return json.load(f)


def build_params(cfg: dict, skull_radius: float) -> SimParams:
    return SimParams(
        Kc=cfg_get(cfg, "Kc", 2.0),
        Kb=cfg_get(cfg, "Kb", 3.0),
        damping=cfg_get(cfg, "damping", 0.9),
        skull_center=jnp.zeros(3),
        skull_radius=skull_radius,
        skull_stiffness=100.0,
        carrying_cap_factor=cfg_get(cfg, "carrying_cap_factor", 4.0),
        tau=cfg_get(cfg, "tau", 500.0),
        dt=cfg_get(cfg, "dt", 0.02),
        enable_self_collision=cfg_get(cfg, "enable_self_collision", False),
        self_collision_min_dist=cfg_get(cfg, "self_collision_min_dist", 0.02),
        self_collision_stiffness=cfg_get(cfg, "self_collision_stiffness", 50.0),
        self_collision_n_sample=cfg_get(cfg, "self_collision_n_sample", 256),
        self_collision_use_spatial_hash=cfg_get(cfg, "self_collision_use_spatial_hash", False),
        self_collision_hash_cell_size=cfg_get(cfg, "self_collision_hash_cell_size", 0.02),
        self_collision_hash_neighbor_window=cfg_get(cfg, "self_collision_hash_neighbor_window", 8),
        self_collision_deterministic_fallback=cfg_get(cfg, "self_collision_deterministic_fallback", True),
        self_collision_fallback_n_sample=cfg_get(cfg, "self_collision_fallback_n_sample", 256),
        self_collision_blend_sampled_weight=cfg_get(
            cfg, "self_collision_blend_sampled_weight", 0.0
        ),
        enable_adaptive_substepping=cfg_get(cfg, "enable_adaptive_substepping", False),
        adaptive_substep_min=cfg_get(cfg, "adaptive_substep_min", 1),
        adaptive_substep_max=cfg_get(cfg, "adaptive_substep_max", 4),
        adaptive_target_disp=cfg_get(cfg, "adaptive_target_disp", 0.01),
        adaptive_force_safety_scale=cfg_get(cfg, "adaptive_force_safety_scale", 1.0),
        fail_on_nonfinite=cfg_get(cfg, "fail_on_nonfinite", False),
        high_fidelity=cfg_get(cfg, "high_fidelity", False),
        anisotropy_strength=cfg_get(cfg, "anisotropy_strength", 0.0),
        anisotropy_axis=jnp.array(cfg_get(cfg, "anisotropy_axis", [0.0, 0.0, 1.0])),
        enable_two_layer_approx=cfg_get(cfg, "enable_two_layer_approx", False),
        two_layer_axis=jnp.array(cfg_get(cfg, "two_layer_axis", [0.0, 0.0, 1.0])),
        two_layer_threshold=cfg_get(cfg, "two_layer_threshold", 0.0),
        two_layer_transition_sharpness=cfg_get(cfg, "two_layer_transition_sharpness", 6.0),
        outer_layer_growth_scale=cfg_get(cfg, "outer_layer_growth_scale", 1.15),
        inner_layer_growth_scale=cfg_get(cfg, "inner_layer_growth_scale", 0.85),
        two_layer_coupling=cfg_get(cfg, "two_layer_coupling", 0.1),
        max_growth_rate=cfg_get(cfg, "max_growth_rate", 2.0),
        min_rest_area=cfg_get(cfg, "min_rest_area", 1e-8),
        min_rest_length=cfg_get(cfg, "min_rest_length", 1e-8),
        max_force_norm=cfg_get(cfg, "max_force_norm", 1e3),
        max_acc_norm=cfg_get(cfg, "max_acc_norm", 1e3),
        max_velocity_norm=cfg_get(cfg, "max_velocity_norm", 5.0),
        max_displacement_per_step=cfg_get(cfg, "max_displacement_per_step", 0.05),
    )


def build_growth_and_anisotropy(
    cfg: dict,
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    growth_mode = cfg_get(cfg, "growth_mode", "uniform")
    if growth_mode == "uniform":
        growth = create_uniform_growth(faces.shape[0], rate=cfg_get(cfg, "uniform_rate", 0.5))
    else:
        growth = create_regional_growth(
            vertices,
            faces,
            high_rate=cfg_get(cfg, "high_rate", 0.8),
            low_rate=cfg_get(cfg, "low_rate", 0.1),
            axis=2,
            threshold=0.0,
        )
    anisotropy = create_anisotropy_field(
        mode=cfg_get(cfg, "anisotropy_mode", "none"),
        vertices=vertices,
        faces=faces,
        high_value=cfg_get(cfg, "anisotropy_high", 1.0),
        low_value=cfg_get(cfg, "anisotropy_low", 0.0),
        axis=2,
        threshold=0.0,
    )
    return growth, anisotropy


def run_label(
    cfg: dict,
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    topo,
    n_steps: int,
    skull_radius: float,
    gi_plausible_min: float,
    gi_plausible_max: float,
) -> tuple[jnp.ndarray, dict]:
    growth, anisotropy = build_growth_and_anisotropy(cfg, vertices, faces)
    params = build_params(cfg, skull_radius)
    init = make_initial_state(vertices, topo)

    t0 = time.perf_counter()
    final_state, trajectory = simulate(
        init,
        topo,
        growth,
        params,
        face_anisotropy=anisotropy,
        n_steps=n_steps,
    )
    jax.block_until_ready(final_state.vertices)
    runtime_s = time.perf_counter() - t0

    verts_final = np.asarray(final_state.vertices)
    verts_initial = np.asarray(vertices)
    stable = bool(np.isfinite(verts_final).all())

    if not stable:
        metrics = {
            "stable": 0,
            "fail_reason": "non_finite_vertices",
            "gi": float("nan"),
            "gi_plausible": 0,
            "disp_p95": float("nan"),
            "outside_skull_frac": float("nan"),
            "runtime_s": runtime_s,
        }
        return trajectory, metrics

    disp = np.linalg.norm(verts_final - verts_initial, axis=1)
    radii = np.linalg.norm(verts_final, axis=1)
    metrics = {
        "stable": 1,
        "fail_reason": "none",
        "gi": float(gyrification_index(final_state.vertices, topo, skull_radius)),
        "disp_p95": float(np.percentile(disp, 95)),
        "outside_skull_frac": float(np.mean(radii > skull_radius)),
        "runtime_s": runtime_s,
    }
    metrics["gi_plausible"] = int(
        gi_plausible_min <= metrics["gi"] <= gi_plausible_max
    )
    return trajectory, metrics


def main() -> None:
    args = parse_args()

    grid = load_grid_config(args.config_path)
    by_label = {cfg.get("label", ""): cfg for cfg in grid}
    if args.baseline_label not in by_label:
        raise SystemExit(f"Missing baseline label in config: {args.baseline_label}")
    if args.improved_label not in by_label:
        raise SystemExit(f"Missing improved label in config: {args.improved_label}")

    baseline_cfg = by_label[args.baseline_label]
    improved_cfg = by_label[args.improved_label]
    sweep_hash = config_hash(grid)
    git_commit = current_git_commit(workdir=".")

    source_rows = load_source_row_map(args.source_csv)
    source_summary = load_source_summary(args.source_summary)
    baseline_source = source_rows.get(args.baseline_label, {})
    improved_source = source_rows.get(args.improved_label, {})

    summary_out = Path(args.output_summary)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out = Path(args.output_metadata)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            "python3.11",
            "scripts/generate_week7_comparison_animation.py",
            "--config-path",
            args.config_path,
            "--baseline-label",
            args.baseline_label,
            "--improved-label",
            args.improved_label,
            "--n-steps",
            str(args.n_steps),
            "--seed",
            str(args.seed),
            "--output-gif",
            args.output_gif,
            "--output-mp4",
            args.output_mp4,
            "--output-summary",
            args.output_summary,
            "--output-metadata",
            args.output_metadata,
        ]
    ]

    baseline_metrics = {}
    improved_metrics = {}
    if not args.dry_run:
        vertices, faces = create_icosphere(subdivisions=args.subdivisions, radius=args.radius)
        topo = build_topology(vertices, faces)

        baseline_trajectory, baseline_metrics = run_label(
            baseline_cfg,
            vertices,
            topo.faces,
            topo,
            args.n_steps,
            args.skull_radius,
            args.gi_plausible_min,
            args.gi_plausible_max,
        )
        improved_trajectory, improved_metrics = run_label(
            improved_cfg,
            vertices,
            topo.faces,
            topo,
            args.n_steps,
            args.skull_radius,
            args.gi_plausible_min,
            args.gi_plausible_max,
        )

        save_comparison_animation(
            baseline_trajectory,
            improved_trajectory,
            topo.faces,
            output_paths=[args.output_gif, args.output_mp4],
            fps=args.fps,
            stride=args.stride,
            dpi=args.dpi,
            rotate=args.rotate,
            baseline_title=f"Baseline: {args.baseline_label}",
            improved_title=f"Improved: {args.improved_label}",
            title="Week 7 Baseline vs Improved Comparison",
        )

    baseline_metrics = baseline_metrics or {
        "stable": 0,
        "fail_reason": "dry_run",
        "gi": float("nan"),
        "gi_plausible": 0,
        "disp_p95": float("nan"),
        "outside_skull_frac": float("nan"),
        "runtime_s": 0.0,
    }
    improved_metrics = improved_metrics or {
        "stable": 0,
        "fail_reason": "dry_run",
        "gi": float("nan"),
        "gi_plausible": 0,
        "disp_p95": float("nan"),
        "outside_skull_frac": float("nan"),
        "runtime_s": 0.0,
    }

    out_gif = Path(args.output_gif)
    out_mp4 = Path(args.output_mp4)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_steps": args.n_steps,
        "fps": args.fps,
        "stride": args.stride,
        "dpi": args.dpi,
        "subdivisions": args.subdivisions,
        "radius": args.radius,
        "skull_radius": args.skull_radius,
        "config_path": args.config_path,
        "sweep_config_hash": sweep_hash,
        "git_commit": git_commit,
        "source_csv": args.source_csv,
        "source_summary": args.source_summary,
        "baseline_label": args.baseline_label,
        "improved_label": args.improved_label,
        "baseline_run_config_hash": config_hash(baseline_cfg),
        "improved_run_config_hash": config_hash(improved_cfg),
        "baseline_source_run_id": int(float(baseline_source["run_id"]))
        if baseline_source.get("run_id")
        else None,
        "improved_source_run_id": int(float(improved_source["run_id"]))
        if improved_source.get("run_id")
        else None,
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "delta_gi_improved_minus_baseline": float(improved_metrics["gi"] - baseline_metrics["gi"]),
        "delta_disp_p95_improved_minus_baseline": float(
            improved_metrics["disp_p95"] - baseline_metrics["disp_p95"]
        ),
        "delta_outside_skull_frac_improved_minus_baseline": float(
            improved_metrics["outside_skull_frac"] - baseline_metrics["outside_skull_frac"]
        ),
        "output_gif": str(out_gif),
        "output_mp4": str(out_mp4),
        "output_metadata": str(meta_out),
        "source_sweep_config_hash": source_summary.get("sweep_config_hash"),
        "source_git_commit": source_summary.get("git_commit"),
        "commands": commands,
    }
    summary["acceptance_both_runs_stable"] = bool(
        baseline_metrics["stable"] == 1 and improved_metrics["stable"] == 1
    )
    summary["acceptance_improved_gi_not_worse"] = bool(
        improved_metrics["gi"] >= baseline_metrics["gi"] - 0.40
    )
    summary["acceptance_animation_outputs_exist"] = bool(
        out_gif.exists() and out_mp4.exists()
    )

    metadata = {
        "figure_id": "week7_baseline_vs_improved_animation",
        "generated_at_utc": summary["generated_at_utc"],
        "output_files": [str(out_gif), str(out_mp4)],
        "output_exists": [out_gif.exists(), out_mp4.exists()],
        "source_artifacts": {
            "config": args.config_path,
            "source_csv": args.source_csv,
            "source_summary": args.source_summary,
            "comparison_summary": args.output_summary,
        },
        "source_run_ids": [
            summary.get("baseline_source_run_id"),
            summary.get("improved_source_run_id"),
        ],
        "source_run_labels": [args.baseline_label, args.improved_label],
        "source_run_config_hashes": [
            summary["baseline_run_config_hash"],
            summary["improved_run_config_hash"],
        ],
        "source_sweep_config_hash": summary.get("source_sweep_config_hash"),
        "source_git_commit": summary.get("source_git_commit"),
        "week7_sweep_config_hash": summary.get("sweep_config_hash"),
    }

    with summary_out.open("w") as f:
        json.dump(summary, f, indent=2)
    with meta_out.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {summary_out}")
    print(f"Saved: {meta_out}")


if __name__ == "__main__":
    main()
