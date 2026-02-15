"""Generate publication-grade baseline-vs-high-fidelity comparison renders."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cortical_folding.benchmarking import config_hash, current_git_commit, load_grid_config
from cortical_folding.high_fidelity import (
    HIGH_FIDELITY_PROFILE_VERSION,
    apply_high_fidelity_profile,
)
from cortical_folding.losses import gyrification_index
from cortical_folding.mesh import build_topology
from cortical_folding.solver import SimParams, make_initial_state, simulate
from cortical_folding.synthetic import (
    create_anisotropy_field,
    create_icosphere,
    create_regional_growth,
    create_uniform_growth,
)
from cortical_folding.viz import save_publication_comparison_animation


def cfg_get(cfg: dict, key: str, default):
    return cfg[key] if key in cfg else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default="configs/high_fidelity_publication_render.json",
        help="Config grid containing baseline and high-fidelity labels.",
    )
    parser.add_argument(
        "--baseline-label",
        default="publication_baseline",
        help="Baseline config label.",
    )
    parser.add_argument(
        "--improved-label",
        default="publication_high_fidelity",
        help="High-fidelity config label.",
    )
    parser.add_argument("--subdivisions", type=int, default=3, help="Icosphere subdivisions.")
    parser.add_argument("--radius", type=float, default=1.0, help="Initial mesh radius.")
    parser.add_argument("--skull-radius", type=float, default=1.5, help="Skull radius for metrics.")
    parser.add_argument("--n-steps", type=int, default=180, help="Simulation timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed metadata for outputs.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second.")
    parser.add_argument("--stride", type=int, default=4, help="Frame stride for output videos.")
    parser.add_argument("--dpi", type=int, default=120, help="Base DPI.")
    parser.add_argument("--width-px", type=int, default=1920, help="Render width in pixels.")
    parser.add_argument("--height-px", type=int, default=1080, help="Render height in pixels.")
    parser.add_argument(
        "--supersample-scale",
        type=int,
        default=2,
        help="Supersampling multiplier applied to DPI during export.",
    )
    parser.add_argument("--no-rotate", action="store_true", help="Disable smooth camera motion.")
    parser.add_argument(
        "--with-metric-overlays",
        action="store_true",
        help="Overlay GI, disp_p95, and outside_skull_frac on both panels.",
    )
    parser.add_argument(
        "--output-gif",
        default="docs/assets/high_fidelity/publication_comparison.gif",
        help="Publication GIF output path.",
    )
    parser.add_argument(
        "--output-mp4",
        default="docs/assets/high_fidelity/publication_comparison.mp4",
        help="Publication MP4 output path.",
    )
    parser.add_argument(
        "--output-summary",
        default="results/high_fidelity/publication_render_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--output-manifest",
        default="results/high_fidelity/publication_render_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit render contract without executing simulations.",
    )
    return parser.parse_args()


def materialize_cfg(cfg: dict) -> dict:
    mode = str(cfg_get(cfg, "simulation_mode", "standard"))
    if mode == "high_fidelity" or bool(cfg_get(cfg, "high_fidelity", False)):
        out = apply_high_fidelity_profile(cfg)
        out["profile_version"] = HIGH_FIDELITY_PROFILE_VERSION
        return out
    out = dict(cfg)
    out.setdefault("profile_version", "standard_v1")
    out.setdefault("simulation_mode", "standard")
    return out


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
    cfg: dict, vertices: jnp.ndarray, faces: jnp.ndarray
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


def metric_series(
    trajectory: jnp.ndarray,
    topo,
    skull_radius: float,
    initial_vertices: jnp.ndarray,
    stride: int,
) -> dict[str, np.ndarray]:
    traj_np = np.asarray(trajectory)
    init_np = np.asarray(initial_vertices)
    gi_values: list[float] = []
    disp_p95_values: list[float] = []
    outside_values: list[float] = []
    for verts_np in traj_np:
        verts = jnp.asarray(verts_np)
        gi_values.append(float(gyrification_index(verts, topo, skull_radius)))
        disp = np.linalg.norm(verts_np - init_np, axis=1)
        disp_p95_values.append(float(np.percentile(disp, 95)))
        radii = np.linalg.norm(verts_np, axis=1)
        outside_values.append(float(np.mean(radii > skull_radius)))
    return {
        "gi": np.asarray(gi_values)[::stride],
        "disp_p95": np.asarray(disp_p95_values)[::stride],
        "outside_skull_frac": np.asarray(outside_values)[::stride],
    }


def run_label(
    cfg: dict,
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    topo,
    n_steps: int,
    skull_radius: float,
    stride: int,
) -> tuple[jnp.ndarray, dict, dict[str, np.ndarray]]:
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

    final_np = np.asarray(final_state.vertices)
    init_np = np.asarray(vertices)
    stable = bool(np.isfinite(final_np).all())
    if not stable:
        return trajectory, {
            "stable": 0,
            "fail_reason": "non_finite_vertices",
            "gi": float("nan"),
            "disp_p95": float("nan"),
            "outside_skull_frac": float("nan"),
            "runtime_s": runtime_s,
        }, {
            "gi": np.zeros(trajectory.shape[0] // max(stride, 1) + 1, dtype=np.float64),
            "disp_p95": np.zeros(trajectory.shape[0] // max(stride, 1) + 1, dtype=np.float64),
            "outside_skull_frac": np.zeros(
                trajectory.shape[0] // max(stride, 1) + 1, dtype=np.float64
            ),
        }

    disp = np.linalg.norm(final_np - init_np, axis=1)
    radii = np.linalg.norm(final_np, axis=1)
    metrics = {
        "stable": 1,
        "fail_reason": "none",
        "gi": float(gyrification_index(final_state.vertices, topo, skull_radius)),
        "disp_p95": float(np.percentile(disp, 95)),
        "outside_skull_frac": float(np.mean(radii > skull_radius)),
        "runtime_s": runtime_s,
    }
    return trajectory, metrics, metric_series(trajectory, topo, skull_radius, vertices, stride)


def main() -> None:
    args = parse_args()
    config_grid = load_grid_config(args.config_path)
    by_label = {cfg.get("label", ""): materialize_cfg(cfg) for cfg in config_grid}

    if args.baseline_label not in by_label:
        raise SystemExit(f"Missing baseline label: {args.baseline_label}")
    if args.improved_label not in by_label:
        raise SystemExit(f"Missing improved label: {args.improved_label}")

    baseline_cfg = by_label[args.baseline_label]
    improved_cfg = by_label[args.improved_label]

    output_summary = Path(args.output_summary)
    output_manifest = Path(args.output_manifest)
    output_gif = Path(args.output_gif)
    output_mp4 = Path(args.output_mp4)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        baseline_metrics = {
            "stable": 0,
            "fail_reason": "dry_run",
            "gi": float("nan"),
            "disp_p95": float("nan"),
            "outside_skull_frac": float("nan"),
            "runtime_s": 0.0,
        }
        improved_metrics = dict(baseline_metrics)
        render_runtime_s = 0.0
    else:
        vertices, faces = create_icosphere(subdivisions=args.subdivisions, radius=args.radius)
        topo = build_topology(vertices, faces)

        baseline_traj, baseline_metrics, baseline_series = run_label(
            baseline_cfg, vertices, topo.faces, topo, args.n_steps, args.skull_radius, args.stride
        )
        improved_traj, improved_metrics, improved_series = run_label(
            improved_cfg, vertices, topo.faces, topo, args.n_steps, args.skull_radius, args.stride
        )

        overlays = None
        if args.with_metric_overlays:
            overlays = {
                "GI": (baseline_series["gi"], improved_series["gi"]),
                "disp_p95": (baseline_series["disp_p95"], improved_series["disp_p95"]),
                "outside_skull_frac": (
                    baseline_series["outside_skull_frac"],
                    improved_series["outside_skull_frac"],
                ),
            }

        render_t0 = time.perf_counter()
        save_publication_comparison_animation(
            baseline_traj,
            improved_traj,
            topo.faces,
            output_paths=[str(output_gif), str(output_mp4)],
            fps=args.fps,
            stride=args.stride,
            dpi=args.dpi,
            width_px=args.width_px,
            height_px=args.height_px,
            supersample_scale=args.supersample_scale,
            rotate=not args.no_rotate,
            baseline_title=f"Baseline ({args.baseline_label})",
            improved_title=f"Improved ({args.improved_label})",
            title="High-Fidelity Publication Comparison",
            metric_overlays=overlays,
        )
        render_runtime_s = time.perf_counter() - render_t0

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "generated_at_utc": generated_at,
        "seed": args.seed,
        "n_steps": args.n_steps,
        "fps": args.fps,
        "stride": args.stride,
        "subdivisions": args.subdivisions,
        "radius": args.radius,
        "skull_radius": args.skull_radius,
        "width_px": args.width_px,
        "height_px": args.height_px,
        "dpi": args.dpi,
        "supersample_scale": args.supersample_scale,
        "dry_run": args.dry_run,
        "metric_overlays_enabled": args.with_metric_overlays,
        "config_path": args.config_path,
        "sweep_config_hash": config_hash(config_grid),
        "baseline_label": args.baseline_label,
        "improved_label": args.improved_label,
        "baseline_run_config_hash": config_hash(baseline_cfg),
        "improved_run_config_hash": config_hash(improved_cfg),
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "delta_gi_improved_minus_baseline": float(
            improved_metrics["gi"] - baseline_metrics["gi"]
        ),
        "delta_disp_p95_improved_minus_baseline": float(
            improved_metrics["disp_p95"] - baseline_metrics["disp_p95"]
        ),
        "delta_outside_skull_frac_improved_minus_baseline": float(
            improved_metrics["outside_skull_frac"] - baseline_metrics["outside_skull_frac"]
        ),
        "runtime_s": {
            "baseline_sim": baseline_metrics["runtime_s"],
            "improved_sim": improved_metrics["runtime_s"],
            "render": render_runtime_s,
            "total": baseline_metrics["runtime_s"] + improved_metrics["runtime_s"] + render_runtime_s,
        },
        "output_gif": str(output_gif),
        "output_mp4": str(output_mp4),
        "git_commit": current_git_commit(workdir="."),
        "commands": [
            [
                "python3.11",
                "scripts/generate_high_fidelity_publication_render.py",
                "--config-path",
                args.config_path,
                "--baseline-label",
                args.baseline_label,
                "--improved-label",
                args.improved_label,
                "--n-steps",
                str(args.n_steps),
                "--output-gif",
                str(output_gif),
                "--output-mp4",
                str(output_mp4),
                "--output-summary",
                str(output_summary),
                "--output-manifest",
                str(output_manifest),
            ]
        ],
    }
    summary["acceptance_both_runs_stable"] = bool(
        baseline_metrics["stable"] == 1 and improved_metrics["stable"] == 1
    )
    summary["acceptance_outputs_exist"] = bool(output_gif.exists() and output_mp4.exists())
    summary["acceptance_resolution_1080p_or_higher"] = bool(
        args.width_px >= 1920 and args.height_px >= 1080
    )
    summary["passed"] = bool(
        summary["acceptance_both_runs_stable"]
        and summary["acceptance_outputs_exist"]
        and summary["acceptance_resolution_1080p_or_higher"]
    )

    manifest = {
        "generated_at_utc": generated_at,
        "config_hashes": {
            "sweep": summary["sweep_config_hash"],
            "baseline_run": summary["baseline_run_config_hash"],
            "improved_run": summary["improved_run_config_hash"],
        },
        "profile_versions": {
            "baseline": baseline_cfg.get("profile_version", "standard_v1"),
            "improved": improved_cfg.get("profile_version", "standard_v1"),
        },
        "outputs": {
            "summary": str(output_summary),
            "gif": str(output_gif),
            "mp4": str(output_mp4),
        },
        "runtime_s": summary["runtime_s"],
        "passed": summary["passed"],
    }

    with output_summary.open("w") as f:
        json.dump(summary, f, indent=2)
    with output_manifest.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {output_summary}")
    print(f"Saved: {output_manifest}")


if __name__ == "__main__":
    main()
