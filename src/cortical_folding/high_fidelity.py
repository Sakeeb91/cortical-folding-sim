"""High-fidelity profile controls for simulation and validation pipelines."""

from __future__ import annotations

from typing import Mapping


HIGH_FIDELITY_PROFILE_VERSION = "hf_v1_2026_02_15"


def high_fidelity_defaults() -> dict[str, float | int | bool]:
    """Deterministic high-fidelity controls for tighter numerical safety."""
    return {
        "high_fidelity": True,
        "enable_adaptive_substepping": True,
        "adaptive_substep_min": 1,
        "adaptive_substep_max": 4,
        "adaptive_target_disp": 0.012,
        "adaptive_force_safety_scale": 1.0,
        "fail_on_nonfinite": True,
        "max_growth_rate": 1.2,
        "max_force_norm": 260.0,
        "max_acc_norm": 220.0,
        "max_velocity_norm": 0.95,
        "max_displacement_per_step": 0.018,
        "enable_self_collision": True,
        "self_collision_min_dist": 0.04,
        "self_collision_stiffness": 70.0,
        "self_collision_n_sample": 1024,
        "self_collision_use_spatial_hash": True,
        "self_collision_hash_cell_size": 0.04,
        "self_collision_hash_neighbor_window": 10,
        "self_collision_deterministic_fallback": True,
        "self_collision_fallback_n_sample": 640,
        "self_collision_blend_sampled_weight": 0.35,
    }


def apply_high_fidelity_profile(cfg: Mapping[str, object]) -> dict[str, object]:
    """Return config with high-fidelity controls overlaid."""
    merged = dict(cfg)
    merged.update(high_fidelity_defaults())
    merged["profile_version"] = HIGH_FIDELITY_PROFILE_VERSION
    merged["simulation_mode"] = "high_fidelity"
    return merged
