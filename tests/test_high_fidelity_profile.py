"""Tests for high-fidelity profile helpers."""

from cortical_folding.high_fidelity import (
    HIGH_FIDELITY_PROFILE_VERSION,
    apply_high_fidelity_profile,
)


def test_apply_high_fidelity_profile_sets_expected_fields():
    cfg = {"label": "x", "growth_mode": "uniform", "uniform_rate": 0.3}
    out = apply_high_fidelity_profile(cfg)
    assert out["simulation_mode"] == "high_fidelity"
    assert out["profile_version"] == HIGH_FIDELITY_PROFILE_VERSION
    assert out["enable_adaptive_substepping"] is True
    assert out["high_fidelity"] is True
    assert out["self_collision_use_spatial_hash"] is True
