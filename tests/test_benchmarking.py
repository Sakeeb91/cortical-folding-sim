"""Tests for benchmark reproducibility utilities."""

import json

from cortical_folding.benchmarking import config_hash, is_gi_plausible, load_grid_config


def test_config_hash_is_order_invariant_for_dict_keys():
    """Hash should be stable regardless of dict key order."""
    a = {"x": 1, "y": 2.0, "z": {"k": 3}}
    b = {"z": {"k": 3}, "y": 2.0, "x": 1}
    assert config_hash(a) == config_hash(b)


def test_gi_plausibility_bounds_are_inclusive():
    """Values at bounds should be considered plausible."""
    assert is_gi_plausible(0.8, 0.8, 3.5)
    assert is_gi_plausible(3.5, 0.8, 3.5)
    assert not is_gi_plausible(0.79, 0.8, 3.5)
    assert not is_gi_plausible(3.51, 0.8, 3.5)


def test_load_grid_config_reads_list(tmp_path):
    """Grid config loader should read a JSON list of run configs."""
    path = tmp_path / "grid.json"
    expected = [{"growth_mode": "uniform", "Kc": 2.0}]
    path.write_text(json.dumps(expected))
    assert load_grid_config(path) == expected
