"""Integration tests for Week 5 layered-mode sweep and summary scripts."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def test_forward_sweep_emits_layered_columns(tmp_path: Path):
    csv_path = tmp_path / "week5.csv"
    summary_path = tmp_path / "week5_summary.json"
    manifest_path = tmp_path / "week5_manifest.json"

    cmd = [
        "python3.11",
        "scripts/run_forward_sweep.py",
        "--config-path",
        "configs/week5_layered_ablation.json",
        "--max-runs",
        "2",
        "--n-steps",
        "20",
        "--output-csv",
        str(csv_path),
        "--output-summary",
        str(summary_path),
        "--output-manifest",
        str(manifest_path),
    ]
    subprocess.check_call(cmd)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2

    expected = {
        "enable_two_layer_approx",
        "two_layer_threshold",
        "two_layer_transition_sharpness",
        "outer_layer_growth_scale",
        "inner_layer_growth_scale",
        "two_layer_coupling",
        "max_force_norm",
        "max_acc_norm",
        "max_velocity_norm",
        "max_displacement_per_step",
    }
    assert expected.issubset(set(rows[0].keys()))

    with summary_path.open() as f:
        summary = json.load(f)
    assert "n_two_layer_runs" in summary


def test_week5_layered_ablation_script_writes_acceptance_fields(tmp_path: Path):
    csv_path = tmp_path / "week5.csv"
    summary_path = tmp_path / "week5_summary.json"
    manifest_path = tmp_path / "week5_manifest.json"
    output_json = tmp_path / "week5_comparison.json"

    cmd = [
        "python3.11",
        "scripts/run_week5_layered_ablation.py",
        "--n-steps",
        "30",
        "--output-csv",
        str(csv_path),
        "--output-summary",
        str(summary_path),
        "--output-manifest",
        str(manifest_path),
        "--output-json",
        str(output_json),
    ]
    subprocess.check_call(cmd)

    with output_json.open() as f:
        payload = json.load(f)
    assert "acceptance_layered_mode_stable_on_reduced_grid" in payload
    assert "acceptance_robust_parameter_region_found" in payload
    assert "acceptance_week5_passed" in payload
    assert "robust_region_count" in payload
