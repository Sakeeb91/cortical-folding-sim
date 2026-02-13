"""Integration tests for collision diagnostics in forward sweep output."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def test_forward_sweep_emits_collision_diagnostic_columns(tmp_path: Path):
    csv_path = tmp_path / "sweep.csv"
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"

    cmd = [
        "python3.11",
        "scripts/run_forward_sweep.py",
        "--config-path",
        "configs/week4_collision_ablation.json",
        "--max-runs",
        "1",
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
    assert len(rows) == 1
    row = rows[0]
    expected = {
        "collision_mode",
        "collision_force_l2",
        "total_force_l2",
        "collision_force_share",
        "collision_overlap_mean",
        "collision_overlap_p95",
        "collision_overlap_max",
        "collision_overlap_count",
        "collision_overlap_frac",
    }
    assert expected.issubset(set(row.keys()))

    with summary_path.open() as f:
        summary = json.load(f)
    assert "collision_force_share_mean" in summary
    assert "collision_overlap_max_max" in summary


def test_week4_collision_ablation_script_writes_acceptance_fields(tmp_path: Path):
    csv_path = tmp_path / "week4.csv"
    summary_path = tmp_path / "week4_summary.json"
    manifest_path = tmp_path / "week4_manifest.json"
    output_json = tmp_path / "week4_comparison.json"

    cmd = [
        "python3.11",
        "scripts/run_week4_collision_ablation.py",
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
    assert "acceptance_collision_outliers_reduced" in payload
    assert "acceptance_runtime_within_budget" in payload
    assert "acceptance_week4_passed" in payload
