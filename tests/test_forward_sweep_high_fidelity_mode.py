"""Integration tests for high-fidelity mode in forward sweep runner."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def test_forward_sweep_high_fidelity_mode_emits_profile_columns(tmp_path: Path):
    csv_path = tmp_path / "hf.csv"
    summary_path = tmp_path / "hf_summary.json"
    manifest_path = tmp_path / "hf_manifest.json"

    cmd = [
        "python3.11",
        "scripts/run_forward_sweep.py",
        "--high-fidelity",
        "--quick",
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
        row = list(csv.DictReader(f))[0]
    assert row["simulation_mode"] == "high_fidelity"
    assert int(row["high_fidelity"]) == 1
    assert int(row["enable_adaptive_substepping"]) == 1

    with summary_path.open() as f:
        summary = json.load(f)
    assert summary["simulation_mode"] == "high_fidelity"
    assert "n_high_fidelity_runs" in summary
    assert "n_adaptive_substepping_runs" in summary
