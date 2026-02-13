"""Integration test for fail-fast behavior in forward sweep runner."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


def test_forward_sweep_fail_fast_marks_unstable(tmp_path: Path):
    csv_path = tmp_path / "sweep.csv"
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"

    cmd = [
        "python3.11",
        "scripts/run_forward_sweep.py",
        "--quick",
        "--max-runs",
        "1",
        "--n-steps",
        "40",
        "--output-csv",
        str(csv_path),
        "--output-summary",
        str(summary_path),
        "--output-manifest",
        str(manifest_path),
        "--fail-fast-disp-max",
        "0.001",
    ]
    subprocess.check_call(cmd)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["stable"]) == 0
    assert rows[0]["fail_reason"] in {"dispersion_explosion", "penetration_explosion"}
