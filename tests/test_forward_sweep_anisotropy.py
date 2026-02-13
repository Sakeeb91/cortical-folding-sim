"""Integration test for anisotropy-enabled sweep configs."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


def test_forward_sweep_reads_anisotropy_config(tmp_path: Path):
    csv_path = tmp_path / "ab.csv"
    summary_path = tmp_path / "ab_summary.json"
    manifest_path = tmp_path / "ab_manifest.json"
    cmd = [
        "python3.11",
        "scripts/run_forward_sweep.py",
        "--config-path",
        "configs/week3_anisotropy_ab.json",
        "--n-steps",
        "30",
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
    labels = {r["label"] for r in rows}
    assert labels == {"isotropic_baseline", "anisotropic_z_bias"}
