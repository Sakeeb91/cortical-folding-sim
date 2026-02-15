"""Tests for high-fidelity hardened validation script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_validate_high_fidelity_dry_run_outputs_contract(tmp_path: Path):
    output_dir = tmp_path / "hf"
    cmd = [
        "python3.11",
        "scripts/validate_high_fidelity.py",
        "--dry-run",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.check_call(cmd)

    report_path = output_dir / "validation_report.json"
    with report_path.open() as f:
        report = json.load(f)

    assert report["dry_run"] is True
    assert report["reproducibility"]["status"] == "dry_run"
    assert report["matrix"]["n_records"] >= 6
    assert "runtime_budget" in report
    assert "ci_parity" in report
