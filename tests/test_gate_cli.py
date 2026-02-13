"""CLI tests for check_forward_sweep_gates.py."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def _write_csv(path: Path) -> None:
    rows = [
        {
            "stable": "1",
            "outside_skull_frac": "0.05",
            "skull_penetration_p95": "0.01",
            "disp_p95": "0.2",
        },
        {
            "stable": "1",
            "outside_skull_frac": "0.08",
            "skull_penetration_p95": "0.02",
            "disp_p95": "0.25",
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_gate_cli_fail_mode_returns_nonzero(tmp_path: Path):
    csv_path = tmp_path / "sweep.csv"
    summary_path = tmp_path / "summary.json"
    gates_path = tmp_path / "gates.json"
    report_path = tmp_path / "report.json"
    _write_csv(csv_path)
    summary_path.write_text(json.dumps({"stability_rate": 0.6, "gi_plausible_rate": 0.1}))
    gates_path.write_text(
        json.dumps(
            {
                "min_stability_rate": 0.95,
                "min_gi_plausible_rate": 0.5,
                "max_outside_skull_frac_p95": 0.01,
                "max_skull_penetration_p95": 0.005,
                "max_disp_p95": 0.05,
            }
        )
    )

    cmd = [
        "python3.11",
        "scripts/check_forward_sweep_gates.py",
        "--input-csv",
        str(csv_path),
        "--input-summary",
        str(summary_path),
        "--gate-config",
        str(gates_path),
        "--output-report",
        str(report_path),
        "--fail-on-failure",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "Validation gates failed" in proc.stdout
    assert report_path.exists()
