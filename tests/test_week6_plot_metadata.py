"""Tests for Week 6 figure metadata sidecars."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def test_week3_plot_writes_metadata_sidecar(tmp_path: Path):
    csv_path = tmp_path / "week3.csv"
    summary_path = tmp_path / "week3_summary.json"
    comparison_path = tmp_path / "week3_comparison.json"
    output_png = tmp_path / "week3.png"
    output_meta = tmp_path / "week3.meta.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "label", "run_config_hash"])
        writer.writeheader()
        writer.writerows(
            [
                {"run_id": "1", "label": "isotropic_baseline", "run_config_hash": "iso"},
                {"run_id": "2", "label": "anisotropic_z_bias", "run_config_hash": "aniso"},
            ]
        )

    summary_path.write_text(json.dumps({"sweep_config_hash": "hash", "git_commit": "abc123"}))
    comparison_path.write_text(
        json.dumps({"delta_gi": 0.4, "delta_curv_p90": -0.1, "delta_disp_p95": 0.2})
    )

    cmd = [
        "python3.11",
        "scripts/plot_week3_anisotropy.py",
        "--input-csv",
        str(csv_path),
        "--input-summary",
        str(summary_path),
        "--input-json",
        str(comparison_path),
        "--output-png",
        str(output_png),
        "--output-metadata",
        str(output_meta),
    ]
    subprocess.check_call(cmd)

    assert output_png.exists()
    with output_meta.open() as f:
        payload = json.load(f)
    assert payload["figure_id"] == "week3_anisotropy_delta"
    assert payload["source_run_ids"] == [1, 2]
