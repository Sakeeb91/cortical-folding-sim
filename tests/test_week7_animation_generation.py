"""Tests for Week 7 animation generation script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_generate_week7_comparison_animation_dry_run_writes_contract(tmp_path: Path):
    config_path = tmp_path / "configs.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "label": "layered_off_reference",
                    "growth_mode": "regional",
                    "high_rate": 1.1,
                    "low_rate": 0.2,
                },
                {
                    "label": "layered_on_reference",
                    "growth_mode": "regional",
                    "high_rate": 1.1,
                    "low_rate": 0.2,
                    "enable_two_layer_approx": True,
                },
            ]
        )
    )

    source_csv = tmp_path / "source.csv"
    source_csv.write_text("run_id,label,run_config_hash\n1,layered_off_reference,a\n2,layered_on_reference,b\n")
    source_summary = tmp_path / "source_summary.json"
    source_summary.write_text(json.dumps({"sweep_config_hash": "hash", "git_commit": "abc123"}))

    output_summary = tmp_path / "week7_summary.json"
    output_meta = tmp_path / "week7.meta.json"
    output_gif = tmp_path / "week7.gif"
    output_mp4 = tmp_path / "week7.mp4"

    cmd = [
        "python3.11",
        "scripts/generate_week7_comparison_animation.py",
        "--dry-run",
        "--config-path",
        str(config_path),
        "--source-csv",
        str(source_csv),
        "--source-summary",
        str(source_summary),
        "--output-summary",
        str(output_summary),
        "--output-metadata",
        str(output_meta),
        "--output-gif",
        str(output_gif),
        "--output-mp4",
        str(output_mp4),
    ]
    subprocess.check_call(cmd)

    with output_summary.open() as f:
        payload = json.load(f)

    assert payload["dry_run"] is True
    assert payload["baseline_source_run_id"] == 1
    assert payload["improved_source_run_id"] == 2
    assert payload["acceptance_animation_outputs_exist"] is False
    assert output_meta.exists()
