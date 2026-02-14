"""Tests for Week 8 one-command final packaging regeneration driver."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_regenerate_week8_final_package_dry_run_emits_expected_steps(tmp_path: Path):
    summary_path = tmp_path / "week8_packaging_summary.json"

    cmd = [
        "python3.11",
        "scripts/regenerate_week8_final_package.py",
        "--dry-run",
        "--output-json",
        str(summary_path),
        "--week7-animation-summary-json",
        str(tmp_path / "week7_animation_summary.json"),
        "--week7-results-index-doc",
        str(tmp_path / "results_index.md"),
        "--week7-results-index-json",
        str(tmp_path / "week7_results_index_summary.json"),
        "--week7-manifest-json",
        str(tmp_path / "week7_manifest.json"),
        "--week7-pack-summary-json",
        str(tmp_path / "week7_pack_summary.json"),
        "--packet-doc",
        str(tmp_path / "week8_packet.md"),
        "--captions-doc",
        str(tmp_path / "week8_captions.md"),
        "--methods-settings-json",
        str(tmp_path / "week8_methods.json"),
        "--tables-json",
        str(tmp_path / "week8_tables.json"),
        "--captions-json",
        str(tmp_path / "week8_captions.json"),
        "--commands-json",
        str(tmp_path / "week8_commands.json"),
        "--artifact-bundle-json",
        str(tmp_path / "week8_bundle.json"),
        "--submission-summary-json",
        str(tmp_path / "week8_submission_summary.json"),
    ]
    subprocess.check_call(cmd)

    with summary_path.open() as f:
        payload = json.load(f)

    assert payload["dry_run"] is True
    assert payload["n_steps_executed"] == 0
    assert [step["name"] for step in payload["steps"]] == [
        "run_week7_pack",
        "build_week8_submission_packet",
    ]

