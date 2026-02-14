"""Tests for Week 7 one-command animation pack regeneration driver."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_regenerate_week7_animation_pack_dry_run_emits_expected_steps(tmp_path: Path):
    summary_path = tmp_path / "week7_pack_summary.json"
    animation_summary = tmp_path / "week7_animation_summary.json"
    results_index_doc = tmp_path / "results_index.md"
    results_index_json = tmp_path / "week7_results_index_summary.json"
    manifest_path = tmp_path / "week7_manifest.json"

    cmd = [
        "python3.11",
        "scripts/regenerate_week7_animation_pack.py",
        "--dry-run",
        "--output-json",
        str(summary_path),
        "--animation-summary-json",
        str(animation_summary),
        "--results-index-doc",
        str(results_index_doc),
        "--results-index-json",
        str(results_index_json),
        "--manifest-json",
        str(manifest_path),
    ]
    subprocess.check_call(cmd)

    with summary_path.open() as f:
        payload = json.load(f)

    assert payload["dry_run"] is True
    assert payload["n_steps_executed"] == 0
    assert [step["name"] for step in payload["steps"]] == [
        "run_week5",
        "generate_week7_animation",
        "build_results_index",
        "build_week7_manifest",
    ]
