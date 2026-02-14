"""Tests for Week 6 one-command figure regeneration driver."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_regenerate_week6_figures_dry_run_emits_expected_steps(tmp_path: Path):
    summary_path = tmp_path / "week6_summary.json"
    manifest_path = tmp_path / "week6_manifest.json"
    cmd = [
        "python3.11",
        "scripts/regenerate_week6_figures.py",
        "--dry-run",
        "--output-json",
        str(summary_path),
        "--manifest-json",
        str(manifest_path),
        "--n-steps",
        "30",
    ]
    subprocess.check_call(cmd)

    with summary_path.open() as f:
        payload = json.load(f)
    assert payload["dry_run"] is True
    assert payload["n_steps_executed"] == 0
    step_names = [s["name"] for s in payload["steps"]]
    assert step_names == [
        "run_week3",
        "plot_week3",
        "run_week4",
        "plot_week4",
        "run_week5",
        "plot_week5",
        "build_manifest",
    ]
