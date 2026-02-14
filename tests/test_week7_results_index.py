"""Tests for Week 7 results index generation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _touch(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        path.write_text("ok")
    else:
        path.write_text(json.dumps(payload))


def test_build_results_index_flags_missing_claim_artifacts(tmp_path: Path):
    week3 = tmp_path / "results/week3_anisotropy_comparison.json"
    week4 = tmp_path / "results/week4_collision_comparison.json"
    week5 = tmp_path / "results/week5_layered_comparison.json"
    week6_summary = tmp_path / "results/week6_figure_pipeline_summary.json"
    week6_manifest = tmp_path / "docs/assets/week6_figure_manifest.json"
    week7_summary = tmp_path / "results/week7_animation_comparison_summary.json"
    week7_manifest = tmp_path / "docs/assets/week7_animation_manifest.json"

    for path in (week3, week4, week5, week6_summary, week6_manifest, week7_summary):
        _touch(path, {"ok": True})

    output_doc = tmp_path / "docs/results_index.md"
    output_json = tmp_path / "results/week7_results_index_summary.json"

    cmd = [
        "python3.11",
        "scripts/build_results_index.py",
        "--output-doc",
        str(output_doc),
        "--output-json",
        str(output_json),
        "--week3-comparison",
        str(week3),
        "--week4-comparison",
        str(week4),
        "--week5-comparison",
        str(week5),
        "--week6-summary",
        str(week6_summary),
        "--week6-manifest",
        str(week6_manifest),
        "--week7-summary",
        str(week7_summary),
        "--week7-manifest",
        str(week7_manifest),
    ]
    subprocess.check_call(cmd)

    with output_json.open() as f:
        payload = json.load(f)

    assert output_doc.exists()
    assert payload["n_claims"] == 5
    assert payload["n_claims_with_linked_artifacts"] == 5
    assert payload["all_claims_have_linked_artifacts"] is True
