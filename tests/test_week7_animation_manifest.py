"""Tests for Week 7 animation manifest generation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_build_week7_animation_manifest_maps_source_run_ids(tmp_path: Path):
    gif_path = tmp_path / "docs/assets/week7.gif"
    mp4_path = tmp_path / "docs/assets/week7.mp4"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.write_text("gif")
    mp4_path.write_text("mp4")

    summary_path = tmp_path / "results/week7_animation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "config_path": "configs/week5_layered_ablation.json",
                "source_csv": "results/week5_layered_ablation.csv",
                "source_summary": "results/week5_layered_ablation_summary.json",
            }
        )
    )

    metadata_path = tmp_path / "docs/assets/week7.meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "output_files": [str(gif_path), str(mp4_path)],
                "source_run_ids": [1, 2],
                "source_run_labels": ["layered_off_reference", "layered_on_reference"],
                "source_run_config_hashes": ["h1", "h2"],
                "source_sweep_config_hash": "sweep",
                "source_git_commit": "abc123",
            }
        )
    )

    results_index_path = tmp_path / "results/week7_results_index_summary.json"
    results_index_path.write_text(json.dumps({"all_claims_have_linked_artifacts": True}))

    output_path = tmp_path / "docs/assets/week7_animation_manifest.json"
    cmd = [
        "python3.11",
        "scripts/build_week7_animation_manifest.py",
        "--input-summary",
        str(summary_path),
        "--input-metadata",
        str(metadata_path),
        "--input-results-index",
        str(results_index_path),
        "--output-json",
        str(output_path),
        "--fail-on-missing-run-ids",
    ]
    subprocess.check_call(cmd)

    with output_path.open() as f:
        payload = json.load(f)
    assert payload["all_assets_have_source_run_ids"] is True
    assert payload["results_index_all_claims_have_linked_artifacts"] is True
    assert payload["assets"][0]["all_outputs_exist"] is True
