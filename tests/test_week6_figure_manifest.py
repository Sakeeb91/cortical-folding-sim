"""Tests for Week 6 figure manifest generation."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_build_week6_figure_manifest_maps_source_run_ids(tmp_path: Path):
    results_dir = tmp_path / "results"
    assets_dir = tmp_path / "assets"
    results_dir.mkdir()
    assets_dir.mkdir()

    _write_csv(
        results_dir / "week3_anisotropy_ab.csv",
        [
            {"run_id": "1", "label": "isotropic_baseline", "run_config_hash": "w3a"},
            {"run_id": "2", "label": "anisotropic_z_bias", "run_config_hash": "w3b"},
        ],
    )
    _write_csv(
        results_dir / "week4_collision_ablation.csv",
        [
            {"run_id": "1", "label": "collision_disabled_baseline", "run_config_hash": "w4a"},
            {"run_id": "2", "label": "collision_sampled_deterministic", "run_config_hash": "w4b"},
            {"run_id": "3", "label": "collision_spatial_hash", "run_config_hash": "w4c"},
        ],
    )
    _write_csv(
        results_dir / "week5_layered_ablation.csv",
        [
            {"run_id": "1", "label": "layered_off_reference", "run_config_hash": "w5a"},
            {"run_id": "2", "label": "layered_on_reference", "run_config_hash": "w5b"},
        ],
    )

    for name in (
        "week3_anisotropy_ab_summary.json",
        "week4_collision_ablation_summary.json",
        "week5_layered_ablation_summary.json",
    ):
        _write_json(results_dir / name, {"sweep_config_hash": "hash", "git_commit": "abc123"})

    for name in (
        "week3_anisotropy_ab_manifest.json",
        "week4_collision_ablation_manifest.json",
        "week5_layered_ablation_manifest.json",
        "week3_anisotropy_comparison.json",
        "week4_collision_comparison.json",
        "week5_layered_comparison.json",
    ):
        _write_json(results_dir / name, {"ok": True})

    for stem in (
        "week3_anisotropy_delta",
        "week4_collision_ablation",
        "week5_layered_ablation",
    ):
        (assets_dir / f"{stem}.png").write_text("png")
        (assets_dir / f"{stem}.meta.json").write_text("{}")

    output_path = tmp_path / "week6_figure_manifest.json"
    cmd = [
        "python3.11",
        "scripts/build_week6_figure_manifest.py",
        "--results-dir",
        str(results_dir),
        "--assets-dir",
        str(assets_dir),
        "--output-json",
        str(output_path),
        "--fail-on-missing-run-ids",
    ]
    subprocess.check_call(cmd)

    with output_path.open() as f:
        payload = json.load(f)
    assert payload["all_figures_have_source_run_ids"] is True
    assert payload["n_figures"] == 3
    assert payload["n_figures_with_source_run_ids"] == 3
