"""Tests for high-fidelity publication render script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_publication_render_dry_run_writes_summary_and_manifest(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            [
                {"label": "publication_baseline", "growth_mode": "uniform", "uniform_rate": 0.2},
                {
                    "label": "publication_high_fidelity",
                    "growth_mode": "uniform",
                    "uniform_rate": 0.2,
                    "simulation_mode": "high_fidelity",
                },
            ]
        )
    )
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    gif_path = tmp_path / "out.gif"
    mp4_path = tmp_path / "out.mp4"

    cmd = [
        "python3.11",
        "scripts/generate_high_fidelity_publication_render.py",
        "--dry-run",
        "--config-path",
        str(config_path),
        "--output-summary",
        str(summary_path),
        "--output-manifest",
        str(manifest_path),
        "--output-gif",
        str(gif_path),
        "--output-mp4",
        str(mp4_path),
    ]
    subprocess.check_call(cmd)

    with summary_path.open() as f:
        payload = json.load(f)
    assert payload["dry_run"] is True
    assert payload["acceptance_outputs_exist"] is False
    assert manifest_path.exists()
