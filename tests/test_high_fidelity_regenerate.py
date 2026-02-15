"""Tests for high-fidelity one-command regeneration driver."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_regenerate_high_fidelity_package_dry_run(tmp_path: Path):
    output_json = tmp_path / "package_summary.json"
    cmd = [
        "python3.11",
        "scripts/regenerate_high_fidelity_package.py",
        "--dry-run",
        "--output-json",
        str(output_json),
    ]
    subprocess.check_call(cmd)

    with output_json.open() as f:
        payload = json.load(f)
    assert payload["dry_run"] is True
    assert len(payload["steps"]) == 3
    assert payload["passed"] is True
