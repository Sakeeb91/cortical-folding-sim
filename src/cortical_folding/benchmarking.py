"""Utility helpers for benchmark reproducibility and reporting."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def canonical_json(data) -> str:
    """Serialize JSON deterministically for reproducible hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def config_hash(data) -> str:
    """Return stable SHA-256 hash of a JSON-serializable object."""
    payload = canonical_json(data).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_git_commit(workdir: str | None = None) -> str:
    """Return short git commit hash if available, else 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=workdir,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def is_gi_plausible(gi: float, min_gi: float, max_gi: float) -> bool:
    """Return whether GI lies inside configured plausibility bounds."""
    return min_gi <= gi <= max_gi


def load_grid_config(path: str | Path) -> list[dict]:
    """Load a forward sweep configuration grid from JSON."""
    with Path(path).open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Grid config must be a JSON list of run configurations.")
    return data
