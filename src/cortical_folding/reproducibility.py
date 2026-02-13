"""Helpers for seeded reproducibility checks."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def compare_summary_metrics(a: dict, b: dict, keys: list[str], atol: float = 1e-6) -> list[str]:
    """Return mismatch messages for selected numeric summary keys."""
    mismatches = []
    for key in keys:
        va = float(a.get(key, float("nan")))
        vb = float(b.get(key, float("nan")))
        if abs(va - vb) > atol:
            mismatches.append(
                f"Mismatch key={key}: first={va:.9f} second={vb:.9f} tol={atol:.1e}"
            )
    return mismatches
