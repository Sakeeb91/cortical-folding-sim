"""Validation helpers for benchmark quality gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateThresholds:
    min_stability_rate: float
    min_gi_plausible_rate: float
    max_outside_skull_frac_p95: float
    max_skull_penetration_p95: float
    max_disp_p95: float


def load_gate_thresholds(path: str | Path) -> GateThresholds:
    """Load gate thresholds from JSON."""
    with Path(path).open() as f:
        data = json.load(f)
    return GateThresholds(**data)
