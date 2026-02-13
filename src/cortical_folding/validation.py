"""Validation helpers for benchmark quality gates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GateThresholds:
    min_stability_rate: float
    min_gi_plausible_rate: float
    max_outside_skull_frac_p95: float
    max_skull_penetration_p95: float
    max_disp_p95: float


@dataclass(frozen=True)
class GateMetrics:
    stability_rate: float
    gi_plausible_rate: float
    outside_skull_frac_p95: float
    skull_penetration_p95: float
    disp_p95: float


def load_gate_thresholds(path: str | Path) -> GateThresholds:
    """Load gate thresholds from JSON."""
    with Path(path).open() as f:
        data = json.load(f)
    return GateThresholds(**data)


def compute_gate_metrics(rows: list[dict], summary: dict) -> GateMetrics:
    """Compute quality-gate metrics from CSV rows and summary payload."""
    stable_rows = [r for r in rows if int(r["stable"]) == 1]
    if stable_rows:
        outside = [float(r["outside_skull_frac"]) for r in stable_rows]
        skull_p = [float(r["skull_penetration_p95"]) for r in stable_rows]
        disp_p = [float(r["disp_p95"]) for r in stable_rows]
        outside_p95 = float(np.percentile(outside, 95))
        skull_pen_p95 = float(np.percentile(skull_p, 95))
        disp_p95 = float(np.percentile(disp_p, 95))
    else:
        outside_p95 = math.inf
        skull_pen_p95 = math.inf
        disp_p95 = math.inf

    return GateMetrics(
        stability_rate=float(summary.get("stability_rate", 0.0)),
        gi_plausible_rate=float(summary.get("gi_plausible_rate", 0.0)),
        outside_skull_frac_p95=outside_p95,
        skull_penetration_p95=skull_pen_p95,
        disp_p95=disp_p95,
    )
