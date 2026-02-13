"""Validation helpers for benchmark quality gates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class GateCheckResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    comparator: str
    message: str


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


def evaluate_gate_checks(
    metrics: GateMetrics, thresholds: GateThresholds
) -> list[GateCheckResult]:
    """Evaluate each gate and return detailed pass/fail results."""
    checks = [
        (
            "stability_rate",
            metrics.stability_rate,
            thresholds.min_stability_rate,
            ">=",
            metrics.stability_rate >= thresholds.min_stability_rate,
        ),
        (
            "gi_plausible_rate",
            metrics.gi_plausible_rate,
            thresholds.min_gi_plausible_rate,
            ">=",
            metrics.gi_plausible_rate >= thresholds.min_gi_plausible_rate,
        ),
        (
            "outside_skull_frac_p95",
            metrics.outside_skull_frac_p95,
            thresholds.max_outside_skull_frac_p95,
            "<=",
            metrics.outside_skull_frac_p95 <= thresholds.max_outside_skull_frac_p95,
        ),
        (
            "skull_penetration_p95",
            metrics.skull_penetration_p95,
            thresholds.max_skull_penetration_p95,
            "<=",
            metrics.skull_penetration_p95 <= thresholds.max_skull_penetration_p95,
        ),
        (
            "disp_p95",
            metrics.disp_p95,
            thresholds.max_disp_p95,
            "<=",
            metrics.disp_p95 <= thresholds.max_disp_p95,
        ),
    ]

    results = []
    for name, observed, threshold, comparator, passed in checks:
        verdict = "PASS" if passed else "FAIL"
        msg = (
            f"{verdict}: {name} observed={observed:.6f} "
            f"{comparator} threshold={threshold:.6f}"
        )
        results.append(
            GateCheckResult(
                name=name,
                passed=passed,
                observed=float(observed),
                threshold=float(threshold),
                comparator=comparator,
                message=msg,
            )
        )
    return results


def build_gate_report(
    *,
    rows: list[dict],
    summary: dict,
    thresholds: GateThresholds,
    checks: list[GateCheckResult],
    metadata: dict | None = None,
) -> dict:
    """Build normalized gate report payload for JSON export."""
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "n_failures": int(sum(0 if c.passed else 1 for c in checks)),
        "passed": bool(all(c.passed for c in checks)),
        "thresholds": {
            "min_stability_rate": thresholds.min_stability_rate,
            "min_gi_plausible_rate": thresholds.min_gi_plausible_rate,
            "max_outside_skull_frac_p95": thresholds.max_outside_skull_frac_p95,
            "max_skull_penetration_p95": thresholds.max_skull_penetration_p95,
            "max_disp_p95": thresholds.max_disp_p95,
        },
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "observed": c.observed,
                "threshold": c.threshold,
                "comparator": c.comparator,
                "message": c.message,
            }
            for c in checks
        ],
        "summary_ref": {
            "stability_rate": float(summary.get("stability_rate", 0.0)),
            "gi_plausible_rate": float(summary.get("gi_plausible_rate", 0.0)),
            "git_commit": summary.get("git_commit", "unknown"),
            "seed": summary.get("seed", "unknown"),
            "sweep_config_hash": summary.get("sweep_config_hash", "unknown"),
        },
    }
    if metadata:
        payload["metadata"] = metadata
    return payload
