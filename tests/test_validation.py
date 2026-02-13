"""Tests for benchmark validation gate helpers."""

from cortical_folding.validation import (
    GateThresholds,
    build_gate_report,
    compute_gate_metrics,
    evaluate_gate_checks,
)


def _sample_rows():
    return [
        {
            "stable": 1,
            "outside_skull_frac": 0.1,
            "skull_penetration_p95": 0.05,
            "disp_p95": 0.4,
        },
        {
            "stable": 1,
            "outside_skull_frac": 0.2,
            "skull_penetration_p95": 0.06,
            "disp_p95": 0.5,
        },
    ]


def test_validation_checks_pass_for_good_metrics():
    rows = _sample_rows()
    summary = {"stability_rate": 1.0, "gi_plausible_rate": 0.8}
    thresholds = GateThresholds(
        min_stability_rate=0.95,
        min_gi_plausible_rate=0.5,
        max_outside_skull_frac_p95=0.4,
        max_skull_penetration_p95=0.2,
        max_disp_p95=0.8,
    )
    metrics = compute_gate_metrics(rows, summary)
    checks = evaluate_gate_checks(metrics, thresholds)
    assert all(c.passed for c in checks)


def test_validation_report_flags_failures():
    rows = _sample_rows()
    summary = {"stability_rate": 0.6, "gi_plausible_rate": 0.2}
    thresholds = GateThresholds(
        min_stability_rate=0.95,
        min_gi_plausible_rate=0.5,
        max_outside_skull_frac_p95=0.05,
        max_skull_penetration_p95=0.01,
        max_disp_p95=0.1,
    )
    checks = evaluate_gate_checks(compute_gate_metrics(rows, summary), thresholds)
    report = build_gate_report(rows=rows, summary=summary, thresholds=thresholds, checks=checks)
    assert report["passed"] is False
    assert report["n_failures"] >= 1
