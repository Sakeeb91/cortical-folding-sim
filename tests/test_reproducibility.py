"""Tests for seeded reproducibility helpers."""

from cortical_folding.reproducibility import compare_summary_metrics


def test_compare_summary_metrics_passes_within_tolerance():
    a = {"stability_rate": 1.0, "gi_mean": 1.5}
    b = {"stability_rate": 1.0 + 1e-8, "gi_mean": 1.5 + 1e-8}
    mismatches = compare_summary_metrics(a, b, ["stability_rate", "gi_mean"], atol=1e-6)
    assert mismatches == []


def test_compare_summary_metrics_reports_mismatch():
    a = {"stability_rate": 1.0, "gi_mean": 1.5}
    b = {"stability_rate": 0.7, "gi_mean": 2.2}
    mismatches = compare_summary_metrics(a, b, ["stability_rate", "gi_mean"], atol=1e-6)
    assert len(mismatches) == 2
    assert "stability_rate" in mismatches[0]
