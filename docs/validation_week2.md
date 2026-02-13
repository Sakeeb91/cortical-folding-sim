# Week 2 Validation Notes

This document captures the Week 2 validation framework additions.

## What Was Added

1. Quality gate evaluation with machine-readable reports.
2. Fail-fast instability checks in forward sweep runs.
3. Seeded reproducibility checker for summary metrics.
4. GitHub Actions jobs for quick validation and reproducibility checks.

## Core Artifacts

1. `results/validation_gate_report.json`
2. `results/validation_gate_report_failcase.json`
3. `configs/validation_gates_default.json`
4. `configs/validation_gates_failcase.json`

## Gate Semantics

1. `stability_rate`: minimum threshold.
2. `gi_plausible_rate`: minimum threshold.
3. `outside_skull_frac_p95`: maximum threshold.
4. `skull_penetration_p95`: maximum threshold.
5. `disp_p95`: maximum threshold.

## Acceptance Mapping

1. Quick CI must pass with default gates.
2. Intentional strict failcase must fail with actionable messages.
3. Seeded repeated runs must satisfy reproducibility tolerance.
