# Cortical Folding Research Plan

## Goal
Produce a small but publishable research paper that demonstrates:

1. A stable differentiable forward simulator for cortical folding.
2. Inverse recovery of synthetic growth fields.
3. Clear limitations and a concrete roadmap to real-data validation.

## Scope and Positioning

- Primary paper type: methods + synthetic validation.
- Secondary contribution: small real-data pilot (optional for first submission).
- Keep claims limited to what is experimentally verified.

## Research Questions

1. Forward: How do stiffness and growth parameters affect folding morphology?
2. Inverse: Can the differentiable pipeline recover spatial growth patterns from final folded geometry?
3. Robustness: How stable and reproducible are outcomes across parameter sweeps and seeds?

## Metrics (Pre-Registered)

1. Gyrification index (GI).
2. Surface area ratio (final area / initial area).
3. Curvature statistics (mean, std, max absolute mean curvature).
4. Vertex reconstruction error (inverse runs).
5. Growth recovery quality (MAE and correlation between true and recovered growth).
6. Stability rate (fraction of runs without numerical failure).

## Execution Plan

## Phase 1: Reproducible Forward Benchmark (Week 1)

- Build a script for deterministic parameter sweeps.
- Log per-run metrics to CSV and aggregate summary JSON.
- Generate baseline plots for GI and curvature trends.

Deliverables:

1. `scripts/run_forward_sweep.py`
2. `results/forward_sweep.csv` (or timestamped equivalent)
3. `results/forward_sweep_summary.json`

## Phase 2: Inverse Recovery Benchmark (Week 1-2)

- Benchmark recovery on synthetic targets with known growth maps.
- Run multiple seeds and report mean +/- std.
- Include recovery failure cases.

Deliverables:

1. Inverse benchmark script with metrics logging.
2. Growth recovery table and representative figures.

## Phase 3: Robustness Upgrades (Week 2-4)

- Improve collision handling and monitoring.
- Add stress/stability diagnostics.
- Perform ablation study of solver choices and parameters.

Deliverables:

1. Ablation table.
2. Stability plots.
3. Updated reproducibility report.

## Phase 4: Real-Data Pilot (Week 4-8)

- Add minimal MRI surface ingestion pipeline.
- Fit a small cohort and compare inferred growth patterns.
- Restrict claims to feasibility.

Deliverables:

1. Data preprocessing notes and reproducible scripts.
2. Pilot results with uncertainty and caveats.

## Paper Outline

1. Abstract.
2. Introduction.
3. Methods.
4. Experimental setup.
5. Results.
6. Limitations and future work.
7. Conclusion.

## Immediate Actions (Start Now)

1. Implement forward sweep benchmark script and run baseline results.
2. Add simple plotting utility for sweep outputs.
3. Draft figures and tables for forward section.

## Acceptance Criteria for This Repo Iteration

1. One-command forward sweep producing machine-readable outputs.
2. Reproducible run configs recorded in outputs.
3. At least one table-ready result file for paper drafting.
