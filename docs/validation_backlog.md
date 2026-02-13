# Validation Backlog

This checklist tracks concrete work needed to turn the simulator into a stronger research-grade system.
Each checkbox is intentionally narrow so progress can be tracked with high granularity.
 - [ ] Add deterministic seed controls across all runnable scripts.
 - [ ] Add solver stability KPI logging (stability rate, divergence rate).
 - [ ] Add a NaN/Inf incident counter to per-run diagnostics output.
 - [ ] Export max displacement histograms for forward sweeps.
 - [x] Report curvature percentiles (p50/p90/p99) in benchmark outputs.
 - [x] Report skull penetration percentiles per run.
 - [x] Track collision-force contribution relative to total force norm.
 - [x] Write a reproducibility manifest for each parameter sweep run.
 - [x] Hash experiment configs and store hash with metrics artifacts.
 - [ ] Add growth-field recovery MAE metric for inverse benchmarks.
 - [ ] Add growth-field recovery Pearson/Spearman correlation metrics.
 - [ ] Add a multi-seed inverse benchmark runner with aggregate summary.
 - [ ] Add confidence interval utilities for reported scalar metrics.
 - [ ] Add identifiability stress-test protocol for ambiguous growth fields.
 - [ ] Add ablation plan for force/velocity/displacement clipping thresholds.
 - [x] Add ablation plan for collision sampling density and min distance.
 - [ ] Add runtime profiling checklist for step function performance.
 - [ ] Add memory profiling checklist for long-trajectory simulations.
 - [ ] Add biologically plausible GI range checks to evaluation reports.
 - [x] Add a quality gate that flags runs with instability symptoms.
 - [ ] Add data-ingestion interface placeholder for future MRI surfaces.
 - [ ] Add uncertainty reporting template for inverse-fit results.
 - [ ] Add publication reproducibility checklist for code and artifacts.
