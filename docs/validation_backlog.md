# Validation Backlog

This checklist tracks concrete work needed to turn the simulator into a stronger research-grade system.
Each checkbox is intentionally narrow so progress can be tracked with high granularity.
 - [ ] Add deterministic seed controls across all runnable scripts.
 - [ ] Add solver stability KPI logging (stability rate, divergence rate).
 - [ ] Add a NaN/Inf incident counter to per-run diagnostics output.
 - [ ] Export max displacement histograms for forward sweeps.
 - [ ] Report curvature percentiles (p50/p90/p99) in benchmark outputs.
 - [ ] Report skull penetration percentiles per run.
 - [ ] Track collision-force contribution relative to total force norm.
 - [ ] Write a reproducibility manifest for each parameter sweep run.
 - [ ] Hash experiment configs and store hash with metrics artifacts.
 - [ ] Add growth-field recovery MAE metric for inverse benchmarks.
