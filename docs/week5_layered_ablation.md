# Week 5 Layered Approximation + Core Ablations

## Scope Delivered

1. Added optional two-layer approximation mode in the solver, configurable per run.
2. Added Week 5 ablation grid for clipping thresholds, damping, and `dt`.
3. Added hardened validation workflow with reproducibility, matrix, regression, and runtime checks.
4. Added Week 5 plotting pipeline and generated artifact.

## Main Additions

1. `SimParams` now supports:
   - `enable_two_layer_approx`
   - `two_layer_axis`, `two_layer_threshold`, `two_layer_transition_sharpness`
   - `outer_layer_growth_scale`, `inner_layer_growth_scale`, `two_layer_coupling`
2. Forward sweep now accepts per-config overrides for:
   - `dt`
   - clipping thresholds (`max_force_norm`, `max_acc_norm`, `max_velocity_norm`, `max_displacement_per_step`)
3. New Week 5 scripts:
   - `scripts/run_week5_layered_ablation.py`
   - `scripts/plot_week5_layered_ablation.py`
   - `scripts/validate_week5_hardened.py`

## Simplifications

1. Two-layer mode is a reduced approximation on a single-surface mesh (smooth axis-based face partition + growth scaling), not a full volumetric multilayer tissue model.
2. Layer coupling is represented as scalar blending of per-face growth rates instead of explicit inter-layer mechanics.
3. Seed matrix currently validates deterministic stability/metrics behavior; it does not yet introduce stochastic perturbations in growth fields.

## Known Failure Modes

1. Extreme `dt` can inflate GI variance (`layered_dt_large`) while remaining numerically stable.
2. Very tight clipping (`layered_clip_tight`) suppresses fold amplitude and can underfit morphology targets.
3. Runtime ratio checks are sensitive to machine load; runtime pass/fail should be interpreted with fixed host conditions.

## Acceptance Snapshot

From `results/week5_layered_comparison.json` and `results/week5_hardened_validation.json`:

1. Week 5 plan gate 1 (layered mode stable on reduced grid): `pass` (`reduced_grid_all_stable=true`).
2. Week 5 plan gate 2 (robust parameter region identified): `pass` (`robust_region_count=4`).
3. Reproducibility check: `pass` (config hash, key metrics, acceptance flags match at `atol=1e-6`).
4. Matrix check (3 seeds x 2 parameter settings): `pass` (stability rate `1.0`, no failures).
5. Regression check vs Week 4 baseline: `pass` (all critical metrics marked acceptable).
6. Runtime budget vs Week 4 path: `pass` (`runtime_ratio_week5_over_week4=1.1416` <= `1.35`).

## Artifacts

1. `configs/week5_layered_ablation.json`
2. `results/week5_layered_ablation.csv`
3. `results/week5_layered_ablation_summary.json`
4. `results/week5_layered_ablation_manifest.json`
5. `results/week5_layered_comparison.json`
6. `results/week5_hardened_validation.json`
7. `results/week5_matrix_check.csv`
8. `docs/assets/week5_layered_ablation.png`
