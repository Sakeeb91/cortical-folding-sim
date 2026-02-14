# Week 6 Figure Pipeline Standardization

## Scope Delivered

1. Added one-command deterministic regeneration for Week 3/4/5 core figures.
2. Standardized style and axis conventions across week figure scripts.
3. Attached per-figure sidecar metadata and a consolidated Week 6 figure manifest.
4. Added hardened Week 6 validation covering reproducibility, matrix, regression, runtime, and CI parity checks.

## Main Additions

1. Shared style helpers:
   - `src/cortical_folding/figure_style.py`
2. Updated figure scripts with standardized style + sidecar metadata:
   - `scripts/plot_week3_anisotropy.py`
   - `scripts/plot_week4_collision.py`
   - `scripts/plot_week5_layered_ablation.py`
3. New Week 6 orchestration and validation scripts:
   - `scripts/regenerate_week6_figures.py`
   - `scripts/build_week6_figure_manifest.py`
   - `scripts/validate_week6_hardened.py`

## Acceptance Snapshot

From `results/week6_figure_pipeline_summary.json`, `docs/assets/week6_figure_manifest.json`, and `results/week6_hardened_validation.json`:

1. Week 6 gate 1 (one-command figure regeneration from clean checkout): `pass`.
2. Week 6 gate 2 (all figures include mapped source run IDs): `pass`.
3. Reproducibility check (same seed, repeated Week 6 pipeline): `pass`.
4. Matrix check (3 seeds x 2 parameter settings): `pass` (`stability_rate=1.0`).
5. Regression check vs Week 5 baseline artifact: `pass` (all critical checks marked acceptable).
6. Runtime budget check vs Week 5 baseline path: `pass` (`runtime_ratio_week6_over_week5=1.0095` <= `1.35`).
7. CI parity checks (`run_validation_quick.sh`, `run_validation_full.sh`): `pass`.

## Artifacts

1. `docs/assets/week3_anisotropy_delta.png`
2. `docs/assets/week4_collision_ablation.png`
3. `docs/assets/week5_layered_ablation.png`
4. `docs/assets/week3_anisotropy_delta.meta.json`
5. `docs/assets/week4_collision_ablation.meta.json`
6. `docs/assets/week5_layered_ablation.meta.json`
7. `docs/assets/week6_figure_manifest.json`
8. `results/week6_figure_pipeline_summary.json`
9. `results/week6_hardened_validation.json`
10. `results/week6_matrix_check.csv`
