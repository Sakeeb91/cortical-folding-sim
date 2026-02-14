# Week 7 Animation Comparison Pack + Results Index

## Scope Delivered

1. Added deterministic baseline-vs-improved comparison animation generation from one pipeline.
2. Exported GIF and MP4 variants for the same comparison run.
3. Added `docs/results_index.md` mapping planned-paper claims to linked artifacts.
4. Added hardened Week 7 validation covering reproducibility, matrix, regression, runtime, and CI parity checks.

## Main Additions

1. Comparison animation support in visualization utilities:
   - `src/cortical_folding/viz.py`
2. New Week 7 scripts:
   - `scripts/generate_week7_comparison_animation.py`
   - `scripts/build_results_index.py`
   - `scripts/build_week7_animation_manifest.py`
   - `scripts/regenerate_week7_animation_pack.py`
   - `scripts/validate_week7_hardened.py`
3. Week 7 documentation and index outputs:
   - `docs/week7_animation_pack.md`
   - `docs/results_index.md`

## Acceptance Snapshot

From `results/week7_animation_pack_summary.json` and `results/week7_hardened_validation.json`:

1. Week 7 gate 1 (animation outputs regenerate deterministically from documented commands): `pass`.
2. Week 7 gate 2 (each key claim has at least one linked artifact): `pass`.
3. Reproducibility check (same seed, repeated Week 7 flow): `pass` (`mismatches=[]`, `atol=1e-6`).
4. Matrix check (3 seeds x 2 parameter settings): `pass` (`stability_rate=1.0`, no failures).
5. Regression check vs Week 6 baseline artifact: `pass` (all checks marked acceptable).
6. Runtime budget check vs Week 6 path: `pass` (`runtime_ratio_week7_over_week6=0.9767` <= `1.25`).
7. CI parity checks (`run_validation_quick.sh`, `run_validation_full.sh`): `pass`.

## Artifacts

1. `docs/assets/week7_baseline_vs_improved.gif`
2. `docs/assets/week7_baseline_vs_improved.mp4`
3. `docs/assets/week7_baseline_vs_improved.meta.json`
4. `docs/assets/week7_animation_manifest.json`
5. `docs/results_index.md`
6. `results/week7_animation_comparison_summary.json`
7. `results/week7_animation_pack_summary.json`
8. `results/week7_results_index_summary.json`
9. `results/week7_hardened_validation.json`
10. `results/week7_matrix_check.csv`
