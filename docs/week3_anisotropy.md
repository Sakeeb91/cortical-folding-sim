# Week 3 Anisotropic Growth Prototype

## Scope Delivered

1. Directional anisotropic growth controls integrated into solver.
2. Isotropic vs anisotropic toggles added to sweep/demo scripts.
3. Minimal anisotropy tests and A/B comparison artifacts generated.

## Main Additions

1. `SimParams` now supports `anisotropy_strength` and `anisotropy_axis`.
2. Solver can accept `face_anisotropy` and apply directional rest-length scaling.
3. Synthetic helpers generate anisotropy fields (`none`, `uniform`, `regional`).
4. Week 3 comparison scripts:
   - `scripts/run_anisotropy_comparison.py`
   - `scripts/plot_week3_anisotropy.py`

## Acceptance Snapshot

1. Stability across A/B config: `100%` (2/2 runs stable).
2. Morphology difference check: passed (`morphology_difference_detected=true`).
3. Example metric deltas from `results/week3_anisotropy_comparison.json`:
   - `delta_gi = 1.1605`
   - `delta_curv_p90 = -0.0863`

## Artifacts

1. `results/week3_anisotropy_ab.csv`
2. `results/week3_anisotropy_ab_summary.json`
3. `results/week3_anisotropy_ab_manifest.json`
4. `results/week3_anisotropy_comparison.json`
5. `docs/assets/week3_anisotropy_delta.png`
