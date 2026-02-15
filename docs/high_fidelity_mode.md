# High-Fidelity Mode

## What Changed

1. Added a configurable `high_fidelity` simulation profile with deterministic adaptive substepping.
2. Added stricter numerical safety controls (tighter force/acceleration/velocity/displacement bounds).
3. Added a hybrid collision path (`spatial_hash + deterministic sampled blend`) for more robust contact response.
4. Added publication render pipeline with:
   - 1080p+ output defaults (`1920x1080`)
   - smooth camera easing
   - shaded lighting
   - supersampling controls
   - optional live metric overlays (`GI`, `disp_p95`, `outside_skull_frac`)
   - shared source pipeline exporting both GIF and MP4
5. Added hardened high-fidelity validator with reproducibility, matrix, regression, runtime-budget, and CI-parity checks.

## Design Assumptions

1. Determinism remains mandatory for seeded paths.
2. High-fidelity mode may trade runtime for stability/quality.
3. Collision handling remains approximate and mesh-surface based (not full volumetric contact).
4. Synthetic benchmark scope remains unchanged.

## Known Limits

1. Adaptive substepping uses deterministic heuristic estimation, not full local truncation error control.
2. Shading/lighting is renderer-level visual enhancement, not physically based rendering.
3. Supersampling increases export cost and output size.
4. Clinical validity is out of scope.

## Commands

Run high-fidelity sweep:

```bash
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py \
  --mode high_fidelity \
  --config-path configs/high_fidelity_forward_sweep.json \
  --n-steps 120 \
  --output-csv results/high_fidelity/forward_sweep.csv \
  --output-summary results/high_fidelity/forward_sweep_summary.json \
  --output-manifest results/high_fidelity/forward_sweep_manifest.json
```

Generate publication comparison render (GIF + MP4 from one pipeline):

```bash
MPLBACKEND=Agg python3.11 scripts/generate_high_fidelity_publication_render.py \
  --config-path configs/high_fidelity_publication_render.json \
  --n-steps 180 \
  --with-metric-overlays \
  --output-gif docs/assets/high_fidelity/publication_comparison.gif \
  --output-mp4 docs/assets/high_fidelity/publication_comparison.mp4 \
  --output-summary results/high_fidelity/publication_render_summary.json \
  --output-manifest results/high_fidelity/publication_render_manifest.json
```

Run hardened validation:

```bash
python3.11 scripts/validate_high_fidelity.py \
  --output-dir results/high_fidelity \
  --n-steps 120
```

One-command regeneration:

```bash
MPLBACKEND=Agg python3.11 scripts/regenerate_high_fidelity_package.py \
  --n-steps 120 \
  --render-steps 180 \
  --output-json results/high_fidelity/package_summary.json
```
