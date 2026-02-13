# Week 4 Contact/Collision Upgrade

## Scope Delivered

1. Added a spatial-hash neighborhood collision path in `src/cortical_folding/constraints.py`.
2. Preserved deterministic sampled fallback mode for reproducible collision behavior.
3. Added collision-force and collision-overlap diagnostics to benchmark outputs.
4. Added Week 4 collision ablation config, runner, plotting script, and artifacts.

## Main Additions

1. `SimParams` collision controls now include:
   - `self_collision_use_spatial_hash`
   - `self_collision_hash_cell_size`
   - `self_collision_hash_neighbor_window`
   - `self_collision_deterministic_fallback`
   - `self_collision_fallback_n_sample`
2. Forward sweep rows now include:
   - `collision_force_l2`, `total_force_l2`, `collision_force_share`
   - `collision_overlap_mean`, `collision_overlap_p95`, `collision_overlap_max`
   - `collision_overlap_count`, `collision_overlap_frac`
3. Sweep summary now includes collision diagnostic aggregates.

## Acceptance Snapshot

From `results/week4_collision_comparison.json`:

1. Stability: `100%` (3/3 stable).
2. Penetration outlier reduction (overlap-count proxy):
   - baseline: `73`
   - sampled: `73`
   - spatial-hash: `58`
   - reduction vs baseline: `15`
3. Runtime overhead ratio (spatial-hash / sampled): `1.5502`
4. Runtime budget target: `1.6`
5. Week 4 gate result: `acceptance_week4_passed=true`

## Artifacts

1. `configs/week4_collision_ablation.json`
2. `results/week4_collision_ablation.csv`
3. `results/week4_collision_ablation_summary.json`
4. `results/week4_collision_ablation_manifest.json`
5. `results/week4_collision_comparison.json`
6. `docs/assets/week4_collision_ablation.png`
