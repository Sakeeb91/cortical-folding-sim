# Cortical Folding Plan (Physics + Validation + Visualization)

## Objective
Upgrade the project into a stronger research-grade simulator by combining:

1. Physics realism upgrades.
2. A formal validation framework.
3. Reproducible, paper-ready visualization outputs.

## Scope

1. Keep this cycle synthetic-data-first (no clinical claims).
2. Prioritize trustworthiness and reproducibility over feature count.
3. Deliver artifacts that are immediately usable in a paper draft.

## Out of Scope (This Cycle)

1. Clinical deployment or diagnostic claims.
2. Large-scale hospital data integration.
3. Production-grade GUI or cloud platforming.

## Success Criteria

1. Stability rate >= 95% on the benchmark grid.
2. Improved realism model beats baseline on at least two morphology metrics.
3. One-command regeneration of main figures and animations.
4. Every reported metric has a saved config + summary artifact.

## Program Cadence

1. Weekly planning lock on Monday with a frozen task list.
2. Mid-week checkpoint on Wednesday against acceptance gates.
3. Weekly review on Friday with artifacts pushed and documented.
4. Any task that misses its gate gets moved to the risk log with a fallback decision.

## Workstream A: Physics Realism

### A1. Anisotropic/Directional Growth

1. Add face-local directional growth controls.
2. Support axis-biased growth in synthetic generators.
3. Add toggles to compare isotropic vs anisotropic behavior.

Deliverables:

1. Solver parameter additions in `src/cortical_folding/solver.py`.
2. Synthetic scenario generators in `src/cortical_folding/synthetic.py`.
3. A/B comparison config files for benchmark runs.

### A2. Collision/Contact Quality

1. Replace or augment sampled collision checks with spatial hash neighborhood checks.
2. Keep deterministic fallback mode for reproducibility.
3. Track collision-force share in diagnostics.

Deliverables:

1. Updated contact logic in `src/cortical_folding/constraints.py`.
2. Collision diagnostics in benchmark outputs.

### A3. Layered Mechanics Approximation

1. Add an optional two-layer approximation (outer cortex and inner substrate coupling).
2. Expose coupling stiffness and differential growth controls.
3. Document known simplifications and limits.

Deliverables:

1. Optional layered mode in `src/cortical_folding/solver.py`.
2. Targeted tests and small ablations.

## Workstream B: Validation Framework

### B1. Benchmark Protocol

Metrics:

1. Stability rate.
2. GI and GI plausibility flags.
3. Area ratio.
4. Curvature percentiles (p50/p90/p99).
5. Runtime and memory snapshot.
6. Skull penetration statistics.

Deliverables:

1. Config-driven runners in `scripts/`.
2. Standardized CSV/JSON schema under `results/`.
3. Config hash + manifest for each run.

### B2. Quality Gates

1. Add thresholds that fail runs with instability symptoms.
2. Add CI checks for benchmark regressions.
3. Add seeded reproducibility checks.

Deliverables:

1. Additional tests in `tests/`.
2. CI-friendly benchmark command set in README.

### B3. Ablations

1. Clipping thresholds ablation.
2. Collision sampling/neighbor-density ablation.
3. Time-step and damping sensitivity ablation.

Deliverables:

1. Ablation tables and summary charts in `docs/assets/`.

## Workstream C: Visualization and Communication

### C1. Figure Pipeline

1. Build deterministic scripts that regenerate all paper figures.
2. Standardize color scales and axis conventions across figures.
3. Save plot metadata (source run IDs) with each figure.

Deliverables:

1. Figure scripts in `scripts/`.
2. Published assets in `docs/assets/`.

### C2. Animation Pipeline

1. Extend animation outputs to comparison panels (baseline vs improved model).
2. Export GIF and MP4 from the same source pipeline.
3. Add script presets for short and long demos.

Deliverables:

1. Extended animation workflow in `src/cortical_folding/viz.py`.
2. Comparison animation script in `scripts/`.

### C3. Paper-Ready Outputs

1. Add ready-to-paste figure captions and table templates.
2. Add a results index that maps each claim to artifacts.
3. Keep claims tied to measured evidence.

Deliverables:

1. `docs/results_index.md`.
2. Updated `README.md` research results section.

## Detailed 8-Week Execution Plan

### Week 1: Baseline Protocol Lock

Tasks:

1. Freeze baseline forward sweep config grid.
2. Add curvature percentiles and skull penetration stats to output schema.
3. Add GI plausibility flags to summary reports.
4. Add run manifest fields (config hash, seed, git commit).

Deliverables:

1. Updated benchmark script and schema docs.
2. Baseline `results/` artifacts with manifest.
3. README benchmark command section updated.

Acceptance Gates:

1. Baseline run completes with no schema drift.
2. All runs include GI, area ratio, curvature p50/p90/p99, penetration stats.
3. Re-running with same seed reproduces numerically identical summary metrics.

### Week 2: Validation Quality Gates + CI Hooks

Tasks:

1. Implement benchmark quality gate checks.
2. Add seeded reproducibility test job.
3. Add instability fail-fast checks (NaN/Inf, exploding displacement).
4. Add CI command presets for quick and full validation.

Deliverables:

1. Test updates in `tests/`.
2. CI-facing command docs.
3. Gate report in `results/validation_gate_report.json`.

Acceptance Gates:

1. CI quick suite passes on default branch.
2. Quality gate failure emits actionable error messages.
3. At least one intentional failure case is detected by gates.

### Week 3: Anisotropic Growth Prototype

Tasks:

1. Add directional growth controls to solver/synthetic pipeline.
2. Add isotropic vs anisotropic config toggles.
3. Add minimal tests covering directional behavior.

Deliverables:

1. New anisotropic config presets.
2. Comparative run outputs for baseline vs anisotropic.

Acceptance Gates:

1. Anisotropic mode runs with stability >= 90% initially.
2. At least one morphology metric differs significantly from isotropic baseline.

### Week 4: Contact/Collision Upgrade

Tasks:

1. Add spatial-hash neighborhood collision path.
2. Keep deterministic fallback mode.
3. Add collision-force contribution diagnostics.

Deliverables:

1. Updated `constraints.py` collision logic.
2. Collision ablation metrics in results.

Acceptance Gates:

1. Collision-enabled runs reduce penetration outliers vs baseline.
2. Runtime overhead remains within budget target.

### Week 5: Layered Approximation + Core Ablations

Tasks:

1. Add optional two-layer approximation mode.
2. Run ablations for clipping thresholds, damping, and dt.
3. Document simplifications and known failure modes.

Deliverables:

1. Layered mode toggles and tests.
2. Ablation tables for key parameters.

Acceptance Gates:

1. Layered mode stable on reduced benchmark grid.
2. Ablation table identifies at least one robust parameter region.

### Week 6: Figure Pipeline Standardization

Tasks:

1. Build deterministic scripts for all core figures.
2. Standardize styling and axis conventions.
3. Attach source metadata to figure outputs.

Deliverables:

1. Figure scripts under `scripts/`.
2. Versioned assets under `docs/assets/`.

Acceptance Gates:

1. One-command figure regeneration succeeds from clean checkout.
2. All figures include mapped source run IDs.

### Week 7: Animation Comparison Pack + Results Index

Tasks:

1. Generate baseline vs improved comparison animations.
2. Export GIF and MP4 variants from same pipeline.
3. Create `docs/results_index.md` mapping claims to artifacts.

Deliverables:

1. Comparison animations in `docs/assets/`.
2. Results index document.

Acceptance Gates:

1. Animation outputs regenerate deterministically from documented commands.
2. Each key claim in planned paper has at least one linked artifact.

### Week 8: Final Packaging for Writing

Tasks:

1. Freeze benchmark artifacts and methods settings.
2. Compile final tables and figure captions.
3. Prepare methods/results draft packet.

Deliverables:

1. Frozen artifact bundle with config manifests.
2. Finalized README research-results section.
3. Submission-ready reproducibility command list.

Acceptance Gates:

1. All top-level success criteria are met or explicitly waived with rationale.
2. Draft packet is complete enough to start paper writing without new experiments.

## Task-Level Acceptance Thresholds

1. Stability threshold:
   Target >= 95% stable runs on the main benchmark grid.
2. Penetration threshold:
   `outside_skull_frac` p95 <= baseline p95 by Week 5 improved model.
3. Reproducibility threshold:
   Metric drift across repeat seeded runs <= 1e-6 for deterministic paths.
4. Runtime threshold:
   Quick benchmark <= 30 minutes wall-clock on target machine.
5. Artifact completeness threshold:
   Every figure/table maps to a result file + config hash + commit.

## Compute and Runtime Budget

1. Development machine target:
   Local Apple Silicon / CPU-first workflow with optional accelerator support.
2. Weekly compute budget:
   10-15 hours total run time, split into:
   3 hours baseline validation, 8 hours ablations, 4 hours figure/animation generation.
3. Benchmark tiers:
   Quick tier: <= 30 minutes.
   Full tier: <= 6 hours.
4. Storage budget:
   Keep tracked assets <= 1.5 GB in repo; archive larger artifacts externally.

## Risk Register and Fallback Plan

1. Risk: Layered mode unstable.
   Impact: High.
   Fallback: Keep layered mode behind experimental flag and publish anisotropic + contact improvements only.
2. Risk: Collision upgrade too slow.
   Impact: Medium.
   Fallback: Use hybrid mode (spatial hash on sparse cadence, sampled mode on other steps).
3. Risk: Metrics show weak realism gains.
   Impact: High.
   Fallback: Reframe contribution as robustness and reproducibility framework with transparent negative results.
4. Risk: Figure pipeline drifts from metrics schema.
   Impact: Medium.
   Fallback: Version schema and enforce compatibility checks before plotting.
5. Risk: Timeline slips.
   Impact: Medium.
   Fallback: Prioritize Workstream B then C; defer layered mechanics to next cycle.

## Ownership and Priority Rules

1. Primary owner:
   Project maintainer (single-thread ownership assumed unless reassigned).
2. Priority order when schedule is constrained:
   Validation framework > visualization reproducibility > physics extensions.
3. Must-not-slip items:
   Benchmark schema stability, quality gates, and reproducible figure commands.
4. Nice-to-have items:
   Full layered mechanics and advanced collision variants.

## Definition of Done (End of Week 8)

1. Stable, reproducible benchmark pipeline with enforced quality gates.
2. At least one validated realism upgrade with measured gains.
3. Complete figure + animation pack reproducible from documented commands.
4. Results index linking claims to artifacts and configs.
5. Paper drafting can begin without additional engineering work.
