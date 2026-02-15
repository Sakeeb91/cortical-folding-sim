# Week 8 Paper Takeaways (Comprehensive)

## Executive Summary

1. The full 8-week plan completed with all Week 8 hardening gates passing, including reproducibility, stability matrix, regression protection, runtime budget, and CI parity.
2. The strongest technical result is reliability: deterministic reruns matched within `1e-6`, matrix stability was `1.0`, and top-level acceptance criteria were all satisfied without waivers.
3. The strongest modeling result is directional sensitivity plus incremental realism gains: anisotropic growth changes morphology substantially, spatial-hash contact improves overlap behavior, and layered mode improves reference metrics modestly but consistently.

## Methods (Draft-Ready Narrative)

### Modeling Approach

1. The simulator is a differentiable mechanics pipeline that couples constrained growth with elastic and bending responses on a cortical mesh.
2. Physical constraints include skull interaction and self-contact handling, with a spatial-hash collision path and deterministic fallback logic.
3. Growth behavior is tested under isotropic and directional (anisotropic) controls, and an optional layered approximation is used to mimic differential behavior between an outer region and substrate-like coupling.

### Validation Protocol

1. Benchmarking is configuration-driven and versioned through saved manifests and config hashes.
2. Core quality gates evaluate:
   - Stability rate.
   - GI plausibility rate.
   - Outside-skull fraction p95.
   - Skull-penetration p95.
   - Displacement p95.
3. Reproducibility checks rerun the same seeded flow and require deterministic metric agreement within absolute tolerance `1e-6`.
4. Robustness checks use a seed/parameter matrix (at least `3 seeds x 2 settings`) and report stability plus failure-reason counts.
5. Regression checks compare against prior-week frozen artifacts to ensure packaging changes do not degrade key metrics.
6. Runtime checks compare end-to-end pipeline runtime ratio against a budget threshold (Week 8 used `<= 1.25` vs Week 7 baseline path).

### Reproducibility and Artifact Discipline

1. Every major claim is tied to concrete artifacts under `results/` and `docs/`.
2. Figure/animation outputs include source-run mapping metadata.
3. Week 8 adds a frozen artifact bundle manifest and a methods/settings freeze payload for writing handoff.

## Results (What We Observed)

### A. Baseline Validation State

From `results/forward_sweep_summary.json` and `results/validation_gate_report.json`:

1. `stability_rate = 1.0` on `n_runs = 16`.
2. `gi_plausible_rate = 0.5625` (above threshold `0.5`).
3. All default gates pass:
   - `outside_skull_frac_p95 = 0.375389` (threshold `<= 0.4`).
   - `skull_penetration_p95 = 0.213005` (threshold `<= 0.25`).
   - `disp_p95 = 0.773080` (threshold `<= 0.8`).

Interpretation:
The pipeline is in a stable and gate-compliant operating regime for synthetic benchmark usage.

### B. Realism Upgrade Effects by Workstream

From `results/week3_anisotropy_comparison.json`, `results/week4_collision_comparison.json`, and `results/week5_layered_comparison.json`:

| Upgrade | Key Observation | Quantitative Signal |
|---|---|---|
| Anisotropic growth (Week 3) | Morphology changes materially vs isotropic baseline | `delta_gi = +1.251468`, `delta_curv_p90 = -0.449325`, `delta_disp_p95 = +0.025195`, `stability_rate = 1.0` |
| Spatial-hash collision (Week 4) | Contact outlier behavior improves vs sampled collision | `reduction_overlap_count_vs_sampled = 15`, runtime ratio `1.484094` (within set budget), acceptance flags pass |
| Layered approximation (Week 5) | Modest but consistent metric gains on reference setting | `delta_gi = +0.091064`, `delta_outside_skull_frac = -0.006231`, `delta_disp_p95 = -0.001118`, runtime ratio `1.010289`, `robust_region_count = 4` |

Interpretation:

1. Directionality in growth has a first-order effect on morphology.
2. Contact modeling changes can improve collision behavior while keeping acceptable runtime.
3. Layered mode currently behaves as an incremental robustness/quality improvement rather than a dramatic morphology shift.

### C. Week 7/8 Packaging and Reproducibility Outcomes

From `results/week7_animation_comparison_summary.json`, `results/week8_submission_packet_summary.json`, and `results/week8_hardened_validation.json`:

1. Week 7 improved-vs-baseline animation metrics are positive/acceptable:
   - `delta_gi_improved_minus_baseline = +0.091064`
   - `delta_disp_p95_improved_minus_baseline = -0.001118`
   - `delta_outside_skull_frac_improved_minus_baseline = -0.006231`
2. Week 8 top-level success criteria status:
   - `S1` stability criterion: pass.
   - `S2` realism gain criterion: pass.
   - `S3` reproducible visuals criterion: pass.
   - `S4` evidence-linking criterion: pass.
3. Week 8 hardened gates:
   - Reproducibility: pass, `mismatches = []`, `atol = 1e-6`.
   - Matrix: pass, `n_records = 6`, `stability_rate = 1.0`, no failure reasons.
   - Regression vs Week 7 baseline artifacts: pass (all checks acceptable).
   - Runtime budget: pass, `runtime_ratio_week8_over_week7 = 0.998747` (`<= 1.25`).
   - CI parity (quick/full): pass.

Interpretation:
The final packaging process does not degrade model behavior and is operationally reproducible for writing and rerun.

## Limitations and Threats to Validity

1. Synthetic-only scope:
   Findings support method robustness in controlled synthetic settings, not clinical validity.
2. Layer approximation scope:
   The two-layer mode is a reduced approximation and not a full volumetric biomechanical multilayer model.
3. Generalization uncertainty:
   Metrics were validated on fixed project grids/configs; broader parameter regions may expose additional failure modes.
4. Environment sensitivity:
   Deterministic reproducibility is verified in the project path/toolchain; cross-machine reproducibility is not yet comprehensively audited.
5. Runtime criterion semantics:
   Budget thresholds are engineering constraints for this cycle and should not be interpreted as evidence of physiological realism.

## Writing Guidance (Claim Strength)

Use stronger language:

1. "The pipeline is reproducible and stable on the reported synthetic benchmark protocol."
2. "Directional growth controls produce measurable morphology differences relative to isotropic settings."
3. "Spatial-hash contact handling improves overlap-count behavior under the tested ablation settings."
4. "Layered approximation yields consistent incremental improvements on selected reference metrics."

Use cautious language:

1. "These findings are synthetic-benchmark results and do not establish clinical performance."
2. "The layered mode is an approximation and should not be interpreted as a full anatomical tissue model."
3. "Cross-hardware determinism requires additional validation."

## Artifact Evidence Map

1. Baseline gate status:
   - `results/forward_sweep_summary.json`
   - `results/validation_gate_report.json`
2. Upgrade-specific outcomes:
   - `results/week3_anisotropy_comparison.json`
   - `results/week4_collision_comparison.json`
   - `results/week5_layered_comparison.json`
3. Packaging and reproducibility:
   - `results/week7_animation_comparison_summary.json`
   - `results/week8_submission_packet_summary.json`
   - `results/week8_hardened_validation.json`
4. Writing-ready packet:
   - `docs/week8_methods_results_packet.md`
   - `docs/assets/week8_figure_captions.md`
   - `results/week8_frozen_artifact_bundle.json`

## Suggested Methods Paragraph

"We evaluated a differentiable cortical folding simulator in a synthetic, config-driven protocol with fixed quality gates and seeded reproducibility checks. The model combines growth dynamics with elastic/bending mechanics, skull constraints, and self-contact handling. Across the development cycle, we introduced directional growth controls, spatial-hash collision handling, and a layered approximation mode. All reported metrics and visuals were tied to versioned artifacts, config hashes, and source-run mappings, and final packaging was validated through repeated seeded runs, seed/parameter matrix checks, regression checks against prior frozen artifacts, and runtime/CI parity gates."

## Suggested Results Paragraph

"The benchmark remained stable (`stability_rate=1.0`) and gate-compliant on the reported grid. Directional anisotropic growth produced substantial morphology changes relative to isotropic growth (`delta_gi=+1.251468`, `delta_curv_p90=-0.449325`). Spatial-hash collision handling reduced overlap-count outliers versus sampled collision checks, while layered approximation delivered consistent incremental improvements on reference metrics (`delta_gi=+0.091064`, `delta_outside_skull_frac=-0.006231`, `delta_disp_p95=-0.001118`). Final Week 8 hardening passed reproducibility (`mismatches=[]`, `atol=1e-6`), matrix robustness (`3 seeds x 2 settings`, `stability_rate=1.0`), regression, runtime budget (`week8/week7=0.998747 <= 1.25`), and CI parity checks."
