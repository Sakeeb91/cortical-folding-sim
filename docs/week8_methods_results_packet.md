# Week 8 Methods and Results Packet

Source plan: `PLAN.md`

## Acceptance Criteria Status

| ID | Criterion | Passed | Waived | Effective pass |
|---|---|---|---|---|
| S1 | Stability rate >= 95% on benchmark grid. | true | false | true |
| S2 | Improved realism beats baseline on at least two morphology metrics. | true | false | true |
| S3 | Main figures and animations regenerate from one documented command path. | true | false | true |
| S4 | Every reported metric links to saved config + summary artifact. | true | false | true |

## Final Metrics Table

| Comparison | Delta GI | Delta Curvature p90 | Delta Disp p95 | Notes |
|---|---|---|---|---|
| Week 3 anisotropic vs isotropic | 1.251468 | -0.449325 | 0.025195 | stable_both=true |
| Week 4 spatial-hash vs sampled collision | - | - | - | delta_overlap_count=15.0, runtime_ratio=1.4841 |
| Week 5 layered vs non-layered | 0.091064 | - | -0.001118 | delta_outside=-0.006231 |

## Reproducibility Commands

1. `python3.11 -m pytest tests -q`
1. `./scripts/run_validation_quick.sh`
1. `./scripts/run_validation_full.sh`
1. `MPLBACKEND=Agg python3.11 scripts/regenerate_week6_figures.py --n-steps 140`
1. `MPLBACKEND=Agg python3.11 scripts/regenerate_week7_animation_pack.py --n-steps 140`
1. `MPLBACKEND=Agg python3.11 scripts/regenerate_week8_final_package.py --n-steps 140`
1. `MPLBACKEND=Agg python3.11 scripts/validate_week8_hardened.py --n-steps 140`
