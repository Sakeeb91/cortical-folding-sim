# Differentiable Cortical Folding Simulator

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-18%20passing-2EA44F)](#testing)
[![Framework](https://img.shields.io/badge/JAX-differentiable%20physics-F7931E)](https://github.com/jax-ml/jax)

A research-focused simulator for cortical folding that is differentiable end-to-end in JAX.

It supports both:

1. Forward simulation (growth parameters -> folded surface)
2. Inverse optimization (target folded surface -> recovered growth field)

## At A Glance

| Item | Details |
|---|---|
| Core stack | JAX, Equinox, Optax, NumPy, Matplotlib |
| Language | Python 3.11+ |
| Main output | Folded cortical surface trajectories + growth recovery |
| Current validation | Unit tests + synthetic benchmarks |
| Research mode | Methods + synthetic validation with roadmap to real-data pilot |

## Key Features

1. Differentiable mesh-based mechanics (elastic + bending + constrained growth)
2. `lax.scan` simulation loop with checkpointing for scalable backpropagation
3. Growth-field neural network (`GrowthFieldNet`) for inverse fitting
4. Reproducible forward parameter sweep with CSV/JSON outputs
5. Visualization utilities for trajectories, growth maps, and curvature maps

## Quickstart

### 1. Install

```bash
python3.11 -m pip install -e '.[dev]'
```

### 2. Run tests

```bash
python3.11 -m pytest tests -q
```

### 3. Run simulation demos

```bash
MPLBACKEND=Agg python3.11 scripts/demo_sphere.py
MPLBACKEND=Agg python3.11 scripts/run_forward.py
MPLBACKEND=Agg python3.11 scripts/train_inverse.py
MPLBACKEND=Agg python3.11 scripts/animate_forward.py --output forward_simulation.gif --rotate
```

## Reproducible Research Commands

### Forward sweep benchmark

```bash
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py --n-steps 120
```

Generated artifacts:

1. `results/forward_sweep.csv`
2. `results/forward_sweep_summary.json`
3. `results/forward_sweep_manifest.json`
4. `results/validation_gate_report.json` (after gate check)

### Full forward sweep grid

```bash
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py \
  --config-path configs/forward_sweep_baseline.json \
  --n-steps 200 \
  --output-manifest results/forward_sweep_manifest.json

python3.11 scripts/check_forward_sweep_gates.py \
  --input-csv results/forward_sweep.csv \
  --input-summary results/forward_sweep_summary.json \
  --gate-config configs/validation_gates_default.json \
  --output-report results/validation_gate_report.json \
  --fail-on-failure
```

### CI command presets

```bash
./scripts/run_validation_quick.sh
./scripts/run_validation_full.sh
```

## Visualization Outputs

The current repository already includes generated sample outputs:

1. `docs/assets/forward_simulation.png`
2. `docs/assets/growth_field.png`
3. `docs/assets/inverse_training_loss.png`
4. `docs/assets/growth_comparison.png`
5. `docs/assets/forward_simulation.gif` (generated via `scripts/animate_forward.py`)

| Forward Folding | Growth Field |
|---|---|
| ![Forward simulation](docs/assets/forward_simulation.png) | ![Growth field](docs/assets/growth_field.png) |

| Inverse Training Loss | Growth Comparison |
|---|---|
| ![Inverse loss](docs/assets/inverse_training_loss.png) | ![Growth comparison](docs/assets/growth_comparison.png) |

![Forward animation](docs/assets/forward_simulation.gif)

## Robustness Features

The solver now includes configurable numerical safety rails:

1. Per-vertex clipping for force, acceleration, velocity, and step displacement
2. Growth-rate and rest-geometry lower/upper bounds
3. Finite-value guards that fall back to the previous state on numerical overflow
4. Optional self-collision penalty with deterministic sampling and adjacency filtering
5. Trajectory subsampling (`save_every`) for memory-aware long simulations

## Project Layout

```text
.
├── src/cortical_folding/
│   ├── mesh.py
│   ├── physics.py
│   ├── constraints.py
│   ├── solver.py
│   ├── growth_net.py
│   ├── losses.py
│   ├── synthetic.py
│   └── viz.py
├── scripts/
│   ├── demo_sphere.py
│   ├── run_forward.py
│   ├── run_forward_sweep.py
│   ├── train_inverse.py
│   └── animate_forward.py
├── tests/
├── PLAN.md
├── PROJECT_OVERVIEW.md
└── CONTINUATION.md
```

## Research Workflow (Paper-Oriented)

Use this repository as a small methods paper pipeline:

1. Validate solver behavior with forward sweeps
2. Quantify metrics (GI, area ratio, curvature stats, stability)
3. Run inverse recovery on synthetic targets
4. Report mean/std across seeds and include failure modes
5. Extend to a small real-data pilot only after robustness criteria pass

The structured execution plan is documented in `PLAN.md`.

## Testing

Current test suite covers:

1. Mesh geometry correctness
2. Physics module behavior
3. Solver stability and differentiability

Run:

```bash
python3.11 -m pytest tests -v
```

## References

1. Nie, J., Li, G., and Shen, D. (2010). *A computational model of cerebral cortex folding*. Journal of Theoretical Biology.
2. Tallinen, T. et al. (2014). *Gyrification from constrained cortical expansion*. PNAS.
3. Budday, S. et al. (2014). *A mechanical model predicts morphological abnormalities in the developing human brain*. Scientific Reports.

## Citation

If you use this repository in a report/paper, cite it as software and include a commit hash.

Suggested BibTeX template:

```bibtex
@software{cortical_folding_sim,
  title = {Differentiable Cortical Folding Simulator},
  author = {Sakeeb and Contributors},
  year = {2026},
  url = {https://github.com/Sakeeb91/cortical-folding-sim},
  note = {Commit: <insert-commit-hash>}
}
```

## Status

Active research prototype (synthetic-data validated). Not a clinical tool.
