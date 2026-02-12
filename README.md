# Differentiable Cortical Folding Simulator

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-16%20passing-2EA44F)](#testing)
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
```

## Reproducible Research Commands

### Forward sweep benchmark

```bash
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py --quick --n-steps 120
```

Generated artifacts:

1. `results/forward_sweep.csv`
2. `results/forward_sweep_summary.json`

### Full forward sweep grid

```bash
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py --n-steps 200
```

## Visualization Outputs

The current repository already includes generated sample outputs:

1. `demo_folding.png`
2. `forward_simulation.png`
3. `growth_field.png`
4. `inverse_training_loss.png`
5. `growth_comparison.png`

| Forward Folding | Growth Field |
|---|---|
| ![Forward simulation](forward_simulation.png) | ![Growth field](growth_field.png) |

| Inverse Training Loss | Growth Comparison |
|---|---|
| ![Inverse loss](inverse_training_loss.png) | ![Growth comparison](growth_comparison.png) |

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
│   └── train_inverse.py
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
