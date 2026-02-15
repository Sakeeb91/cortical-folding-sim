# Differentiable Cortical Folding Simulator

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-54%20passing-2EA44F)](#testing)
[![Framework](https://img.shields.io/badge/JAX-differentiable%20physics-F7931E)](https://github.com/jax-ml/jax)

A research-oriented simulator for cortical folding with end-to-end differentiability in JAX.

The project supports:

1. Forward simulation (`growth parameters -> folded trajectories`).
2. Inverse optimization (`target morphology -> recovered growth field`).

## Why This Project

This repository is designed as a high-quality computational mechanics project with:

1. Differentiable physics for optimization and learning workflows.
2. Config-driven benchmarking and validation gates.
3. Reproducibility-focused artifact generation (CSV/JSON summaries, manifests, config hashes).
4. Deterministic visualization and reporting pipelines for technical communication.

## Core Capabilities

1. Mesh-based mechanics with elastic and bending forces.
2. Growth modeling with isotropic and directional controls.
3. Constraint handling for skull interaction and self-collision.
4. Stable integration with numerical safety rails and fail-fast checks.
5. Metric reporting for morphology, curvature, penetration, and runtime.
6. Programmatic figure/animation generation and evidence indexing.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.11+ |
| Differentiable compute | JAX |
| NN components | Equinox, Optax |
| Numerics | NumPy |
| Visualization | Matplotlib |
| Testing | Pytest |

## Quickstart

### 1) Install

```bash
python3.11 -m pip install -e ".[dev]"
```

### 2) Run tests

```bash
python3.11 -m pytest tests -q
```

### 3) Run demos

```bash
MPLBACKEND=Agg python3.11 scripts/demo_sphere.py
MPLBACKEND=Agg python3.11 scripts/run_forward.py
MPLBACKEND=Agg python3.11 scripts/train_inverse.py
MPLBACKEND=Agg python3.11 scripts/animate_forward.py --output docs/assets/forward_simulation.gif --rotate
```

## Benchmarking and Validation

### Forward benchmark

```bash
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py \
  --config-path configs/forward_sweep_baseline.json \
  --n-steps 200 \
  --output-csv results/forward_sweep.csv \
  --output-summary results/forward_sweep_summary.json \
  --output-manifest results/forward_sweep_manifest.json
```

### Gate checks

```bash
python3.11 scripts/check_forward_sweep_gates.py \
  --input-csv results/forward_sweep.csv \
  --input-summary results/forward_sweep_summary.json \
  --gate-config configs/validation_gates_default.json \
  --output-report results/validation_gate_report.json \
  --fail-on-failure
```

### CI-aligned validation presets

```bash
./scripts/run_validation_quick.sh
./scripts/run_validation_full.sh
```

### High-fidelity simulation mode

```bash
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py \
  --mode high_fidelity \
  --config-path configs/high_fidelity_forward_sweep.json \
  --n-steps 120 \
  --output-csv results/high_fidelity/forward_sweep.csv \
  --output-summary results/high_fidelity/forward_sweep_summary.json \
  --output-manifest results/high_fidelity/forward_sweep_manifest.json
```

### Publication render preset (GIF + MP4 from one pipeline)

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

### High-fidelity hardened validation

```bash
python3.11 scripts/validate_high_fidelity.py \
  --output-dir results/high_fidelity \
  --n-steps 120
```

### End-to-end packaging and validation

The repository includes automated packaging and hardened validation workflows in `scripts/` that generate publication-style artifacts, reproducibility manifests, and gate reports.

## Representative Artifacts

1. `results/forward_sweep.csv`
2. `results/forward_sweep_summary.json`
3. `results/forward_sweep_manifest.json`
4. `results/validation_gate_report.json`
5. `docs/results_index.md`
6. `docs/assets/forward_simulation.gif`

## Visualization Outputs

| Forward Folding | Growth Field |
|---|---|
| ![Forward simulation](docs/assets/forward_simulation.png) | ![Growth field](docs/assets/growth_field.png) |

| Inverse Training Loss | Growth Comparison |
|---|---|
| ![Inverse loss](docs/assets/inverse_training_loss.png) | ![Growth comparison](docs/assets/growth_comparison.png) |

![Forward animation](docs/assets/forward_simulation.gif)

## Engineering Quality Notes

1. Deterministic checks are enforced for seeded paths with strict tolerances.
2. Validation gates include stability and safety metrics with explicit thresholds.
3. Runtime-budget and regression checks are tracked in machine-readable reports.
4. Output manifests map metrics and visuals back to configs and source runs.

## Project Structure

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
├── tests/
├── configs/
├── docs/
└── results/
```

## Testing

Current automated tests cover:

1. Mesh and geometry operations.
2. Physics and solver behavior.
3. Validation and reproducibility utilities.
4. Pipeline and manifest generation scripts.

Run:

```bash
python3.11 -m pytest tests -q
```

Optional negative smoke test (expected failure path):

```bash
python3.11 scripts/check_forward_sweep_gates.py \
  --input-csv results/forward_sweep.csv \
  --input-summary results/forward_sweep_summary.json \
  --gate-config configs/validation_gates_failcase.json \
  --output-report results/validation_gate_report_failcase.json \
  --fail-on-failure
```

## Intended Use and Scope

1. This is a research and engineering prototype for computational modeling.
2. Results are synthetic-data validated and reproducibility-focused.
3. The project is not a clinical diagnostic tool.

## References

1. Nie, J., Li, G., and Shen, D. (2010). *A computational model of cerebral cortex folding*. Journal of Theoretical Biology.
2. Tallinen, T. et al. (2014). *Gyrification from constrained cortical expansion*. PNAS.
3. Budday, S. et al. (2014). *A mechanical model predicts morphological abnormalities in the developing human brain*. Scientific Reports.

## Citation

If you use this repository, cite it as software and include a commit hash.

```bibtex
@software{cortical_folding_sim,
  title = {Differentiable Cortical Folding Simulator},
  author = {Sakeeb and Contributors},
  year = {2026},
  url = {https://github.com/Sakeeb91/cortical-folding-sim},
  note = {Commit: <insert-commit-hash>}
}
```
