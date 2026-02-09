# Cortical Folding Simulator — Implementation Status

## What's Done (100% core implementation)

All source modules, scripts, and tests are **fully implemented and passing** (16/16 tests).

### Files Created

| File | Status | Description |
|------|--------|-------------|
| `pyproject.toml` | DONE | Project config (JAX, Equinox, Optax, trimesh, matplotlib) |
| `src/cortical_folding/__init__.py` | DONE | Package init |
| `src/cortical_folding/mesh.py` | DONE | MeshTopology NamedTuple, build_topology(), all geometry ops (face normals/areas, vertex normals, edge lengths, cotangent Laplacian, mean/Gaussian curvature, vertex areas) |
| `src/cortical_folding/physics.py` | DONE | elastic_force (edge-scatter), bending_force (autodiff), logistic growth, plasticity, rest length updates |
| `src/cortical_folding/constraints.py` | DONE | skull_penalty, self_collision_penalty (random sampling) |
| `src/cortical_folding/solver.py` | DONE | SimState/SimParams NamedTuples, simulation_step (Newmark), simulate() with lax.scan + checkpoint |
| `src/cortical_folding/losses.py` | DONE | curvature_loss, gyrification_index, gi_loss, vertex_loss, total_loss |
| `src/cortical_folding/growth_net.py` | DONE | GrowthFieldNet (Equinox MLP), extract_vertex_features, growth_rates_to_faces |
| `src/cortical_folding/synthetic.py` | DONE | create_icosphere, create_skull, uniform/regional growth, create_target_folded |
| `src/cortical_folding/viz.py` | DONE | plot_mesh, plot_growth_field, plot_curvature_map, plot_simulation_frames |
| `scripts/run_forward.py` | DONE | Forward simulation demo |
| `scripts/train_inverse.py` | DONE | Inverse problem training loop |
| `scripts/demo_sphere.py` | DONE | Minimal 20-line demo |
| `tests/test_mesh.py` | DONE | 8 tests (Euler formula, areas, normals, curvatures) |
| `tests/test_physics.py` | DONE | 5 tests (elastic equilibrium, bending energy, logistic growth) |
| `tests/test_solver.py` | DONE | 3 tests (stability, growth increases area, differentiability through simulate) |

### Test Results
```
16 passed in 3.55s
```

### Key Design Notes
- `make_initial_state` sets `rest_curvatures` to the actual initial mean curvature (not zero), so the sphere is stable without growth
- GI (gyrification index) = cortical area / skull area. For a unit sphere inside a 1.5-radius skull, initial GI ≈ 0.44, which is correct — GI > 1 requires area growth + folding
- Differentiability is verified: `jax.grad` through `simulate()` returns finite, nonzero gradients

## What Could Be Enhanced (optional follow-up)

1. **Run the demo scripts** to verify visual output:
   - `PYTHONPATH=src python3.11 scripts/demo_sphere.py`
   - `PYTHONPATH=src python3.11 scripts/run_forward.py`
   - `PYTHONPATH=src python3.11 scripts/train_inverse.py`

2. **Tune simulation parameters** — the default params work for stability but may need tuning for visually compelling folds:
   - Increase `n_steps` (200-500) for more folding
   - Adjust `Kb` (bending stiffness) vs `Kc` (elastic stiffness) ratio
   - Try higher `carrying_cap_factor` for more area growth

3. **Self-collision** — currently uses random vertex-pair sampling which is approximate. Could upgrade to spatial hashing or BVH for better collision detection.

4. **Performance** — for higher resolution meshes (subdivisions=5+), may need:
   - `jax.checkpoint` tuning (already used in simulate)
   - Batched collision detection
   - Float64 for numerical stability at high resolution

5. **Git initialization** — project is not yet a git repo. Initialize with:
   ```bash
   cd "/Users/sakeeb/Code repositories/cortical-folding-sim"
   git init && git add . && git commit -m "Initial implementation of differentiable cortical folding simulator"
   ```

## Continuation Prompt

If you need Claude to continue working on this project, use:

---

**Prompt:**

> Continue working on the differentiable cortical folding simulator at `/Users/sakeeb/Code repositories/cortical-folding-sim/`.
>
> The full implementation is complete with all source modules, scripts, and tests passing (16/16). See `CONTINUATION.md` for detailed status.
>
> The project implements a JAX-based differentiable cortical folding simulator with:
> - `mesh.py` — topology + differential geometry (cotangent Laplacian, curvatures)
> - `physics.py` — elastic forces (edge-scatter), bending forces (autodiff), logistic growth
> - `constraints.py` — skull penalty, self-collision
> - `solver.py` — Newmark integration with `lax.scan` + `jax.checkpoint`
> - `growth_net.py` — Equinox neural network for growth field prediction
> - `losses.py` — curvature, GI, vertex position losses
> - `synthetic.py` — icosphere + test data generation
> - `viz.py` — matplotlib visualization
> - Scripts: `run_forward.py`, `train_inverse.py`, `demo_sphere.py`
> - Tests: 16 passing tests covering mesh geometry, physics, solver stability, and differentiability
>
> Install: `python3.11 -m pip install -e ".[dev]"`
> Test: `python3.11 -m pytest tests/ -v`
>
> **Task:** [describe what you want to do next]

---
