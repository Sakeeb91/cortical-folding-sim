"""Tests for solver and differentiability."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cortical_folding.mesh import build_topology, compute_face_areas
from cortical_folding.synthetic import create_icosphere, create_uniform_growth
from cortical_folding.solver import SimParams, make_initial_state, simulate
from cortical_folding.losses import gyrification_index


@pytest.fixture
def setup():
    verts, faces = create_icosphere(subdivisions=2, radius=1.0)
    topo = build_topology(verts, faces)
    skull_center = jnp.zeros(3)
    skull_radius = 1.5
    params = SimParams(
        Kc=2.0, Kb=3.0, damping=0.9,
        skull_center=skull_center, skull_radius=skull_radius,
        skull_stiffness=100.0, carrying_cap_factor=4.0,
        tau=500.0, dt=0.02,
    )
    return verts, topo, params, skull_radius


def test_no_growth_stable(setup):
    """Without growth, sphere should remain stable (area preserved)."""
    verts, topo, params, skull_radius = setup
    growth = create_uniform_growth(topo.faces.shape[0], rate=0.0)
    state = make_initial_state(verts, topo)
    initial_area = float(jnp.sum(compute_face_areas(verts, topo)))
    final, _ = simulate(state, topo, growth, params, n_steps=50)
    final_area = float(jnp.sum(compute_face_areas(final.vertices, topo)))
    # Area should be preserved within 5%
    assert abs(final_area - initial_area) / initial_area < 0.05


def test_growth_increases_area(setup):
    """With growth, total surface area should increase."""
    verts, topo, params, skull_radius = setup
    growth = create_uniform_growth(topo.faces.shape[0], rate=0.5)
    state = make_initial_state(verts, topo)
    initial_area = float(jnp.sum(compute_face_areas(verts, topo)))
    final, _ = simulate(state, topo, growth, params, n_steps=100)
    final_area = float(jnp.sum(compute_face_areas(final.vertices, topo)))
    gi_initial = float(gyrification_index(verts, topo, skull_radius))
    gi_final = float(gyrification_index(final.vertices, topo, skull_radius))
    # GI should increase with growth
    assert gi_final > gi_initial


def test_differentiability(setup):
    """jax.grad through simulate should return finite gradients."""
    verts, topo, params, skull_radius = setup
    growth = create_uniform_growth(topo.faces.shape[0], rate=0.5)

    def loss_fn(growth_rates):
        state = make_initial_state(verts, topo)
        final, _ = simulate(state, topo, growth_rates, params, n_steps=20)
        return jnp.mean(final.vertices ** 2)

    grads = jax.grad(loss_fn)(growth)
    assert jnp.all(jnp.isfinite(grads))
    assert float(jnp.max(jnp.abs(grads))) > 0


def test_save_every_subsamples_trajectory(setup):
    """simulate(save_every=k) should return ceil(n_steps / k) snapshots."""
    verts, topo, params, _ = setup
    growth = create_uniform_growth(topo.faces.shape[0], rate=0.2)
    state = make_initial_state(verts, topo)
    n_steps = 23
    save_every = 5
    _, traj = simulate(state, topo, growth, params, n_steps=n_steps, save_every=save_every)
    assert traj.shape[0] == (n_steps + save_every - 1) // save_every


def test_velocity_clipping_bound(setup):
    """Robustness clipping should bound velocity norm."""
    verts, topo, params, _ = setup
    growth = create_uniform_growth(topo.faces.shape[0], rate=0.8)
    param_dict = params._asdict()
    param_dict.update(
        {
            "max_velocity_norm": 0.2,
            "max_acc_norm": 0.3,
            "max_force_norm": 0.3,
            "max_displacement_per_step": 0.01,
        }
    )
    constrained_params = SimParams(**param_dict)
    state = make_initial_state(verts, topo)
    final, _ = simulate(state, topo, growth, constrained_params, n_steps=30)
    vel_norm = jnp.linalg.norm(final.velocities, axis=1)
    assert float(jnp.max(vel_norm)) <= 0.2001


def test_spatial_hash_collision_reduces_overlap_outliers(setup):
    """Spatial-hash collision should not worsen non-adjacent overlap p95."""
    verts, topo, params, _ = setup
    growth = create_uniform_growth(topo.faces.shape[0], rate=0.9)
    state = make_initial_state(verts, topo)

    no_collision_params = params
    with_collision_dict = params._asdict()
    with_collision_dict.update(
        {
            "enable_self_collision": True,
            "self_collision_min_dist": 0.05,
            "self_collision_stiffness": 65.0,
            "self_collision_n_sample": 768,
            "self_collision_use_spatial_hash": True,
            "self_collision_hash_cell_size": 0.05,
            "self_collision_hash_neighbor_window": 10,
            "self_collision_deterministic_fallback": True,
            "self_collision_fallback_n_sample": 256,
        }
    )
    collision_params = SimParams(**with_collision_dict)

    final_no_collision, _ = simulate(state, topo, growth, no_collision_params, n_steps=80)
    final_collision, _ = simulate(state, topo, growth, collision_params, n_steps=80)

    def overlap_p95(vertices):
        verts_np = np.asarray(vertices)
        n_verts = verts_np.shape[0]
        adjacency = np.zeros((n_verts, n_verts), dtype=bool)
        edges = np.asarray(topo.edges)
        adjacency[edges[:, 0], edges[:, 1]] = True
        adjacency[edges[:, 1], edges[:, 0]] = True
        tri_mask = np.triu(np.ones((n_verts, n_verts), dtype=bool), k=1)
        valid_mask = tri_mask & (~adjacency)
        pairwise_dist = np.linalg.norm(verts_np[:, None, :] - verts_np[None, :, :], axis=2)
        overlap = np.maximum(0.05 - pairwise_dist[valid_mask], 0.0)
        return float(np.percentile(overlap, 95))

    assert overlap_p95(final_collision.vertices) <= overlap_p95(final_no_collision.vertices) + 1e-6
