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
