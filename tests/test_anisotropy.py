"""Tests for anisotropic growth extensions."""

import jax.numpy as jnp

from cortical_folding.synthetic import (
    create_anisotropy_field,
    create_icosphere,
    create_regional_anisotropy,
    create_uniform_growth,
    create_uniform_anisotropy,
)
from cortical_folding.mesh import build_topology
from cortical_folding.solver import SimParams, make_initial_state, simulate


def test_anisotropy_test_scaffold():
    verts, _ = create_icosphere(subdivisions=1, radius=1.0)
    assert verts.shape[1] == 3


def test_uniform_anisotropy_shape_and_value():
    field = create_uniform_anisotropy(12, value=0.25)
    assert field.shape == (12,)
    assert jnp.allclose(field, 0.25)


def test_regional_anisotropy_field_has_two_regions():
    verts, faces = create_icosphere(subdivisions=2, radius=1.0)
    field = create_regional_anisotropy(
        verts, faces, high_value=1.0, low_value=0.0, axis=2, threshold=0.0
    )
    assert float(jnp.max(field)) == 1.0
    assert float(jnp.min(field)) == 0.0


def test_zero_anisotropy_matches_isotropic_trajectory():
    verts, faces = create_icosphere(subdivisions=2, radius=1.0)
    topo = build_topology(verts, faces)
    growth = create_uniform_growth(topo.faces.shape[0], rate=0.5)
    state = make_initial_state(verts, topo)
    params = SimParams(dt=0.02, anisotropy_strength=0.0)
    no_aniso_state, _ = simulate(state, topo, growth, params, n_steps=20)
    explicit_zero = create_anisotropy_field(
        "uniform", verts, topo.faces, high_value=0.0, low_value=0.0
    )
    zero_state, _ = simulate(
        state, topo, growth, params, face_anisotropy=explicit_zero, n_steps=20
    )
    assert jnp.allclose(no_aniso_state.vertices, zero_state.vertices)
