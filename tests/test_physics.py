"""Tests for physics forces and growth."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cortical_folding.mesh import build_topology, compute_edge_lengths, compute_face_areas
from cortical_folding.physics import (
    elastic_force,
    bending_energy,
    grow_rest_areas,
    update_rest_lengths_from_areas,
)
from cortical_folding.synthetic import create_icosphere


@pytest.fixture
def sphere():
    verts, faces = create_icosphere(subdivisions=3, radius=1.0)
    topo = build_topology(verts, faces)
    return verts, faces, topo


def test_elastic_zero_at_rest(sphere):
    """Elastic force should be ~zero when mesh is at rest lengths."""
    verts, _, topo = sphere
    rest_lengths = compute_edge_lengths(verts, topo)
    forces = elastic_force(verts, topo, rest_lengths, Kc=1.0)
    max_force = float(jnp.max(jnp.abs(forces)))
    assert max_force < 1e-5


def test_elastic_nonzero_stretched(sphere):
    """Elastic force should be nonzero when mesh is stretched."""
    verts, _, topo = sphere
    rest_lengths = compute_edge_lengths(verts, topo) * 0.8  # shorter rest = stretched
    forces = elastic_force(verts, topo, rest_lengths, Kc=1.0)
    max_force = float(jnp.max(jnp.abs(forces)))
    assert max_force > 0.01


def test_bending_energy_sphere(sphere):
    """Bending energy with rest_curvature=H should be ~0."""
    verts, _, topo = sphere
    from cortical_folding.mesh import compute_mean_curvature
    H = compute_mean_curvature(verts, topo)
    energy = bending_energy(verts, topo, H, Kb=5.0)
    assert float(energy) < 1e-3


def test_logistic_growth():
    """Logistic growth should increase areas and saturate."""
    areas = jnp.ones(10)
    rates = jnp.full(10, 0.5)
    cap = jnp.full(10, 3.0)
    for _ in range(1000):
        areas = grow_rest_areas(areas, rates, cap, dt=0.05)
    # Should approach carrying capacity
    np.testing.assert_allclose(areas, 3.0, atol=0.1)


def test_rest_length_from_area_identity(sphere):
    """With no area growth, rest lengths should remain unchanged."""
    verts, _, topo = sphere
    init_lengths = compute_edge_lengths(verts, topo)
    init_areas = compute_face_areas(verts, topo)
    new_lengths = update_rest_lengths_from_areas(init_lengths, init_areas, init_areas, topo)
    np.testing.assert_allclose(new_lengths, init_lengths, atol=1e-6)
