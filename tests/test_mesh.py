"""Tests for mesh geometry operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cortical_folding.mesh import (
    build_topology,
    compute_face_normals,
    compute_face_areas,
    compute_vertex_normals,
    compute_edge_lengths,
    compute_vertex_areas,
    compute_mean_curvature,
    compute_gaussian_curvature,
)
from cortical_folding.synthetic import create_icosphere


@pytest.fixture
def sphere():
    verts, faces = create_icosphere(subdivisions=3, radius=1.0)
    topo = build_topology(verts, faces)
    return verts, faces, topo


def test_topology_counts(sphere):
    verts, faces, topo = sphere
    V, F = verts.shape[0], faces.shape[0]
    E = topo.edges.shape[0]
    # Euler: V - E + F = 2 for sphere
    assert V - E + F == 2


def test_face_areas_sum(sphere):
    """Sum of face areas should approximate 4*pi*r^2 for unit sphere."""
    verts, _, topo = sphere
    areas = compute_face_areas(verts, topo)
    total = float(jnp.sum(areas))
    expected = 4.0 * np.pi  # r=1
    assert abs(total - expected) / expected < 0.02  # within 2%


def test_face_normals_unit(sphere):
    verts, _, topo = sphere
    normals = compute_face_normals(verts, topo)
    norms = jnp.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_vertex_normals_outward(sphere):
    """Vertex normals on unit sphere should point radially outward."""
    verts, _, topo = sphere
    vnormals = compute_vertex_normals(verts, topo)
    # For unit sphere, normal ≈ vertex position (normalized)
    expected = verts / jnp.linalg.norm(verts, axis=1, keepdims=True)
    dots = jnp.sum(vnormals * expected, axis=1)
    assert float(jnp.min(dots)) > 0.95


def test_mean_curvature_sphere(sphere):
    """Mean curvature of unit sphere should be ~1.0 everywhere."""
    verts, _, topo = sphere
    H = compute_mean_curvature(verts, topo)
    np.testing.assert_allclose(H, 1.0, atol=0.15)


def test_gaussian_curvature_sphere(sphere):
    """Gaussian curvature of unit sphere should be ~1.0 everywhere."""
    verts, _, topo = sphere
    K = compute_gaussian_curvature(verts, topo)
    np.testing.assert_allclose(K, 1.0, atol=0.2)


def test_vertex_areas_sum(sphere):
    """Sum of vertex areas should also approximate 4*pi."""
    verts, _, topo = sphere
    v_areas = compute_vertex_areas(verts, topo)
    total = float(jnp.sum(v_areas))
    expected = 4.0 * np.pi
    assert abs(total - expected) / expected < 0.02


def test_edge_lengths_sphere(sphere):
    """All edge lengths on icosphere should be approximately equal."""
    verts, _, topo = sphere
    lengths = compute_edge_lengths(verts, topo)
    mean_len = float(jnp.mean(lengths))
    std_len = float(jnp.std(lengths))
    assert std_len / mean_len < 0.15  # relatively uniform
