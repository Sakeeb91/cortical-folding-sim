"""Tests for collision/contact constraint behavior."""

import jax.numpy as jnp
import numpy as np

from cortical_folding.constraints import self_collision_penalty
from cortical_folding.mesh import build_topology
from cortical_folding.synthetic import create_icosphere


def _sphere_topology(subdivisions: int = 1):
    verts, faces = create_icosphere(subdivisions=subdivisions, radius=1.0)
    topo = build_topology(np.asarray(verts), np.asarray(faces))
    return verts, topo


def test_sampled_collision_path_is_deterministic_without_key():
    verts, topo = _sphere_topology()
    f1 = self_collision_penalty(
        verts,
        topo,
        min_dist=0.15,
        stiffness=20.0,
        n_sample=256,
    )
    f2 = self_collision_penalty(
        verts,
        topo,
        min_dist=0.15,
        stiffness=20.0,
        n_sample=256,
    )
    np.testing.assert_allclose(np.asarray(f1), np.asarray(f2), atol=1e-8)


def test_spatial_hash_collision_path_is_deterministic():
    verts, topo = _sphere_topology()
    kwargs = dict(
        min_dist=0.15,
        stiffness=20.0,
        use_spatial_hash=True,
        hash_cell_size=0.15,
        hash_neighbor_window=8,
        deterministic_fallback=False,
    )
    f1 = self_collision_penalty(verts, topo, **kwargs)
    f2 = self_collision_penalty(verts, topo, **kwargs)
    np.testing.assert_allclose(np.asarray(f1), np.asarray(f2), atol=1e-8)


def test_spatial_hash_can_fall_back_to_deterministic_sampling():
    verts, topo = _sphere_topology()
    sampled = self_collision_penalty(
        verts,
        topo,
        min_dist=10.0,
        stiffness=1.0,
        n_sample=64,
    )
    fallback = self_collision_penalty(
        verts,
        topo,
        min_dist=10.0,
        stiffness=1.0,
        use_spatial_hash=True,
        hash_cell_size=1e-6,
        hash_neighbor_window=1,
        deterministic_fallback=True,
        fallback_n_sample=64,
    )
    np.testing.assert_allclose(np.asarray(fallback), np.asarray(sampled), atol=1e-8)


def test_spatial_hash_repels_close_nonadjacent_vertices():
    verts = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.70, 0.00, 0.00],
            [0.00, 0.70, 0.00],
            [0.01, 0.00, 0.00],
            [0.70, 0.00, 0.00],
            [0.00, 0.70, 0.00],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    topo = build_topology(verts, faces)
    forces = self_collision_penalty(
        jnp.asarray(verts),
        topo,
        min_dist=0.05,
        stiffness=100.0,
        use_spatial_hash=True,
        hash_cell_size=0.05,
        hash_neighbor_window=4,
        deterministic_fallback=False,
    )
    force_norms = np.linalg.norm(np.asarray(forces), axis=1)
    assert force_norms[0] > 0.0
    assert force_norms[3] > 0.0
