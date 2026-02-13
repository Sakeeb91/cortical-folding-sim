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
