"""Soft penalty constraints for cortical folding simulation."""

import jax
import jax.numpy as jnp

from .mesh import MeshTopology


def _deterministic_pair_indices(n_verts: int, n_sample: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return deterministic pseudo-random pair indices."""
    idx_a = jnp.arange(n_sample, dtype=jnp.int32) % n_verts
    idx_b = (idx_a * 1103515245 + 12345) % n_verts
    return idx_a, idx_b


def _pair_adjacency_mask(
    idx_a: jnp.ndarray,
    idx_b: jnp.ndarray,
    edges: jnp.ndarray,
) -> jnp.ndarray:
    """Return mask for pairs that correspond to existing mesh edges."""
    a_col = idx_a[:, None]
    b_col = idx_b[:, None]
    return jnp.any(
        ((a_col == edges[None, :, 0]) & (b_col == edges[None, :, 1]))
        | ((a_col == edges[None, :, 1]) & (b_col == edges[None, :, 0])),
        axis=1,
    )


def skull_penalty(
    verts: jnp.ndarray,
    skull_center: jnp.ndarray,
    skull_radius: float,
    stiffness: float = 100.0,
) -> jnp.ndarray:
    """Soft penalty pushing vertices inside the skull boundary.

    Returns (V, 3) force that pushes inward when vertices exceed skull_radius.
    """
    disp = verts - skull_center[None, :]  # (V, 3)
    dist = jnp.linalg.norm(disp, axis=1, keepdims=True)  # (V, 1)
    safe_dist = jnp.maximum(dist, 1e-12)
    penetration = jnp.maximum(dist - skull_radius, 0.0)
    return -stiffness * penetration * disp / safe_dist


def self_collision_penalty(
    verts: jnp.ndarray,
    topo: MeshTopology,
    min_dist: float = 0.02,
    stiffness: float = 50.0,
    n_sample: int = 256,
    key=None,
) -> jnp.ndarray:
    """Approximate self-collision penalty via sampled non-adjacent pairs.

    For efficiency, this samples candidate vertex pairs and applies repulsion
    if they are closer than `min_dist`. Immediate topological neighbors are
    ignored so local mesh edges are not treated as collisions.
    """
    if key is None:
        # Deterministic pseudo-randomized pairing when no key is passed.
        idx_a = jnp.arange(n_sample) % verts.shape[0]
        idx_b = (idx_a * 1103515245 + 12345) % verts.shape[0]
    else:
        n_verts = verts.shape[0]
        k1, k2 = jax.random.split(key)
        idx_a = jax.random.randint(k1, (n_sample,), 0, n_verts)
        idx_b = jax.random.randint(k2, (n_sample,), 0, n_verts)

    n_verts = verts.shape[0]

    va = verts[idx_a]  # (n_sample, 3)
    vb = verts[idx_b]  # (n_sample, 3)
    diff = va - vb  # (n_sample, 3)
    dist = jnp.linalg.norm(diff, axis=1, keepdims=True)  # (n_sample, 1)
    safe_dist = jnp.maximum(dist, 1e-12)

    # Ignore trivial and topological-neighbor pairs.
    same = (idx_a == idx_b)[:, None]
    edges = topo.edges
    a_col = idx_a[:, None]
    b_col = idx_b[:, None]
    is_adjacent = jnp.any(
        ((a_col == edges[None, :, 0]) & (b_col == edges[None, :, 1]))
        | ((a_col == edges[None, :, 1]) & (b_col == edges[None, :, 0])),
        axis=1,
    )[:, None]

    gap = jnp.maximum(min_dist - dist, 0.0)
    repulsion = stiffness * gap * diff / safe_dist  # (n_sample, 3)
    repulsion = jnp.where(same | is_adjacent, 0.0, repulsion)

    forces = jnp.zeros_like(verts)
    forces = forces.at[idx_a].add(repulsion)
    forces = forces.at[idx_b].add(-repulsion)
    return forces
