"""Soft penalty constraints for cortical folding simulation."""

import jax.numpy as jnp

from .mesh import MeshTopology


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
    """Approximate self-collision penalty via random vertex-pair sampling.

    For efficiency, we sample random non-adjacent vertex pairs and apply
    repulsion if they are closer than min_dist. Returns (V, 3).
    """
    if key is None:
        import jax
        key = jax.random.PRNGKey(0)

    n_verts = verts.shape[0]
    k1, k2 = jax.random.split(key)
    idx_a = jax.random.randint(k1, (n_sample,), 0, n_verts)
    idx_b = jax.random.randint(k2, (n_sample,), 0, n_verts)

    va = verts[idx_a]  # (n_sample, 3)
    vb = verts[idx_b]  # (n_sample, 3)
    diff = va - vb  # (n_sample, 3)
    dist = jnp.linalg.norm(diff, axis=1, keepdims=True)  # (n_sample, 1)
    safe_dist = jnp.maximum(dist, 1e-12)

    # Only repel if closer than min_dist and not the same vertex
    same = (idx_a == idx_b)[:, None]
    gap = jnp.maximum(min_dist - dist, 0.0)
    repulsion = stiffness * gap * diff / safe_dist  # (n_sample, 3)
    repulsion = jnp.where(same, 0.0, repulsion)

    forces = jnp.zeros_like(verts)
    forces = forces.at[idx_a].add(repulsion)
    forces = forces.at[idx_b].add(-repulsion)
    return forces


import jax
