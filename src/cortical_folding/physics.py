"""Physical forces and growth model for cortical folding simulation."""

import jax
import jax.numpy as jnp

from .mesh import (
    MeshTopology,
    compute_edge_lengths,
    compute_mean_curvature,
    compute_vertex_areas,
    _cotangent_weights,
)


def elastic_force(
    verts: jnp.ndarray,
    topo: MeshTopology,
    rest_lengths: jnp.ndarray,
    Kc: float = 1.0,
) -> jnp.ndarray:
    """Edge-based elastic (spring) forces. Returns (V, 3)."""
    edge_vecs = verts[topo.edges[:, 1]] - verts[topo.edges[:, 0]]  # (E, 3)
    lengths = jnp.linalg.norm(edge_vecs, axis=1, keepdims=True)  # (E, 1)
    safe_lengths = jnp.maximum(lengths, 1e-12)
    strain = (safe_lengths - rest_lengths[:, None]) / rest_lengths[:, None]
    force_per_edge = Kc * strain * edge_vecs / safe_lengths  # (E, 3)

    forces = jnp.zeros_like(verts)
    forces = forces.at[topo.edges[:, 0]].add(force_per_edge)
    forces = forces.at[topo.edges[:, 1]].add(-force_per_edge)
    return forces


def bending_energy(
    verts: jnp.ndarray,
    topo: MeshTopology,
    rest_curvatures: jnp.ndarray,
    Kb: float = 5.0,
) -> float:
    """Bending energy: Kb * sum((H - H0)^2 * A_v). Returns scalar."""
    H = compute_mean_curvature(verts, topo)
    A = compute_vertex_areas(verts, topo)
    return Kb * jnp.sum((H - rest_curvatures) ** 2 * A)


def bending_force(
    verts: jnp.ndarray,
    topo: MeshTopology,
    rest_curvatures: jnp.ndarray,
    Kb: float = 5.0,
) -> jnp.ndarray:
    """Bending force via autodiff of bending energy. Returns (V, 3)."""
    grad_fn = jax.grad(bending_energy, argnums=0)
    return -grad_fn(verts, topo, rest_curvatures, Kb)


def grow_rest_areas(
    rest_areas: jnp.ndarray,
    growth_rates: jnp.ndarray,
    carrying_cap: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Logistic growth of rest areas. Returns (F,)."""
    dA = rest_areas * growth_rates * (1.0 - rest_areas / carrying_cap) * dt
    return rest_areas + dA


def update_rest_lengths_plasticity(
    rest_lengths: jnp.ndarray,
    current_lengths: jnp.ndarray,
    tau: float = 1000.0,
    dt: float = 0.05,
) -> jnp.ndarray:
    """Viscoplastic rest length adaptation. Returns (E,)."""
    return rest_lengths + (current_lengths - rest_lengths) * (dt / tau)


def update_rest_lengths_from_areas(
    initial_edge_lengths: jnp.ndarray,
    rest_areas: jnp.ndarray,
    initial_areas: jnp.ndarray,
    topo: MeshTopology,
) -> jnp.ndarray:
    """Scale rest lengths proportionally to sqrt of area growth ratio.

    Handles boundary edges (edge_faces == -1) by using scale=1.0 for missing faces.
    Returns (E,).
    """
    face_scale = jnp.sqrt(rest_areas / jnp.maximum(initial_areas, 1e-12))

    ef = topo.edge_faces  # (E, 2)
    valid0 = ef[:, 0] >= 0
    valid1 = ef[:, 1] >= 0

    scale0 = jnp.where(valid0, face_scale[jnp.maximum(ef[:, 0], 0)], 1.0)
    scale1 = jnp.where(valid1, face_scale[jnp.maximum(ef[:, 1], 0)], 1.0)

    n_valid = valid0.astype(jnp.float32) + valid1.astype(jnp.float32)
    edge_scale = (scale0 * valid0 + scale1 * valid1) / jnp.maximum(n_valid, 1.0)

    return initial_edge_lengths * edge_scale
