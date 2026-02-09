"""Loss functions for inverse cortical folding problem."""

from typing import NamedTuple

import jax.numpy as jnp

from .mesh import MeshTopology, compute_mean_curvature, compute_face_areas


class LossWeights(NamedTuple):
    curv: float = 1.0
    gi: float = 10.0
    vertex: float = 0.1


def curvature_loss(
    pred_verts: jnp.ndarray,
    target_verts: jnp.ndarray,
    topo: MeshTopology,
) -> float:
    """MSE of mean curvature between predicted and target surfaces."""
    H_pred = compute_mean_curvature(pred_verts, topo)
    H_target = compute_mean_curvature(target_verts, topo)
    return jnp.mean((H_pred - H_target) ** 2)


def gyrification_index(
    verts: jnp.ndarray, topo: MeshTopology, skull_radius: float
) -> float:
    """Ratio of actual cortical area to smooth sphere area."""
    actual_area = jnp.sum(compute_face_areas(verts, topo))
    smooth_area = 4.0 * jnp.pi * skull_radius**2
    return actual_area / smooth_area


def gi_loss(
    pred_verts: jnp.ndarray,
    target_gi: float,
    topo: MeshTopology,
    skull_radius: float,
) -> float:
    """Squared error between predicted and target gyrification index."""
    pred_gi = gyrification_index(pred_verts, topo, skull_radius)
    return (pred_gi - target_gi) ** 2


def vertex_loss(pred_verts: jnp.ndarray, target_verts: jnp.ndarray) -> float:
    """Mean squared vertex position error."""
    return jnp.mean(jnp.sum((pred_verts - target_verts) ** 2, axis=1))


def total_loss(
    pred_verts: jnp.ndarray,
    target_verts: jnp.ndarray,
    topo: MeshTopology,
    skull_radius: float,
    target_gi: float,
    weights: LossWeights = LossWeights(),
) -> float:
    """Combined loss for inverse problem."""
    l_curv = curvature_loss(pred_verts, target_verts, topo)
    l_gi = gi_loss(pred_verts, target_gi, topo, skull_radius)
    l_vert = vertex_loss(pred_verts, target_verts)
    return weights.curv * l_curv + weights.gi * l_gi + weights.vertex * l_vert
