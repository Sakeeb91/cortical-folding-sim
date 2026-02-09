"""Neural network for predicting spatially-varying growth fields."""

import jax
import jax.numpy as jnp
import equinox as eqx

from .mesh import (
    MeshTopology,
    compute_vertex_normals,
    compute_mean_curvature,
    compute_gaussian_curvature,
    compute_vertex_areas,
    compute_edge_lengths,
)


class GrowthFieldNet(eqx.Module):
    """Per-vertex MLP that predicts growth rate from local geometry features."""

    layers: list

    def __init__(self, key, feature_dim: int = 10, hidden: int = 64):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(feature_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, 1, key=k3),
        ]

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Predict growth rates from per-vertex features. (V, D) -> (V,)."""
        return jax.vmap(self._per_vertex)(features).squeeze(-1)

    def _per_vertex(self, f: jnp.ndarray) -> jnp.ndarray:
        """Single vertex: (D,) -> (1,)."""
        x = f
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return jax.nn.softplus(self.layers[-1](x))  # ensure m > 0


def extract_vertex_features(
    verts: jnp.ndarray, topo: MeshTopology
) -> jnp.ndarray:
    """Extract per-vertex geometry features for the growth network.

    Features: [x, y, z, nx, ny, nz, H, K, area, mean_edge_len] -> (V, 10).
    """
    normals = compute_vertex_normals(verts, topo)  # (V, 3)
    H = compute_mean_curvature(verts, topo)[:, None]  # (V, 1)
    K = compute_gaussian_curvature(verts, topo)[:, None]  # (V, 1)
    areas = compute_vertex_areas(verts, topo)[:, None]  # (V, 1)

    # Mean edge length per vertex (approximate via total edge len / num edges)
    edge_lens = compute_edge_lengths(verts, topo)
    mean_el = jnp.full((verts.shape[0], 1), jnp.mean(edge_lens))

    return jnp.concatenate([verts, normals, H, K, areas, mean_el], axis=1)


def growth_rates_to_faces(
    vertex_growth: jnp.ndarray, topo: MeshTopology
) -> jnp.ndarray:
    """Convert per-vertex growth rates to per-face by averaging face vertices.

    Returns (F,).
    """
    g0 = vertex_growth[topo.faces[:, 0]]
    g1 = vertex_growth[topo.faces[:, 1]]
    g2 = vertex_growth[topo.faces[:, 2]]
    return (g0 + g1 + g2) / 3.0
