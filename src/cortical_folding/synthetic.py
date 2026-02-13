"""Synthetic test data generation for cortical folding simulation."""

import jax.numpy as jnp
import numpy as np
import trimesh

from .mesh import MeshTopology, build_topology, compute_face_areas
from .solver import SimState, SimParams, make_initial_state, simulate


def create_icosphere(
    subdivisions: int = 4, radius: float = 1.0
) -> tuple[jnp.ndarray, np.ndarray]:
    """Create an icosphere mesh using trimesh.

    Returns (vertices as jnp array, faces as np array).
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return jnp.array(mesh.vertices, dtype=jnp.float32), np.array(
        mesh.faces, dtype=np.int32
    )


def create_skull(
    radius: float = 1.5, center: tuple = (0.0, 0.0, 0.0)
) -> tuple[jnp.ndarray, float]:
    """Create skull boundary parameters. Returns (center, radius)."""
    return jnp.array(center, dtype=jnp.float32), radius


def create_uniform_growth(n_faces: int, rate: float = 0.5) -> jnp.ndarray:
    """Uniform growth rate for all faces."""
    return jnp.full(n_faces, rate)


def create_uniform_anisotropy(n_faces: int, value: float = 0.0) -> jnp.ndarray:
    """Uniform anisotropy weight per face."""
    return jnp.full(n_faces, value)


def create_regional_growth(
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    high_rate: float = 1.0,
    low_rate: float = 0.1,
    axis: int = 2,
    threshold: float = 0.0,
) -> jnp.ndarray:
    """Regional growth: high rate where face centroids are above threshold on given axis.

    Returns (F,) per-face growth rates.
    """
    centroids = (
        vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]
    ) / 3.0
    above = centroids[:, axis] > threshold
    return jnp.where(above, high_rate, low_rate)


def create_target_folded(
    vertices: jnp.ndarray,
    topo: MeshTopology,
    growth_rates: jnp.ndarray,
    params: SimParams,
    n_steps: int = 300,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run forward simulation to generate ground truth folded surface.

    Returns (final_vertices, trajectory).
    """
    initial_state = make_initial_state(vertices, topo)
    final_state, trajectory = simulate(
        initial_state, topo, growth_rates, params, n_steps
    )
    return final_state.vertices, trajectory
