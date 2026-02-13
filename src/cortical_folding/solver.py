"""Time integration solver with lax.scan for cortical folding simulation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .mesh import MeshTopology, compute_edge_lengths, compute_face_areas, compute_mean_curvature
from .physics import (
    elastic_force,
    bending_force,
    grow_rest_areas,
    update_rest_lengths_plasticity,
    update_rest_lengths_from_areas,
)
from .constraints import skull_penalty, self_collision_penalty


class SimState(NamedTuple):
    """Dynamic simulation state — evolved at each timestep."""

    vertices: jnp.ndarray  # (V, 3)
    velocities: jnp.ndarray  # (V, 3)
    rest_lengths: jnp.ndarray  # (E,)
    rest_areas: jnp.ndarray  # (F,)
    rest_curvatures: jnp.ndarray  # (V,)


class SimParams(NamedTuple):
    """Simulation parameters."""

    Kc: float = 1.0  # elastic stiffness
    Kb: float = 5.0  # bending stiffness
    damping: float = 0.8  # velocity damping coefficient
    skull_center: jnp.ndarray = jnp.zeros(3)
    skull_radius: float = 1.5
    skull_stiffness: float = 100.0
    carrying_cap_factor: float = 3.0  # carrying capacity = initial_area * factor
    tau: float = 1000.0  # plasticity timescale
    dt: float = 0.05
    # Robustness controls
    max_growth_rate: float = 2.0
    min_rest_area: float = 1e-8
    min_rest_length: float = 1e-8
    max_force_norm: float = 1e3
    max_acc_norm: float = 1e3
    max_velocity_norm: float = 5.0
    max_displacement_per_step: float = 0.05
    enable_self_collision: bool = False
    self_collision_min_dist: float = 0.02
    self_collision_stiffness: float = 50.0
    self_collision_n_sample: int = 256
    self_collision_use_spatial_hash: bool = False
    self_collision_hash_cell_size: float = 0.02
    self_collision_hash_neighbor_window: int = 8
    self_collision_deterministic_fallback: bool = True
    self_collision_fallback_n_sample: int = 256
    # Anisotropic rest-length growth controls
    anisotropy_strength: float = 0.0
    anisotropy_axis: jnp.ndarray = jnp.array([0.0, 0.0, 1.0])


def _clip_vectors_norm(vectors: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    """Clip per-vector L2 norm to avoid unstable updates."""
    if max_norm <= 0:
        return vectors
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norms, 1e-12))
    return vectors * scale


def _finite_or_previous(new_val: jnp.ndarray, previous_val: jnp.ndarray) -> jnp.ndarray:
    """Replace non-finite entries with previous state values."""
    return jnp.where(jnp.isfinite(new_val), new_val, previous_val)


def _normalize_axis(axis: jnp.ndarray) -> jnp.ndarray:
    """Return unit axis with safe fallback."""
    norm = jnp.linalg.norm(axis)
    return axis / jnp.maximum(norm, 1e-12)


def _edge_axis_alignment(verts: jnp.ndarray, topo: MeshTopology, axis: jnp.ndarray) -> jnp.ndarray:
    """Return absolute edge direction alignment to an axis. Shape (E,)."""
    edge_vecs = verts[topo.edges[:, 1]] - verts[topo.edges[:, 0]]
    edge_dirs = edge_vecs / jnp.maximum(jnp.linalg.norm(edge_vecs, axis=1, keepdims=True), 1e-12)
    return jnp.abs(jnp.sum(edge_dirs * axis[None, :], axis=1))


def _edge_face_values(face_values: jnp.ndarray, topo: MeshTopology) -> jnp.ndarray:
    """Average face values onto edges using adjacent faces."""
    ef = topo.edge_faces
    valid0 = ef[:, 0] >= 0
    valid1 = ef[:, 1] >= 0
    v0 = jnp.where(valid0, face_values[jnp.maximum(ef[:, 0], 0)], 0.0)
    v1 = jnp.where(valid1, face_values[jnp.maximum(ef[:, 1], 0)], 0.0)
    denom = valid0.astype(jnp.float32) + valid1.astype(jnp.float32)
    return (v0 + v1) / jnp.maximum(denom, 1.0)


def make_initial_state(
    vertices: jnp.ndarray, topo: MeshTopology
) -> SimState:
    """Create initial simulation state from mesh vertices and topology."""
    rest_lengths = compute_edge_lengths(vertices, topo)
    rest_areas = compute_face_areas(vertices, topo)
    rest_curvatures = compute_mean_curvature(vertices, topo)
    velocities = jnp.zeros_like(vertices)
    return SimState(
        vertices=vertices,
        velocities=velocities,
        rest_lengths=rest_lengths,
        rest_areas=rest_areas,
        rest_curvatures=rest_curvatures,
    )


def simulation_step(
    state: SimState,
    topo: MeshTopology,
    growth_rates: jnp.ndarray,
    face_anisotropy: jnp.ndarray,
    params: SimParams,
    initial_edge_lengths: jnp.ndarray,
    initial_areas: jnp.ndarray,
) -> SimState:
    """Single timestep: forces → integrate → grow."""
    dt = params.dt
    safe_growth_rates = jnp.clip(growth_rates, 0.0, params.max_growth_rate)

    # --- Forces ---
    f_elastic = elastic_force(state.vertices, topo, state.rest_lengths, params.Kc)
    f_bending = bending_force(
        state.vertices, topo, state.rest_curvatures, params.Kb
    )
    f_skull = skull_penalty(
        state.vertices, params.skull_center, params.skull_radius, params.skull_stiffness
    )
    f_collision = jnp.zeros_like(state.vertices)
    if params.enable_self_collision and params.self_collision_stiffness > 0:
        f_collision = self_collision_penalty(
            state.vertices,
            topo,
            min_dist=params.self_collision_min_dist,
            stiffness=params.self_collision_stiffness,
            n_sample=params.self_collision_n_sample,
            use_spatial_hash=params.self_collision_use_spatial_hash,
            hash_cell_size=params.self_collision_hash_cell_size,
            hash_neighbor_window=params.self_collision_hash_neighbor_window,
            deterministic_fallback=params.self_collision_deterministic_fallback,
            fallback_n_sample=params.self_collision_fallback_n_sample,
        )
    f_total = _clip_vectors_norm(
        f_elastic + f_bending + f_skull + f_collision,
        params.max_force_norm,
    )

    # --- Damped Newmark explicit integration ---
    acc = _clip_vectors_norm(
        f_total - params.damping * state.velocities,
        params.max_acc_norm,
    )
    step_disp = dt * state.velocities + 0.5 * dt**2 * acc
    step_disp = _clip_vectors_norm(step_disp, params.max_displacement_per_step)
    new_verts = state.vertices + step_disp
    new_vel = _clip_vectors_norm(state.velocities + dt * acc, params.max_velocity_norm)

    # --- Growth: update rest areas and lengths ---
    carrying_cap = jnp.maximum(
        initial_areas * params.carrying_cap_factor,
        params.min_rest_area,
    )
    new_rest_areas = grow_rest_areas(
        state.rest_areas, safe_growth_rates, carrying_cap, dt
    )
    new_rest_areas = jnp.maximum(new_rest_areas, params.min_rest_area)

    # Scale rest lengths based on area growth
    new_rest_lengths = update_rest_lengths_from_areas(
        initial_edge_lengths, new_rest_areas, initial_areas, topo
    )
    if params.anisotropy_strength > 0:
        axis = _normalize_axis(params.anisotropy_axis)
        edge_align = _edge_axis_alignment(state.vertices, topo, axis)  # (E,)
        edge_face_aniso = _edge_face_values(face_anisotropy, topo)  # (E,)
        # Center alignment so anisotropy redistributes growth directionally.
        centered_align = edge_align - jnp.mean(edge_align)
        aniso_scale = 1.0 + params.anisotropy_strength * edge_face_aniso * centered_align
        aniso_scale = jnp.clip(aniso_scale, 0.5, 1.5)
        new_rest_lengths = new_rest_lengths * aniso_scale

    # Also apply plasticity
    current_lengths = compute_edge_lengths(new_verts, topo)
    new_rest_lengths = update_rest_lengths_plasticity(
        new_rest_lengths, current_lengths, params.tau, dt
    )
    new_rest_lengths = jnp.maximum(new_rest_lengths, params.min_rest_length)

    # Finite guards to keep simulation recoverable in long runs.
    new_verts = _finite_or_previous(new_verts, state.vertices)
    new_vel = _finite_or_previous(new_vel, state.velocities)
    new_rest_areas = _finite_or_previous(new_rest_areas, state.rest_areas)
    new_rest_lengths = _finite_or_previous(new_rest_lengths, state.rest_lengths)

    return SimState(
        vertices=new_verts,
        velocities=new_vel,
        rest_lengths=new_rest_lengths,
        rest_areas=new_rest_areas,
        rest_curvatures=state.rest_curvatures,
    )


def simulate(
    initial_state: SimState,
    topo: MeshTopology,
    growth_rates: jnp.ndarray,
    params: SimParams,
    face_anisotropy: jnp.ndarray | None = None,
    n_steps: int = 200,
    save_every: int = 1,
) -> tuple[SimState, jnp.ndarray]:
    """Run full simulation via lax.scan with gradient checkpointing.

    Returns (final_state, trajectory) where trajectory is (n_steps, V, 3).
    """
    initial_edge_lengths = initial_state.rest_lengths
    initial_areas = initial_state.rest_areas
    if face_anisotropy is None:
        face_anisotropy = jnp.zeros(topo.faces.shape[0], dtype=initial_state.vertices.dtype)

    @jax.checkpoint
    def step_fn(state, _):
        new_state = simulation_step(
            state, topo, growth_rates, face_anisotropy, params,
            initial_edge_lengths, initial_areas,
        )
        return new_state, new_state.vertices

    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    final_state, trajectory = jax.lax.scan(
        step_fn, initial_state, jnp.arange(n_steps)
    )
    return final_state, trajectory[::save_every]
