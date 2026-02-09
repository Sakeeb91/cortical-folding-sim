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
from .constraints import skull_penalty


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
    params: SimParams,
    initial_edge_lengths: jnp.ndarray,
    initial_areas: jnp.ndarray,
) -> SimState:
    """Single timestep: forces → integrate → grow."""
    dt = params.dt

    # --- Forces ---
    f_elastic = elastic_force(state.vertices, topo, state.rest_lengths, params.Kc)
    f_bending = bending_force(
        state.vertices, topo, state.rest_curvatures, params.Kb
    )
    f_skull = skull_penalty(
        state.vertices, params.skull_center, params.skull_radius, params.skull_stiffness
    )
    f_total = f_elastic + f_bending + f_skull

    # --- Damped Newmark explicit integration ---
    acc = f_total - params.damping * state.velocities
    new_verts = state.vertices + dt * state.velocities + 0.5 * dt**2 * acc
    new_vel = state.velocities + dt * acc

    # --- Growth: update rest areas and lengths ---
    carrying_cap = initial_areas * params.carrying_cap_factor
    new_rest_areas = grow_rest_areas(state.rest_areas, growth_rates, carrying_cap, dt)

    # Scale rest lengths based on area growth
    new_rest_lengths = update_rest_lengths_from_areas(
        initial_edge_lengths, new_rest_areas, initial_areas, topo
    )

    # Also apply plasticity
    current_lengths = compute_edge_lengths(new_verts, topo)
    new_rest_lengths = update_rest_lengths_plasticity(
        new_rest_lengths, current_lengths, params.tau, dt
    )

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
    n_steps: int = 200,
    save_every: int = 1,
) -> tuple[SimState, jnp.ndarray]:
    """Run full simulation via lax.scan with gradient checkpointing.

    Returns (final_state, trajectory) where trajectory is (n_steps, V, 3).
    """
    initial_edge_lengths = initial_state.rest_lengths
    initial_areas = initial_state.rest_areas

    @jax.checkpoint
    def step_fn(state, _):
        new_state = simulation_step(
            state, topo, growth_rates, params,
            initial_edge_lengths, initial_areas,
        )
        return new_state, new_state.vertices

    final_state, trajectory = jax.lax.scan(
        step_fn, initial_state, jnp.arange(n_steps)
    )
    return final_state, trajectory
