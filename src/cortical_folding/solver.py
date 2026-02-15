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
    self_collision_blend_sampled_weight: float = 0.0
    enable_adaptive_substepping: bool = False
    adaptive_substep_min: int = 1
    adaptive_substep_max: int = 4
    adaptive_target_disp: float = 0.01
    adaptive_force_safety_scale: float = 1.0
    fail_on_nonfinite: bool = False
    high_fidelity: bool = False
    # Anisotropic rest-length growth controls
    anisotropy_strength: float = 0.0
    anisotropy_axis: jnp.ndarray = jnp.array([0.0, 0.0, 1.0])
    # Experimental two-layer approximation controls
    enable_two_layer_approx: bool = False
    two_layer_axis: jnp.ndarray = jnp.array([0.0, 0.0, 1.0])
    two_layer_threshold: float = 0.0
    two_layer_transition_sharpness: float = 6.0
    outer_layer_growth_scale: float = 1.15
    inner_layer_growth_scale: float = 0.85
    two_layer_coupling: float = 0.1


class ForceComponents(NamedTuple):
    """Per-vertex force decomposition at a single simulation state."""

    elastic: jnp.ndarray
    bending: jnp.ndarray
    skull: jnp.ndarray
    collision: jnp.ndarray
    total: jnp.ndarray


def _clip_vectors_norm(vectors: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    """Clip per-vector L2 norm to avoid unstable updates."""
    max_norm_arr = jnp.asarray(max_norm, dtype=vectors.dtype)
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    scale = jnp.minimum(1.0, max_norm_arr / jnp.maximum(norms, 1e-12))
    clipped = vectors * scale
    return jax.lax.cond(max_norm_arr <= 0, lambda _: vectors, lambda _: clipped, None)


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


def _face_layer_blend(
    verts: jnp.ndarray,
    topo: MeshTopology,
    axis: jnp.ndarray,
    threshold: float,
    sharpness: float,
) -> jnp.ndarray:
    """Compute smooth per-face layer blend in [0, 1]."""
    centroids = (
        verts[topo.faces[:, 0]] + verts[topo.faces[:, 1]] + verts[topo.faces[:, 2]]
    ) / 3.0
    unit_axis = _normalize_axis(axis)
    signed_dist = jnp.sum(centroids * unit_axis[None, :], axis=1) - threshold
    safe_sharpness = jnp.maximum(sharpness, 1e-6)
    return jax.nn.sigmoid(safe_sharpness * signed_dist)


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


def compute_force_components(
    state: SimState,
    topo: MeshTopology,
    params: SimParams,
) -> ForceComponents:
    """Compute raw per-vertex force decomposition for diagnostics."""
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
            sampled_blend_weight=params.self_collision_blend_sampled_weight,
        )
    return ForceComponents(
        elastic=f_elastic,
        bending=f_bending,
        skull=f_skull,
        collision=f_collision,
        total=f_elastic + f_bending + f_skull + f_collision,
    )


def _adaptive_substep_count(
    state: SimState,
    topo: MeshTopology,
    params: SimParams,
) -> jnp.ndarray:
    """Compute deterministic adaptive substep count from current state."""
    if not params.enable_adaptive_substepping:
        return jnp.int32(1)
    max_substeps = max(1, int(params.adaptive_substep_max))
    min_substeps = min(max_substeps, max(1, int(params.adaptive_substep_min)))

    # Estimate displacement with a conservative force safety multiplier.
    force_components = compute_force_components(state, topo, params)
    force_cap = params.max_force_norm * jnp.maximum(params.adaptive_force_safety_scale, 1e-6)
    f_total = _clip_vectors_norm(force_components.total, force_cap)
    acc = _clip_vectors_norm(
        f_total - params.damping * state.velocities,
        params.max_acc_norm,
    )
    pred_disp = params.dt * state.velocities + 0.5 * params.dt**2 * acc
    pred_norm = jnp.linalg.norm(pred_disp, axis=1)
    max_pred_disp = jnp.max(pred_norm)
    target = jnp.maximum(params.adaptive_target_disp, 1e-6)
    raw = jnp.ceil(max_pred_disp / target).astype(jnp.int32)
    raw = jnp.maximum(raw, 1)
    return jnp.clip(raw, min_substeps, max_substeps)


def _simulation_substep(
    state: SimState,
    topo: MeshTopology,
    growth_rates: jnp.ndarray,
    face_anisotropy: jnp.ndarray,
    face_layer_blend: jnp.ndarray,
    params: SimParams,
    initial_edge_lengths: jnp.ndarray,
    initial_areas: jnp.ndarray,
    dt: jnp.ndarray,
) -> SimState:
    """Single deterministic substep: forces → integrate → grow."""
    safe_growth_rates = jnp.clip(growth_rates, 0.0, params.max_growth_rate)
    if params.enable_two_layer_approx:
        inner_scale = jnp.maximum(params.inner_layer_growth_scale, 0.0)
        outer_scale = jnp.maximum(params.outer_layer_growth_scale, 0.0)
        layer_scale = inner_scale + (outer_scale - inner_scale) * face_layer_blend
        safe_growth_rates = safe_growth_rates * layer_scale
        coupling = jnp.clip(params.two_layer_coupling, 0.0, 1.0)
        safe_growth_rates = (
            (1.0 - coupling) * safe_growth_rates
            + coupling * jnp.mean(safe_growth_rates)
        )
        safe_growth_rates = jnp.clip(safe_growth_rates, 0.0, params.max_growth_rate)

    # --- Forces ---
    force_components = compute_force_components(state, topo, params)
    f_total = _clip_vectors_norm(force_components.total, params.max_force_norm)

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
    if not params.fail_on_nonfinite:
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


def simulation_step(
    state: SimState,
    topo: MeshTopology,
    growth_rates: jnp.ndarray,
    face_anisotropy: jnp.ndarray,
    face_layer_blend: jnp.ndarray,
    params: SimParams,
    initial_edge_lengths: jnp.ndarray,
    initial_areas: jnp.ndarray,
) -> SimState:
    """Single timestep with optional deterministic adaptive substepping."""
    if not params.enable_adaptive_substepping:
        return _simulation_substep(
            state=state,
            topo=topo,
            growth_rates=growth_rates,
            face_anisotropy=face_anisotropy,
            face_layer_blend=face_layer_blend,
            params=params,
            initial_edge_lengths=initial_edge_lengths,
            initial_areas=initial_areas,
            dt=jnp.asarray(params.dt, dtype=state.vertices.dtype),
        )

    max_substeps = max(1, int(params.adaptive_substep_max))
    n_substeps = _adaptive_substep_count(state, topo, params)
    dt_sub = jnp.asarray(params.dt, dtype=state.vertices.dtype) / jnp.maximum(
        n_substeps.astype(state.vertices.dtype), 1.0
    )

    def substep_fn(carry: SimState, step_idx: jnp.ndarray) -> tuple[SimState, None]:
        def run_one(sub_state: SimState) -> SimState:
            return _simulation_substep(
                state=sub_state,
                topo=topo,
                growth_rates=growth_rates,
                face_anisotropy=face_anisotropy,
                face_layer_blend=face_layer_blend,
                params=params,
                initial_edge_lengths=initial_edge_lengths,
                initial_areas=initial_areas,
                dt=dt_sub,
            )

        updated = jax.lax.cond(step_idx < n_substeps, run_one, lambda s: s, carry)
        return updated, None

    final_state, _ = jax.lax.scan(substep_fn, state, jnp.arange(max_substeps))
    return final_state


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
    if params.enable_two_layer_approx:
        face_layer_blend = _face_layer_blend(
            initial_state.vertices,
            topo,
            params.two_layer_axis,
            params.two_layer_threshold,
            params.two_layer_transition_sharpness,
        ).astype(initial_state.vertices.dtype)
    else:
        face_layer_blend = jnp.zeros(topo.faces.shape[0], dtype=initial_state.vertices.dtype)

    @jax.checkpoint
    def step_fn(state, _):
        new_state = simulation_step(
            state, topo, growth_rates, face_anisotropy, face_layer_blend, params,
            initial_edge_lengths, initial_areas,
        )
        return new_state, new_state.vertices

    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    final_state, trajectory = jax.lax.scan(
        step_fn, initial_state, jnp.arange(n_steps)
    )
    return final_state, trajectory[::save_every]
