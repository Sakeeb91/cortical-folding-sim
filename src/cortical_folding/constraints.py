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


def _accumulate_repulsion_from_pairs(
    verts: jnp.ndarray,
    topo: MeshTopology,
    idx_a: jnp.ndarray,
    idx_b: jnp.ndarray,
    min_dist: float,
    stiffness: float,
    pair_valid_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scatter-add pairwise repulsion forces and return active collision mask."""
    if pair_valid_mask is None:
        pair_valid_mask = jnp.ones_like(idx_a, dtype=bool)

    safe_idx_a = jnp.where(pair_valid_mask, idx_a, 0)
    safe_idx_b = jnp.where(pair_valid_mask, idx_b, 0)

    va = verts[safe_idx_a]  # (N, 3)
    vb = verts[safe_idx_b]  # (N, 3)
    diff = va - vb  # (N, 3)
    dist = jnp.linalg.norm(diff, axis=1, keepdims=True)  # (N, 1)
    safe_dist = jnp.maximum(dist, 1e-12)

    # Ignore trivial, invalid, and topological-neighbor pairs.
    same = safe_idx_a == safe_idx_b
    is_adjacent = _pair_adjacency_mask(safe_idx_a, safe_idx_b, topo.edges)
    valid = pair_valid_mask & (~same) & (~is_adjacent)

    gap = jnp.maximum(min_dist - dist, 0.0)
    repulsion = stiffness * gap * diff / safe_dist  # (N, 3)
    repulsion = jnp.where(valid[:, None], repulsion, 0.0)

    forces = jnp.zeros_like(verts)
    forces = forces.at[safe_idx_a].add(repulsion)
    forces = forces.at[safe_idx_b].add(-repulsion)
    active_collision = valid & (gap[:, 0] > 0.0)
    return forces, active_collision


def _spatial_hash_neighbor_pairs(
    verts: jnp.ndarray,
    cell_size: float,
    neighbor_window: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build deterministic local candidate pairs from spatial-hash ordering."""
    n_verts = verts.shape[0]
    safe_cell_size = jnp.maximum(cell_size, 1e-6)
    cell_coords = jnp.floor(verts / safe_cell_size).astype(jnp.int32)  # (V, 3)

    # Hash vertices to grid keys, then sort with index tie-break for determinism.
    coords32 = cell_coords.astype(jnp.int32)
    hash_keys = (
        coords32[:, 0] * jnp.int32(73856093)
        ^ coords32[:, 1] * jnp.int32(19349663)
        ^ coords32[:, 2] * jnp.int32(83492791)
    )
    sort_keys = hash_keys * jnp.int32(n_verts + 1) + jnp.arange(n_verts, dtype=jnp.int32)
    order = jnp.argsort(sort_keys)

    sorted_idx = order
    sorted_cells = cell_coords[order]
    base_idx = jnp.arange(n_verts, dtype=jnp.int32)

    pair_idx_a = jnp.zeros((neighbor_window, n_verts), dtype=jnp.int32)
    pair_idx_b = jnp.zeros((neighbor_window, n_verts), dtype=jnp.int32)
    pair_valid = jnp.zeros((neighbor_window, n_verts), dtype=bool)

    for offset in range(1, neighbor_window + 1):
        shifted_idx = jnp.roll(sorted_idx, -offset)
        shifted_cells = jnp.roll(sorted_cells, -offset, axis=0)
        in_range = base_idx < (n_verts - offset)
        near_cell = jnp.all(jnp.abs(sorted_cells - shifted_cells) <= 1, axis=1)
        valid = in_range & near_cell

        pair_idx_a = pair_idx_a.at[offset - 1].set(sorted_idx)
        pair_idx_b = pair_idx_b.at[offset - 1].set(shifted_idx)
        pair_valid = pair_valid.at[offset - 1].set(valid)

    return (
        pair_idx_a.reshape(-1),
        pair_idx_b.reshape(-1),
        pair_valid.reshape(-1),
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


def self_collision_penalty_spatial_hash(
    verts: jnp.ndarray,
    topo: MeshTopology,
    min_dist: float = 0.02,
    stiffness: float = 50.0,
    hash_cell_size: float = 0.02,
    hash_neighbor_window: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Collision penalty using spatial-hash local neighborhood candidates."""
    idx_a, idx_b, valid = _spatial_hash_neighbor_pairs(
        verts=verts,
        cell_size=hash_cell_size,
        neighbor_window=max(1, int(hash_neighbor_window)),
    )
    forces, active_collision = _accumulate_repulsion_from_pairs(
        verts=verts,
        topo=topo,
        idx_a=idx_a,
        idx_b=idx_b,
        min_dist=min_dist,
        stiffness=stiffness,
        pair_valid_mask=valid,
    )
    return forces, active_collision


def self_collision_penalty(
    verts: jnp.ndarray,
    topo: MeshTopology,
    min_dist: float = 0.02,
    stiffness: float = 50.0,
    n_sample: int = 256,
    use_spatial_hash: bool = False,
    hash_cell_size: float | None = None,
    hash_neighbor_window: int = 8,
    deterministic_fallback: bool = True,
    fallback_n_sample: int = 256,
    key=None,
) -> jnp.ndarray:
    """Approximate self-collision penalty via sampled or spatial-hash pairs.

    Spatial-hash mode uses deterministic local neighborhoods. If no active
    collisions are found, deterministic sampling can be used as a fallback.
    """
    if use_spatial_hash:
        cell_size = min_dist if hash_cell_size is None else hash_cell_size
        hash_forces, active = self_collision_penalty_spatial_hash(
            verts=verts,
            topo=topo,
            min_dist=min_dist,
            stiffness=stiffness,
            hash_cell_size=cell_size,
            hash_neighbor_window=hash_neighbor_window,
        )
        if deterministic_fallback:
            idx_a, idx_b = _deterministic_pair_indices(
                verts.shape[0], max(1, int(fallback_n_sample))
            )
            sampled_forces, _ = _accumulate_repulsion_from_pairs(
                verts=verts,
                topo=topo,
                idx_a=idx_a,
                idx_b=idx_b,
                min_dist=min_dist,
                stiffness=stiffness,
            )
            use_fallback = jnp.sum(active.astype(jnp.int32)) == 0
            return jax.lax.cond(use_fallback, lambda _: sampled_forces, lambda _: hash_forces, None)
        return hash_forces

    if key is None:
        # Deterministic pseudo-randomized pairing when no key is passed.
        idx_a, idx_b = _deterministic_pair_indices(verts.shape[0], n_sample)
    else:
        n_verts = verts.shape[0]
        k1, k2 = jax.random.split(key)
        idx_a = jax.random.randint(k1, (n_sample,), 0, n_verts)
        idx_b = jax.random.randint(k2, (n_sample,), 0, n_verts)

    forces, _ = _accumulate_repulsion_from_pairs(
        verts=verts,
        topo=topo,
        idx_a=idx_a,
        idx_b=idx_b,
        min_dist=min_dist,
        stiffness=stiffness,
    )
    return forces
