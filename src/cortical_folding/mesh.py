"""Mesh data structures and differential geometry operators."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class MeshTopology(NamedTuple):
    """Static mesh connectivity — built once from numpy, then used as JAX arrays."""

    faces: jnp.ndarray  # (F, 3) int32
    edges: jnp.ndarray  # (E, 2) int32 — unique undirected edges
    edge_faces: jnp.ndarray  # (E, 2) int32 — adjacent faces per edge (-1 if boundary)
    edge_opposite_verts: jnp.ndarray  # (E, 2) int32 — opposite verts for cotangent weights
    vertex_faces: jnp.ndarray  # (V, max_degree) int32 — padded
    vertex_face_mask: jnp.ndarray  # (V, max_degree) bool


def build_topology(vertices: np.ndarray, faces: np.ndarray) -> MeshTopology:
    """Build mesh topology from vertex positions and face indices.

    Uses numpy (not JAX) — intended to run once at setup time.
    """
    n_verts = len(vertices)
    faces = np.asarray(faces, dtype=np.int32)

    # --- Extract unique undirected edges and map edges to faces ---
    edge_to_idx = {}
    edge_list = []
    edge_face_list = {}  # edge_idx -> [face_idx, ...]
    edge_opposite_list = {}  # edge_idx -> [opposite_vert, ...]

    for fi, face in enumerate(faces):
        for k in range(3):
            v0, v1 = int(face[k]), int(face[(k + 1) % 3])
            opp = int(face[(k + 2) % 3])
            key = (min(v0, v1), max(v0, v1))
            if key not in edge_to_idx:
                edge_to_idx[key] = len(edge_list)
                edge_list.append(key)
                edge_face_list[edge_to_idx[key]] = []
                edge_opposite_list[edge_to_idx[key]] = []
            eidx = edge_to_idx[key]
            edge_face_list[eidx].append(fi)
            edge_opposite_list[eidx].append(opp)

    n_edges = len(edge_list)
    edges = np.array(edge_list, dtype=np.int32)

    # Pad edge_faces and edge_opposite_verts to exactly 2
    edge_faces = np.full((n_edges, 2), -1, dtype=np.int32)
    edge_opposite_verts = np.full((n_edges, 2), -1, dtype=np.int32)
    for eidx in range(n_edges):
        flist = edge_face_list[eidx]
        olist = edge_opposite_list[eidx]
        for j in range(min(len(flist), 2)):
            edge_faces[eidx, j] = flist[j]
            edge_opposite_verts[eidx, j] = olist[j]

    # --- Vertex-to-face adjacency (padded) ---
    vert_faces_raw = [[] for _ in range(n_verts)]
    for fi, face in enumerate(faces):
        for v in face:
            vert_faces_raw[int(v)].append(fi)
    max_degree = max(len(vf) for vf in vert_faces_raw) if vert_faces_raw else 1

    vertex_faces = np.full((n_verts, max_degree), 0, dtype=np.int32)
    vertex_face_mask = np.zeros((n_verts, max_degree), dtype=bool)
    for vi, vf in enumerate(vert_faces_raw):
        for j, fi in enumerate(vf):
            vertex_faces[vi, j] = fi
            vertex_face_mask[vi, j] = True

    return MeshTopology(
        faces=jnp.array(faces),
        edges=jnp.array(edges),
        edge_faces=jnp.array(edge_faces),
        edge_opposite_verts=jnp.array(edge_opposite_verts),
        vertex_faces=jnp.array(vertex_faces),
        vertex_face_mask=jnp.array(vertex_face_mask),
    )


# ---------------------------------------------------------------------------
# Geometry operators (all differentiable via JAX)
# ---------------------------------------------------------------------------


def compute_face_normals(verts: jnp.ndarray, topo: MeshTopology) -> jnp.ndarray:
    """Compute unit face normals. Returns (F, 3)."""
    v0 = verts[topo.faces[:, 0]]
    v1 = verts[topo.faces[:, 1]]
    v2 = verts[topo.faces[:, 2]]
    cross = jnp.cross(v1 - v0, v2 - v0)
    norms = jnp.linalg.norm(cross, axis=1, keepdims=True)
    return cross / jnp.maximum(norms, 1e-12)


def compute_face_areas(verts: jnp.ndarray, topo: MeshTopology) -> jnp.ndarray:
    """Compute face areas. Returns (F,)."""
    v0 = verts[topo.faces[:, 0]]
    v1 = verts[topo.faces[:, 1]]
    v2 = verts[topo.faces[:, 2]]
    cross = jnp.cross(v1 - v0, v2 - v0)
    return 0.5 * jnp.linalg.norm(cross, axis=1)


def compute_vertex_normals(verts: jnp.ndarray, topo: MeshTopology) -> jnp.ndarray:
    """Area-weighted vertex normals. Returns (V, 3)."""
    face_normals = compute_face_normals(verts, topo)
    face_areas = compute_face_areas(verts, topo)
    weighted = face_normals * face_areas[:, None]  # (F, 3)

    # Gather per-vertex: use vertex_faces and mask
    gathered = weighted[topo.vertex_faces]  # (V, max_deg, 3)
    mask = topo.vertex_face_mask[:, :, None]  # (V, max_deg, 1)
    summed = jnp.sum(gathered * mask, axis=1)  # (V, 3)
    norms = jnp.linalg.norm(summed, axis=1, keepdims=True)
    return summed / jnp.maximum(norms, 1e-12)


def compute_edge_lengths(verts: jnp.ndarray, topo: MeshTopology) -> jnp.ndarray:
    """Compute edge lengths. Returns (E,)."""
    edge_vecs = verts[topo.edges[:, 1]] - verts[topo.edges[:, 0]]
    return jnp.linalg.norm(edge_vecs, axis=1)


def compute_vertex_areas(verts: jnp.ndarray, topo: MeshTopology) -> jnp.ndarray:
    """Mixed Voronoi area per vertex (approximated as 1/3 sum of incident face areas).

    Returns (V,).
    """
    face_areas = compute_face_areas(verts, topo)
    gathered = face_areas[topo.vertex_faces]  # (V, max_deg)
    masked = gathered * topo.vertex_face_mask  # (V, max_deg)
    return jnp.sum(masked, axis=1) / 3.0


def _cotangent_weights(verts: jnp.ndarray, topo: MeshTopology) -> jnp.ndarray:
    """Compute cotangent weights per edge. Returns (E,).

    For edge (i, j), the weight is 0.5 * (cot(alpha) + cot(beta)) where
    alpha, beta are the angles opposite the edge in the two adjacent faces.
    """
    vi = verts[topo.edges[:, 0]]  # (E, 3)
    vj = verts[topo.edges[:, 1]]  # (E, 3)

    def _cot_for_side(side: int) -> jnp.ndarray:
        opp_idx = topo.edge_opposite_verts[:, side]  # (E,)
        valid = opp_idx >= 0
        # Use 0 index for invalid entries (will be masked out)
        safe_idx = jnp.where(valid, opp_idx, 0)
        vopp = verts[safe_idx]  # (E, 3)
        a = vi - vopp
        b = vj - vopp
        dot = jnp.sum(a * b, axis=1)
        cross_norm = jnp.linalg.norm(jnp.cross(a, b), axis=1)
        cot = dot / jnp.maximum(cross_norm, 1e-12)
        # Clamp for numerical stability
        cot = jnp.clip(cot, -1e4, 1e4)
        return jnp.where(valid, cot, 0.0)

    return 0.5 * (_cot_for_side(0) + _cot_for_side(1))


def compute_cotangent_laplacian(
    verts: jnp.ndarray, topo: MeshTopology
) -> jnp.ndarray:
    """Cotangent Laplacian applied to vertex positions. Returns (V, 3).

    Lx_i = sum_j w_ij (x_j - x_i)  where w_ij = cotangent weight.
    """
    weights = _cotangent_weights(verts, topo)  # (E,)
    vi = verts[topo.edges[:, 0]]  # (E, 3)
    vj = verts[topo.edges[:, 1]]  # (E, 3)

    # Weighted edge difference
    diff_ij = weights[:, None] * (vj - vi)  # (E, 3)

    laplacian = jnp.zeros_like(verts)
    laplacian = laplacian.at[topo.edges[:, 0]].add(diff_ij)
    laplacian = laplacian.at[topo.edges[:, 1]].add(-diff_ij)
    return laplacian


def compute_mean_curvature(verts: jnp.ndarray, topo: MeshTopology) -> jnp.ndarray:
    """Mean curvature magnitude per vertex via cotangent Laplacian.

    H = |Lx| / (2 * A_v).  Returns (V,).
    """
    lap = compute_cotangent_laplacian(verts, topo)  # (V, 3)
    lap_norm = jnp.linalg.norm(lap, axis=1)  # (V,)
    areas = compute_vertex_areas(verts, topo)  # (V,)
    return lap_norm / (2.0 * jnp.maximum(areas, 1e-12))


def compute_gaussian_curvature(
    verts: jnp.ndarray, topo: MeshTopology
) -> jnp.ndarray:
    """Gaussian curvature via angle defect: K = (2pi - sum_theta) / A_v.

    Returns (V,).
    """
    # Compute angles at each vertex in each incident face
    v0 = verts[topo.faces[:, 0]]  # (F, 3)
    v1 = verts[topo.faces[:, 1]]
    v2 = verts[topo.faces[:, 2]]

    def _angle_at(a, b, c):
        """Angle at vertex a in triangle (a, b, c)."""
        ab = b - a
        ac = c - a
        cos_val = jnp.sum(ab * ac, axis=1) / (
            jnp.linalg.norm(ab, axis=1) * jnp.linalg.norm(ac, axis=1) + 1e-12
        )
        return jnp.arccos(jnp.clip(cos_val, -1.0 + 1e-7, 1.0 - 1e-7))

    angles0 = _angle_at(v0, v1, v2)  # (F,) angle at vertex 0
    angles1 = _angle_at(v1, v2, v0)
    angles2 = _angle_at(v2, v0, v1)

    # Scatter angle sums to vertices
    n_verts = verts.shape[0]
    angle_sum = jnp.zeros(n_verts)
    angle_sum = angle_sum.at[topo.faces[:, 0]].add(angles0)
    angle_sum = angle_sum.at[topo.faces[:, 1]].add(angles1)
    angle_sum = angle_sum.at[topo.faces[:, 2]].add(angles2)

    areas = compute_vertex_areas(verts, topo)
    return (2.0 * jnp.pi - angle_sum) / jnp.maximum(areas, 1e-12)
