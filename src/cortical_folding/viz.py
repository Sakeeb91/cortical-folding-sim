"""Visualization utilities for cortical folding simulation."""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .mesh import MeshTopology, compute_mean_curvature, compute_face_areas


def plot_mesh(
    verts: jnp.ndarray,
    faces: jnp.ndarray,
    scalars: jnp.ndarray | None = None,
    title: str = "",
    ax=None,
    cmap: str = "coolwarm",
    alpha: float = 0.8,
):
    """Plot triangulated mesh with optional per-face scalar coloring."""
    verts_np = np.asarray(verts)
    faces_np = np.asarray(faces)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    triangles = verts_np[faces_np]

    if scalars is not None:
        scalars_np = np.asarray(scalars)
        # Per-face scalars
        if len(scalars_np) == len(faces_np):
            face_colors = plt.cm.get_cmap(cmap)(
                (scalars_np - scalars_np.min())
                / (scalars_np.ptp() + 1e-12)
            )
        else:
            # Per-vertex: average to faces
            face_vals = scalars_np[faces_np].mean(axis=1)
            face_colors = plt.cm.get_cmap(cmap)(
                (face_vals - face_vals.min()) / (face_vals.ptp() + 1e-12)
            )
        poly = Poly3DCollection(triangles, alpha=alpha)
        poly.set_facecolor(face_colors)
    else:
        poly = Poly3DCollection(triangles, alpha=alpha, edgecolor="k", linewidth=0.1)
        poly.set_facecolor("skyblue")

    ax.add_collection3d(poly)

    # Set axis limits
    all_pts = verts_np
    margin = 0.1
    for i, label in enumerate(["X", "Y", "Z"]):
        lo, hi = all_pts[:, i].min() - margin, all_pts[:, i].max() + margin
        getattr(ax, f"set_{label.lower()}lim")(lo, hi)

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    return ax


def plot_growth_field(
    verts: jnp.ndarray,
    faces: jnp.ndarray,
    growth_rates: jnp.ndarray,
    title: str = "Growth Field",
):
    """Color mesh by per-face growth rates."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_mesh(verts, faces, scalars=growth_rates, title=title, ax=ax, cmap="YlOrRd")
    plt.colorbar(
        plt.cm.ScalarMappable(
            cmap="YlOrRd",
            norm=plt.Normalize(
                float(growth_rates.min()), float(growth_rates.max())
            ),
        ),
        ax=ax,
        shrink=0.6,
        label="Growth Rate",
    )
    return fig


def plot_curvature_map(
    verts: jnp.ndarray,
    faces: jnp.ndarray,
    topo: MeshTopology,
    title: str = "Mean Curvature",
):
    """Color mesh by mean curvature."""
    H = compute_mean_curvature(verts, topo)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_mesh(verts, faces, scalars=H, title=title, ax=ax, cmap="coolwarm")
    plt.colorbar(
        plt.cm.ScalarMappable(
            cmap="coolwarm",
            norm=plt.Normalize(float(H.min()), float(H.max())),
        ),
        ax=ax,
        shrink=0.6,
        label="Mean Curvature",
    )
    return fig


def plot_simulation_frames(
    trajectory: jnp.ndarray,
    faces: jnp.ndarray,
    steps: list[int] | None = None,
    title: str = "Simulation Frames",
):
    """Side-by-side snapshots of simulation at selected timesteps."""
    n_total = trajectory.shape[0]
    if steps is None:
        steps = [0, n_total // 4, n_total // 2, 3 * n_total // 4, n_total - 1]
    steps = [s for s in steps if s < n_total]
    n = len(steps)

    fig = plt.figure(figsize=(4 * n, 4))
    for i, s in enumerate(steps):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_mesh(trajectory[s], faces, title=f"Step {s}", ax=ax)

    fig.suptitle(title)
    plt.tight_layout()
    return fig
