"""Visualization utilities for cortical folding simulation."""

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
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
                / (np.ptp(scalars_np) + 1e-12)
            )
        else:
            # Per-vertex: average to faces
            face_vals = scalars_np[faces_np].mean(axis=1)
            face_colors = plt.cm.get_cmap(cmap)(
                (face_vals - face_vals.min()) / (np.ptp(face_vals) + 1e-12)
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


def save_simulation_animation(
    trajectory: jnp.ndarray,
    faces: jnp.ndarray,
    output_path: str = "simulation.gif",
    fps: int = 20,
    stride: int = 1,
    rotate: bool = False,
    dpi: int = 120,
):
    """Save a trajectory animation as GIF or MP4.

    The output format is inferred from file extension (`.gif` or `.mp4`).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if fps < 1:
        raise ValueError("fps must be >= 1")

    traj_np = np.asarray(trajectory)[::stride]
    faces_np = np.asarray(faces)
    if traj_np.shape[0] == 0:
        raise ValueError("trajectory is empty after applying stride")

    # Fix axis bounds across all frames to avoid camera jitter.
    all_pts = traj_np.reshape(-1, 3)
    margin = 0.1
    mins = all_pts.min(axis=0) - margin
    maxs = all_pts.max(axis=0) + margin

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_title("Cortical Folding Simulation")

    poly = Poly3DCollection(
        traj_np[0][faces_np],
        alpha=0.85,
        edgecolor="k",
        linewidth=0.06,
    )
    poly.set_facecolor("skyblue")
    ax.add_collection3d(poly)
    frame_text = ax.text2D(0.03, 0.95, "Step 0", transform=ax.transAxes)

    def update(frame_idx: int):
        verts = traj_np[frame_idx]
        poly.set_verts(verts[faces_np])
        if rotate:
            ax.view_init(elev=20, azim=45 + 0.9 * frame_idx)
        frame_text.set_text(f"Step {frame_idx * stride}")
        return poly, frame_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=traj_np.shape[0],
        interval=1000 / fps,
        blit=False,
    )

    suffix = Path(output_path).suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    else:
        raise ValueError("output_path must end with .gif or .mp4")

    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path
