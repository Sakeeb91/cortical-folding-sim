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


def save_comparison_animation(
    baseline_trajectory: jnp.ndarray,
    improved_trajectory: jnp.ndarray,
    faces: jnp.ndarray,
    output_paths: list[str] | tuple[str, ...],
    fps: int = 20,
    stride: int = 1,
    rotate: bool = False,
    dpi: int = 120,
    baseline_title: str = "Baseline",
    improved_title: str = "Improved",
    title: str = "Baseline vs Improved",
):
    """Save a side-by-side animation for baseline and improved trajectories.

    The same frame pipeline is used for all outputs in `output_paths`.
    Supported suffixes are `.gif` and `.mp4`.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if not output_paths:
        raise ValueError("output_paths must contain at least one output file")
    output_specs: list[tuple[str, str]] = []
    for output_path in output_paths:
        suffix = Path(output_path).suffix.lower()
        if suffix not in {".gif", ".mp4"}:
            raise ValueError("comparison outputs must end with .gif or .mp4")
        output_specs.append((output_path, suffix))

    baseline_np = np.asarray(baseline_trajectory)[::stride]
    improved_np = np.asarray(improved_trajectory)[::stride]
    faces_np = np.asarray(faces)
    n_frames = min(baseline_np.shape[0], improved_np.shape[0])
    if n_frames == 0:
        raise ValueError("comparison trajectories are empty after applying stride")

    baseline_np = baseline_np[:n_frames]
    improved_np = improved_np[:n_frames]

    all_pts = np.concatenate(
        [baseline_np.reshape(-1, 3), improved_np.reshape(-1, 3)],
        axis=0,
    )
    margin = 0.1
    mins = all_pts.min(axis=0) - margin
    maxs = all_pts.max(axis=0) + margin

    fig = plt.figure(figsize=(12, 6))
    ax_base = fig.add_subplot(121, projection="3d")
    ax_improved = fig.add_subplot(122, projection="3d")
    for ax, panel_title in (
        (ax_base, baseline_title),
        (ax_improved, improved_title),
    ):
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_title(panel_title)
    fig.suptitle(title)

    base_poly = Poly3DCollection(
        baseline_np[0][faces_np],
        alpha=0.85,
        edgecolor="k",
        linewidth=0.06,
    )
    base_poly.set_facecolor("skyblue")
    ax_base.add_collection3d(base_poly)

    improved_poly = Poly3DCollection(
        improved_np[0][faces_np],
        alpha=0.85,
        edgecolor="k",
        linewidth=0.06,
    )
    improved_poly.set_facecolor("salmon")
    ax_improved.add_collection3d(improved_poly)

    frame_text = fig.text(0.02, 0.96, "Step 0")

    def update(frame_idx: int):
        base_poly.set_verts(baseline_np[frame_idx][faces_np])
        improved_poly.set_verts(improved_np[frame_idx][faces_np])
        if rotate:
            azim = 45 + 0.9 * frame_idx
            ax_base.view_init(elev=20, azim=azim)
            ax_improved.view_init(elev=20, azim=azim)
        frame_text.set_text(f"Step {frame_idx * stride}")
        return base_poly, improved_poly, frame_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
    )

    saved_paths: list[str] = []
    for output_path, suffix in output_specs:
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(output_path, writer=writer, dpi=dpi)
        saved_paths.append(output_path)

    plt.close(fig)
    return saved_paths


def _face_shaded_colors(
    tris: np.ndarray,
    base_rgb: tuple[float, float, float],
    light_dir: np.ndarray,
    ambient: float,
    diffuse: float,
) -> np.ndarray:
    """Compute per-face RGB colors from Lambertian shading."""
    v1 = tris[:, 1, :] - tris[:, 0, :]
    v2 = tris[:, 2, :] - tris[:, 0, :]
    normals = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norm, 1e-12)
    light = light_dir / max(float(np.linalg.norm(light_dir)), 1e-12)
    ndotl = np.clip(np.sum(normals * light[None, :], axis=1), 0.0, 1.0)
    intensity = np.clip(ambient + diffuse * ndotl, 0.0, 1.0)
    base = np.asarray(base_rgb, dtype=np.float32)[None, :]
    return np.clip(base * intensity[:, None], 0.0, 1.0)


def save_publication_comparison_animation(
    baseline_trajectory: jnp.ndarray,
    improved_trajectory: jnp.ndarray,
    faces: jnp.ndarray,
    output_paths: list[str] | tuple[str, ...],
    fps: int = 24,
    stride: int = 1,
    dpi: int = 120,
    width_px: int = 1920,
    height_px: int = 1080,
    supersample_scale: int = 2,
    rotate: bool = True,
    camera_elev: float = 20.0,
    azim_start: float = 45.0,
    azim_span: float = 110.0,
    baseline_title: str = "Baseline",
    improved_title: str = "High Fidelity",
    title: str = "Publication Comparison",
    baseline_rgb: tuple[float, float, float] = (0.34, 0.62, 0.90),
    improved_rgb: tuple[float, float, float] = (0.94, 0.52, 0.38),
    light_dir: tuple[float, float, float] = (0.35, 0.30, 0.88),
    ambient: float = 0.34,
    diffuse: float = 0.66,
    metric_overlays: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[str]:
    """Save publication-style side-by-side animation with shading and overlays."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if width_px < 1 or height_px < 1:
        raise ValueError("width_px and height_px must be >= 1")
    if supersample_scale < 1:
        raise ValueError("supersample_scale must be >= 1")
    if not output_paths:
        raise ValueError("output_paths must contain at least one output file")

    output_specs: list[tuple[str, str]] = []
    for output_path in output_paths:
        suffix = Path(output_path).suffix.lower()
        if suffix not in {".gif", ".mp4"}:
            raise ValueError("comparison outputs must end with .gif or .mp4")
        output_specs.append((output_path, suffix))

    baseline_np = np.asarray(baseline_trajectory)[::stride]
    improved_np = np.asarray(improved_trajectory)[::stride]
    faces_np = np.asarray(faces)
    n_frames = min(baseline_np.shape[0], improved_np.shape[0])
    if n_frames == 0:
        raise ValueError("comparison trajectories are empty after applying stride")
    baseline_np = baseline_np[:n_frames]
    improved_np = improved_np[:n_frames]

    if metric_overlays:
        for metric_name, (base_series, improved_series) in metric_overlays.items():
            if len(base_series) < n_frames or len(improved_series) < n_frames:
                raise ValueError(
                    f"metric overlay '{metric_name}' must have >= {n_frames} values for both series"
                )

    all_pts = np.concatenate(
        [baseline_np.reshape(-1, 3), improved_np.reshape(-1, 3)],
        axis=0,
    )
    margin = 0.1
    mins = all_pts.min(axis=0) - margin
    maxs = all_pts.max(axis=0) + margin

    render_dpi = dpi * supersample_scale
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi))
    ax_base = fig.add_subplot(121, projection="3d")
    ax_improved = fig.add_subplot(122, projection="3d")
    for ax, panel_title in ((ax_base, baseline_title), (ax_improved, improved_title)):
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_title(panel_title, pad=14.0)
        ax.set_axis_off()

    fig.suptitle(title, y=0.98)
    light_dir_np = np.asarray(light_dir, dtype=np.float32)

    base_tris = baseline_np[0][faces_np]
    base_poly = Poly3DCollection(base_tris, linewidth=0.0, antialiased=True, alpha=1.0)
    base_poly.set_facecolor(
        _face_shaded_colors(base_tris, baseline_rgb, light_dir_np, ambient, diffuse)
    )
    ax_base.add_collection3d(base_poly)

    improved_tris = improved_np[0][faces_np]
    improved_poly = Poly3DCollection(improved_tris, linewidth=0.0, antialiased=True, alpha=1.0)
    improved_poly.set_facecolor(
        _face_shaded_colors(improved_tris, improved_rgb, light_dir_np, ambient, diffuse)
    )
    ax_improved.add_collection3d(improved_poly)

    frame_text = fig.text(0.02, 0.97, "Step 0", fontsize=10)
    base_metric_text = ax_base.text2D(0.03, 0.02, "", transform=ax_base.transAxes, fontsize=9)
    improved_metric_text = ax_improved.text2D(
        0.03, 0.02, "", transform=ax_improved.transAxes, fontsize=9
    )

    metric_names = list(metric_overlays.keys()) if metric_overlays else []

    def metric_text_for_frame(frame_idx: int, panel: str) -> str:
        if not metric_overlays:
            return ""
        values = []
        for name in metric_names:
            base_series, improved_series = metric_overlays[name]
            metric_val = base_series[frame_idx] if panel == "baseline" else improved_series[frame_idx]
            values.append(f"{name}: {float(metric_val):.4f}")
        return "\n".join(values)

    def eased_azim(frame_idx: int) -> float:
        if n_frames <= 1:
            return azim_start
        phase = frame_idx / (n_frames - 1)
        smooth = 0.5 - 0.5 * np.cos(np.pi * phase)
        return azim_start + azim_span * smooth

    def update(frame_idx: int):
        base_tris_frame = baseline_np[frame_idx][faces_np]
        improved_tris_frame = improved_np[frame_idx][faces_np]
        base_poly.set_verts(base_tris_frame)
        improved_poly.set_verts(improved_tris_frame)
        base_poly.set_facecolor(
            _face_shaded_colors(base_tris_frame, baseline_rgb, light_dir_np, ambient, diffuse)
        )
        improved_poly.set_facecolor(
            _face_shaded_colors(improved_tris_frame, improved_rgb, light_dir_np, ambient, diffuse)
        )
        if rotate:
            azim = eased_azim(frame_idx)
            ax_base.view_init(elev=camera_elev, azim=azim)
            ax_improved.view_init(elev=camera_elev, azim=azim)
        frame_text.set_text(f"Step {frame_idx * stride}")
        base_metric_text.set_text(metric_text_for_frame(frame_idx, "baseline"))
        improved_metric_text.set_text(metric_text_for_frame(frame_idx, "improved"))
        return base_poly, improved_poly, frame_text, base_metric_text, improved_metric_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
    )

    saved_paths: list[str] = []
    for output_path, suffix in output_specs:
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(output_path, writer=writer, dpi=render_dpi)
        saved_paths.append(output_path)

    plt.close(fig)
    return saved_paths
