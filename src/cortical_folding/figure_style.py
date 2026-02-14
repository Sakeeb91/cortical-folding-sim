"""Shared plotting style helpers for reproducible paper figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

STYLE_VERSION = "week6_standard_v1"

PALETTE = {
    "baseline": "#6B7280",
    "variant_a": "#1D4ED8",
    "variant_b": "#059669",
    "accent": "#B45309",
    "threshold": "#B91C1C",
}


def apply_standard_style() -> None:
    """Apply a deterministic house style for all week figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": False,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def style_axis(ax, *, title: str, ylabel: str, xlabel: str = "", zero_line: bool = False) -> None:
    """Apply consistent axis formatting."""
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25)
    if zero_line:
        ax.axhline(0.0, color="black", linewidth=1.0)
