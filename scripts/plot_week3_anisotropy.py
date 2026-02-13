"""Plot Week 3 anisotropy comparison metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default="results/week3_anisotropy_comparison.json",
        help="Comparison JSON path.",
    )
    parser.add_argument(
        "--output-png",
        default="docs/assets/week3_anisotropy_delta.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.input_json).open() as f:
        data = json.load(f)

    labels = ["GI", "Curvature p90", "Disp p95"]
    values = [
        data["delta_gi"],
        data["delta_curv_p90"],
        data["delta_disp_p95"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#2f6db3", "#4fae5c", "#d18c2f"])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Week 3: Anisotropic vs Isotropic Delta")
    ax.set_ylabel("Anisotropic - Isotropic")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
