"""Plot Week 4 collision ablation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default="results/week4_collision_comparison.json",
        help="Week 4 comparison JSON input.",
    )
    parser.add_argument(
        "--output",
        default="docs/assets/week4_collision_ablation.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.input_json).open() as f:
        data = json.load(f)

    labels = ["baseline", "sampled", "spatial-hash"]
    overlap = [
        data["baseline_collision_overlap_p95"],
        data["sampled_collision_overlap_p95"],
        data["spatial_hash_collision_overlap_p95"],
    ]
    runtime = [
        1.0,
        1.0,
        data["runtime_ratio_spatial_over_sampled"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].bar(labels, overlap, color=["#7f8c8d", "#2980b9", "#27ae60"])
    axes[0].set_title("Collision overlap p95")
    axes[0].set_ylabel("Overlap depth")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, runtime, color=["#7f8c8d", "#2980b9", "#27ae60"])
    axes[1].axhline(data["runtime_overhead_budget"], color="#c0392b", linestyle="--", linewidth=1.5)
    axes[1].set_title("Runtime ratio (vs sampled)")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Week 4 Collision/Contact Ablation")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
