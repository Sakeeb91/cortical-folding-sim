"""Plot Week 5 layered approximation ablation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="results/week5_layered_ablation.csv",
        help="Week 5 ablation CSV input.",
    )
    parser.add_argument(
        "--input-json",
        default="results/week5_layered_comparison.json",
        help="Week 5 comparison JSON input.",
    )
    parser.add_argument(
        "--output",
        default="docs/assets/week5_layered_ablation.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    with Path(args.input_json).open() as f:
        comparison = json.load(f)

    labels = [r["label"] for r in rows]
    gi = [float(r["gi"]) for r in rows]
    disp = [float(r["disp_p95"]) for r in rows]
    runtime = [float(r["runtime_s"]) for r in rows]
    robust_labels = set(comparison.get("robust_region_labels", []))

    colors = ["#2ca25f" if label in robust_labels else "#7f8c8d" for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    axes[0].bar(labels, gi, color=colors)
    axes[0].set_title("GI by ablation run")
    axes[0].set_ylabel("GI")
    axes[0].tick_params(axis="x", labelrotation=70)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, disp, color=colors)
    axes[1].axhline(
        comparison["robust_region_definition"]["max_disp_p95"],
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
    )
    axes[1].set_title("Displacement p95")
    axes[1].set_ylabel("Distance")
    axes[1].tick_params(axis="x", labelrotation=70)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(labels, runtime, color=colors)
    axes[2].set_title("Runtime per run")
    axes[2].set_ylabel("Seconds")
    axes[2].tick_params(axis="x", labelrotation=70)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Week 5 Layered Approximation Ablation (green = robust region candidates)",
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
