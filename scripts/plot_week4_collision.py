"""Plot Week 4 collision ablation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from cortical_folding.figure_style import PALETTE, STYLE_VERSION, apply_standard_style, style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="results/week4_collision_ablation.csv",
        help="Week 4 ablation CSV input.",
    )
    parser.add_argument(
        "--input-summary",
        default="results/week4_collision_ablation_summary.json",
        help="Week 4 ablation summary JSON input.",
    )
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
    parser.add_argument(
        "--output-metadata",
        default="docs/assets/week4_collision_ablation.meta.json",
        help="Output sidecar metadata JSON path.",
    )
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    apply_standard_style()
    with Path(args.input_json).open() as f:
        data = json.load(f)
    rows = load_rows(args.input_csv)
    with Path(args.input_summary).open() as f:
        summary = json.load(f)
    by_label = {r["label"]: r for r in rows}
    source_labels = [
        data["baseline_label"],
        data["sampled_label"],
        data["spatial_hash_label"],
    ]
    source_rows = [by_label[l] for l in source_labels if l in by_label]

    labels = ["baseline", "sampled", "spatial-hash"]
    overlap = [
        data["baseline_collision_overlap_count"],
        data["sampled_collision_overlap_count"],
        data["spatial_hash_collision_overlap_count"],
    ]
    runtime = [
        1.0,
        1.0,
        data["runtime_ratio_spatial_over_sampled"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    bar_colors = [PALETTE["baseline"], PALETTE["variant_a"], PALETTE["variant_b"]]
    axes[0].bar(labels, overlap, color=bar_colors)
    style_axis(
        axes[0],
        title="Collision overlap count",
        ylabel="Pair count",
        xlabel="Collision mode",
    )

    axes[1].bar(labels, runtime, color=bar_colors)
    axes[1].axhline(
        data["runtime_overhead_budget"],
        color=PALETTE["threshold"],
        linestyle="--",
        linewidth=1.5,
    )
    style_axis(
        axes[1],
        title="Runtime ratio (vs sampled)",
        ylabel="Ratio",
        xlabel="Collision mode",
    )

    fig.suptitle("Week 4 Collision/Contact Ablation")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

    sidecar = Path(args.output_metadata)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_id": "week4_collision_ablation",
        "style_version": STYLE_VERSION,
        "output_png": str(out),
        "source_run_ids": [int(r["run_id"]) for r in source_rows],
        "source_run_labels": [r["label"] for r in source_rows],
        "source_run_config_hashes": [r["run_config_hash"] for r in source_rows],
        "source_sweep_config_hash": summary.get("sweep_config_hash"),
        "source_git_commit": summary.get("git_commit"),
        "source_artifacts": {
            "csv": args.input_csv,
            "summary": args.input_summary,
            "comparison": args.input_json,
        },
    }
    with sidecar.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {out}")
    print(f"Saved metadata: {sidecar}")


if __name__ == "__main__":
    main()
