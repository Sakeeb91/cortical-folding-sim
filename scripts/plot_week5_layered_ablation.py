"""Plot Week 5 layered approximation ablation diagnostics."""

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
        default="results/week5_layered_ablation.csv",
        help="Week 5 ablation CSV input.",
    )
    parser.add_argument(
        "--input-json",
        default="results/week5_layered_comparison.json",
        help="Week 5 comparison JSON input.",
    )
    parser.add_argument(
        "--input-summary",
        default="results/week5_layered_ablation_summary.json",
        help="Week 5 summary JSON input.",
    )
    parser.add_argument(
        "--output",
        default="docs/assets/week5_layered_ablation.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--output-metadata",
        default="docs/assets/week5_layered_ablation.meta.json",
        help="Output sidecar metadata JSON path.",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    apply_standard_style()
    rows = load_rows(args.input_csv)
    with Path(args.input_json).open() as f:
        comparison = json.load(f)
    with Path(args.input_summary).open() as f:
        summary = json.load(f)

    labels = [r["label"] for r in rows]
    gi = [float(r["gi"]) for r in rows]
    disp = [float(r["disp_p95"]) for r in rows]
    runtime = [float(r["runtime_s"]) for r in rows]
    robust_labels = set(comparison.get("robust_region_labels", []))

    colors = [PALETTE["variant_b"] if label in robust_labels else PALETTE["baseline"] for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    axes[0].bar(labels, gi, color=colors)
    style_axis(axes[0], title="GI by ablation run", ylabel="GI", xlabel="Run label")
    axes[0].tick_params(axis="x", labelrotation=70)

    axes[1].bar(labels, disp, color=colors)
    axes[1].axhline(
        comparison["robust_region_definition"]["max_disp_p95"],
        color=PALETTE["threshold"],
        linestyle="--",
        linewidth=1.5,
    )
    style_axis(axes[1], title="Displacement p95", ylabel="Distance", xlabel="Run label")
    axes[1].tick_params(axis="x", labelrotation=70)

    axes[2].bar(labels, runtime, color=colors)
    style_axis(axes[2], title="Runtime per run", ylabel="Seconds", xlabel="Run label")
    axes[2].tick_params(axis="x", labelrotation=70)

    fig.suptitle(
        "Week 5 Layered Approximation Ablation (green = robust region candidates)",
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

    sidecar = Path(args.output_metadata)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_id": "week5_layered_ablation",
        "style_version": STYLE_VERSION,
        "output_png": str(out),
        "source_run_ids": [int(r["run_id"]) for r in rows],
        "source_run_labels": [r["label"] for r in rows],
        "source_run_config_hashes": [r["run_config_hash"] for r in rows],
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
