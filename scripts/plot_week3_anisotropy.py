"""Plot Week 3 anisotropy comparison metrics."""

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
        default="results/week3_anisotropy_ab.csv",
        help="A/B CSV with source run rows.",
    )
    parser.add_argument(
        "--input-summary",
        default="results/week3_anisotropy_ab_summary.json",
        help="Sweep summary JSON path.",
    )
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
    parser.add_argument(
        "--output-metadata",
        default="docs/assets/week3_anisotropy_delta.meta.json",
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
    source_labels = ["isotropic_baseline", "anisotropic_z_bias"]
    source_rows = [by_label[l] for l in source_labels if l in by_label]

    labels = ["GI", "Curvature p90", "Disp p95"]
    values = [
        data["delta_gi"],
        data["delta_curv_p90"],
        data["delta_disp_p95"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        labels,
        values,
        color=[PALETTE["variant_a"], PALETTE["variant_b"], PALETTE["accent"]],
    )
    style_axis(
        ax,
        title="Week 3: Anisotropic vs Isotropic Delta",
        ylabel="Anisotropic - Isotropic",
        zero_line=True,
    )
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    sidecar = Path(args.output_metadata)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_id": "week3_anisotropy_delta",
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
