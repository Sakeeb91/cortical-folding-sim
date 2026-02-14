"""Build Week 6 figure-source manifest with run ID mappings."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cortical_folding.figure_style import STYLE_VERSION


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    asset_name: str
    csv_name: str
    summary_name: str
    manifest_name: str
    comparison_name: str
    source_labels: tuple[str, ...]


FIGURE_SPECS = (
    FigureSpec(
        figure_id="week3_anisotropy_delta",
        asset_name="week3_anisotropy_delta.png",
        csv_name="week3_anisotropy_ab.csv",
        summary_name="week3_anisotropy_ab_summary.json",
        manifest_name="week3_anisotropy_ab_manifest.json",
        comparison_name="week3_anisotropy_comparison.json",
        source_labels=("isotropic_baseline", "anisotropic_z_bias"),
    ),
    FigureSpec(
        figure_id="week4_collision_ablation",
        asset_name="week4_collision_ablation.png",
        csv_name="week4_collision_ablation.csv",
        summary_name="week4_collision_ablation_summary.json",
        manifest_name="week4_collision_ablation_manifest.json",
        comparison_name="week4_collision_comparison.json",
        source_labels=(
            "collision_disabled_baseline",
            "collision_sampled_deterministic",
            "collision_spatial_hash",
        ),
    ),
    FigureSpec(
        figure_id="week5_layered_ablation",
        asset_name="week5_layered_ablation.png",
        csv_name="week5_layered_ablation.csv",
        summary_name="week5_layered_ablation_summary.json",
        manifest_name="week5_layered_ablation_manifest.json",
        comparison_name="week5_layered_comparison.json",
        source_labels=("layered_off_reference", "layered_on_reference"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing result artifacts.",
    )
    parser.add_argument(
        "--assets-dir",
        default="docs/assets",
        help="Directory containing figure assets.",
    )
    parser.add_argument(
        "--output-json",
        default="docs/assets/week6_figure_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--style-version",
        default=STYLE_VERSION,
        help="Figure style version tag.",
    )
    parser.add_argument(
        "--fail-on-missing-run-ids",
        action="store_true",
        help="Exit non-zero when any figure has no mapped run IDs.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_entry(spec: FigureSpec, results_dir: Path, assets_dir: Path) -> dict:
    csv_path = results_dir / spec.csv_name
    summary_path = results_dir / spec.summary_name
    manifest_path = results_dir / spec.manifest_name
    comparison_path = results_dir / spec.comparison_name
    output_path = assets_dir / spec.asset_name
    sidecar_path = assets_dir / f"{Path(spec.asset_name).stem}.meta.json"

    rows = load_csv(csv_path)
    summary = load_json(summary_path)
    _ = load_json(manifest_path)
    _ = load_json(comparison_path)

    by_label = {r["label"]: r for r in rows}
    selected = [by_label[label] for label in spec.source_labels if label in by_label]
    if not selected:
        selected = rows

    entry = {
        "figure_id": spec.figure_id,
        "output_file": str(output_path),
        "output_exists": output_path.exists(),
        "output_metadata_sidecar": str(sidecar_path),
        "metadata_sidecar_exists": sidecar_path.exists(),
        "source_run_ids": [int(float(r["run_id"])) for r in selected if "run_id" in r],
        "source_run_labels": [r["label"] for r in selected],
        "source_run_config_hashes": [r["run_config_hash"] for r in selected if "run_config_hash" in r],
        "source_sweep_config_hash": summary.get("sweep_config_hash"),
        "source_git_commit": summary.get("git_commit"),
        "source_artifacts": {
            "csv": str(csv_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "comparison": str(comparison_path),
        },
    }
    entry["has_source_run_ids"] = len(entry["source_run_ids"]) > 0
    return entry


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    assets_dir = Path(args.assets_dir)
    entries = [build_entry(spec, results_dir, assets_dir) for spec in FIGURE_SPECS]
    missing = [e["figure_id"] for e in entries if not e["has_source_run_ids"]]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "style_version": args.style_version,
        "n_figures": len(entries),
        "n_figures_with_source_run_ids": sum(1 for e in entries if e["has_source_run_ids"]),
        "figures_missing_source_run_ids": missing,
        "all_figures_have_source_run_ids": len(missing) == 0,
        "figures": entries,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out}")

    if args.fail_on_missing_run_ids and missing:
        raise SystemExit(f"Missing source run IDs for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
