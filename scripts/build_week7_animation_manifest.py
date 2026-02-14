"""Build Week 7 animation manifest with source-run mappings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-summary",
        default="results/week7_animation_comparison_summary.json",
        help="Week 7 animation summary JSON path.",
    )
    parser.add_argument(
        "--input-metadata",
        default="docs/assets/week7_baseline_vs_improved.meta.json",
        help="Week 7 animation metadata sidecar JSON path.",
    )
    parser.add_argument(
        "--input-results-index",
        default="results/week7_results_index_summary.json",
        help="Week 7 results index summary JSON path.",
    )
    parser.add_argument(
        "--output-json",
        default="docs/assets/week7_animation_manifest.json",
        help="Output manifest path.",
    )
    parser.add_argument(
        "--fail-on-missing-run-ids",
        action="store_true",
        help="Exit non-zero when source run IDs are missing.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    summary = load_json(args.input_summary)
    metadata = load_json(args.input_metadata)
    results_index = load_json(args.input_results_index)

    output_files = [Path(p) for p in metadata.get("output_files", [])]
    source_run_ids = metadata.get("source_run_ids", [])
    clean_source_run_ids = [rid for rid in source_run_ids if rid is not None]

    entry = {
        "asset_id": "week7_baseline_vs_improved_animation",
        "output_files": [str(p) for p in output_files],
        "output_exists": [p.exists() for p in output_files],
        "all_outputs_exist": all(p.exists() for p in output_files),
        "output_metadata_sidecar": args.input_metadata,
        "metadata_sidecar_exists": Path(args.input_metadata).exists(),
        "source_run_ids": clean_source_run_ids,
        "source_run_labels": metadata.get("source_run_labels", []),
        "source_run_config_hashes": metadata.get("source_run_config_hashes", []),
        "source_sweep_config_hash": metadata.get("source_sweep_config_hash"),
        "source_git_commit": metadata.get("source_git_commit"),
        "source_artifacts": {
            "animation_summary": args.input_summary,
            "results_index": args.input_results_index,
            "source_csv": summary.get("source_csv"),
            "source_summary": summary.get("source_summary"),
            "config": summary.get("config_path"),
        },
        "has_source_run_ids": len(clean_source_run_ids) > 0,
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_assets": 1,
        "n_assets_with_source_run_ids": 1 if entry["has_source_run_ids"] else 0,
        "assets_missing_source_run_ids": [] if entry["has_source_run_ids"] else [entry["asset_id"]],
        "all_assets_have_source_run_ids": entry["has_source_run_ids"],
        "results_index_all_claims_have_linked_artifacts": bool(
            results_index.get("all_claims_have_linked_artifacts")
        ),
        "assets": [entry],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {output_json}")

    if args.fail_on_missing_run_ids and not entry["has_source_run_ids"]:
        raise SystemExit("Missing source run IDs for week7 animation asset")


if __name__ == "__main__":
    main()
