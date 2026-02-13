"""Run Week 4 collision/contact ablation and summarize acceptance metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default="configs/week4_collision_ablation.json",
        help="Collision ablation config JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/week4_collision_ablation.csv",
        help="Sweep CSV output path.",
    )
    parser.add_argument(
        "--output-summary",
        default="results/week4_collision_ablation_summary.json",
        help="Sweep summary output path.",
    )
    parser.add_argument(
        "--output-manifest",
        default="results/week4_collision_ablation_manifest.json",
        help="Sweep manifest output path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week4_collision_comparison.json",
        help="Week 4 comparison summary output path.",
    )
    parser.add_argument("--n-steps", type=int, default=140, help="Simulation steps.")
    parser.add_argument(
        "--runtime-overhead-budget",
        type=float,
        default=1.6,
        help="Maximum allowed runtime ratio (spatial-hash / sampled).",
    )
    return parser.parse_args()


def load_csv_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def to_int(row: dict, key: str) -> int:
    return int(float(row[key]))


def main() -> None:
    args = parse_args()
    cmd = [
        "python3.11",
        "scripts/run_forward_sweep.py",
        "--config-path",
        args.config_path,
        "--n-steps",
        str(args.n_steps),
        "--output-csv",
        args.output_csv,
        "--output-summary",
        args.output_summary,
        "--output-manifest",
        args.output_manifest,
    ]
    subprocess.check_call(cmd)

    rows = load_csv_rows(args.output_csv)
    row_by_label = {r["label"]: r for r in rows}
    baseline = row_by_label["collision_disabled_baseline"]
    sampled = row_by_label["collision_sampled_deterministic"]
    spatial = row_by_label["collision_spatial_hash"]

    baseline_overlap_p95 = to_float(baseline, "collision_overlap_p95")
    sampled_overlap_p95 = to_float(sampled, "collision_overlap_p95")
    spatial_overlap_p95 = to_float(spatial, "collision_overlap_p95")
    baseline_overlap_count = to_float(baseline, "collision_overlap_count")
    sampled_overlap_count = to_float(sampled, "collision_overlap_count")
    spatial_overlap_count = to_float(spatial, "collision_overlap_count")
    sampled_runtime = to_float(sampled, "runtime_s")
    spatial_runtime = to_float(spatial, "runtime_s")
    runtime_ratio = spatial_runtime / max(sampled_runtime, 1e-12)

    payload = {
        "n_runs": len(rows),
        "config_path": args.config_path,
        "baseline_label": "collision_disabled_baseline",
        "sampled_label": "collision_sampled_deterministic",
        "spatial_hash_label": "collision_spatial_hash",
        "baseline_stable": to_int(baseline, "stable"),
        "sampled_stable": to_int(sampled, "stable"),
        "spatial_hash_stable": to_int(spatial, "stable"),
        "baseline_collision_overlap_p95": baseline_overlap_p95,
        "sampled_collision_overlap_p95": sampled_overlap_p95,
        "spatial_hash_collision_overlap_p95": spatial_overlap_p95,
        "baseline_collision_overlap_count": baseline_overlap_count,
        "sampled_collision_overlap_count": sampled_overlap_count,
        "spatial_hash_collision_overlap_count": spatial_overlap_count,
        "reduction_overlap_p95_vs_baseline": baseline_overlap_p95 - spatial_overlap_p95,
        "reduction_overlap_p95_vs_sampled": sampled_overlap_p95 - spatial_overlap_p95,
        "reduction_overlap_count_vs_baseline": baseline_overlap_count - spatial_overlap_count,
        "reduction_overlap_count_vs_sampled": sampled_overlap_count - spatial_overlap_count,
        "sampled_collision_force_share": to_float(sampled, "collision_force_share"),
        "spatial_hash_collision_force_share": to_float(spatial, "collision_force_share"),
        "sampled_runtime_s": sampled_runtime,
        "spatial_hash_runtime_s": spatial_runtime,
        "runtime_ratio_spatial_over_sampled": runtime_ratio,
        "runtime_overhead_budget": args.runtime_overhead_budget,
        "acceptance_collision_outliers_reduced": bool(
            spatial_overlap_count <= baseline_overlap_count
            and spatial_overlap_count <= sampled_overlap_count
            and spatial_overlap_count < baseline_overlap_count
        ),
        "acceptance_runtime_within_budget": bool(runtime_ratio <= args.runtime_overhead_budget),
    }
    payload["acceptance_week4_passed"] = bool(
        payload["acceptance_collision_outliers_reduced"]
        and payload["acceptance_runtime_within_budget"]
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
