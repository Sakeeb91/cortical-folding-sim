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
        default=1.35,
        help="Maximum allowed runtime ratio (spatial-hash / sampled).",
    )
    return parser.parse_args()


def load_csv_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


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
    payload = {
        "n_runs": len(rows),
        "config_path": args.config_path,
        "runtime_overhead_budget": args.runtime_overhead_budget,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
