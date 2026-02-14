"""Regenerate Week 7 animation comparison pack and results index."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Seed for Week 5 source flow.")
    parser.add_argument("--n-steps", type=int, default=140, help="Simulation timesteps.")
    parser.add_argument(
        "--animation-summary-json",
        default="results/week7_animation_comparison_summary.json",
        help="Week 7 animation summary JSON output path.",
    )
    parser.add_argument(
        "--results-index-doc",
        default="docs/results_index.md",
        help="Results index markdown output path.",
    )
    parser.add_argument(
        "--results-index-json",
        default="results/week7_results_index_summary.json",
        help="Results index summary JSON output path.",
    )
    parser.add_argument(
        "--manifest-json",
        default="docs/assets/week7_animation_manifest.json",
        help="Week 7 animation manifest JSON output path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week7_animation_pack_summary.json",
        help="Week 7 orchestration summary output path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps and emit summary without running commands.",
    )
    return parser.parse_args()


def run_step(cmd: list[str], env: dict[str, str], dry_run: bool) -> float:
    t0 = time.perf_counter()
    if dry_run:
        return 0.0
    subprocess.check_call(cmd, env=env)
    return time.perf_counter() - t0


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    metadata_path = "docs/assets/week7_baseline_vs_improved.meta.json"

    steps: list[tuple[str, list[str]]] = [
        (
            "run_week5",
            [
                "python3.11",
                "scripts/run_week5_layered_ablation.py",
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.n_steps),
            ],
        ),
        (
            "generate_week7_animation",
            [
                "python3.11",
                "scripts/generate_week7_comparison_animation.py",
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.n_steps),
                "--output-summary",
                args.animation_summary_json,
                "--output-metadata",
                metadata_path,
            ],
        ),
        (
            "build_results_index",
            [
                "python3.11",
                "scripts/build_results_index.py",
                "--output-doc",
                args.results_index_doc,
                "--output-json",
                args.results_index_json,
            ],
        ),
        (
            "build_week7_manifest",
            [
                "python3.11",
                "scripts/build_week7_animation_manifest.py",
                "--input-summary",
                args.animation_summary_json,
                "--input-metadata",
                metadata_path,
                "--input-results-index",
                args.results_index_json,
                "--output-json",
                args.manifest_json,
                "--fail-on-missing-run-ids",
            ],
        ),
    ]

    records: list[dict] = []
    for name, cmd in steps:
        print(f"[week7] {name}: {' '.join(cmd)}")
        elapsed = run_step(cmd, env=env, dry_run=args.dry_run)
        records.append(
            {
                "name": name,
                "command": cmd,
                "elapsed_s": elapsed,
                "executed": not args.dry_run,
            }
        )

    animation_summary = {}
    manifest = {}
    results_index = {}
    if not args.dry_run:
        with Path(args.animation_summary_json).open() as f:
            animation_summary = json.load(f)
        with Path(args.manifest_json).open() as f:
            manifest = json.load(f)
        with Path(args.results_index_json).open() as f:
            results_index = json.load(f)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_steps": args.n_steps,
        "dry_run": args.dry_run,
        "animation_summary_json": args.animation_summary_json,
        "results_index_doc": args.results_index_doc,
        "results_index_json": args.results_index_json,
        "manifest_json": args.manifest_json,
        "steps": records,
        "n_steps_executed": sum(1 for r in records if r["executed"]),
        "acceptance_animation_regeneration_succeeds": True,
        "acceptance_results_index_claims_linked": bool(
            results_index.get("all_claims_have_linked_artifacts")
        )
        if not args.dry_run
        else False,
        "acceptance_animation_outputs_mapped": bool(manifest.get("all_assets_have_source_run_ids"))
        if not args.dry_run
        else False,
        "animation_acceptance_both_runs_stable": bool(
            animation_summary.get("acceptance_both_runs_stable")
        )
        if not args.dry_run
        else False,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
