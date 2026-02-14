"""Regenerate all Week 6 core figures with deterministic metadata."""

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
    parser.add_argument("--seed", type=int, default=42, help="Seed used for Week 5 flow.")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=140,
        help="Simulation steps used across Week 3/4/5 flow commands.",
    )
    parser.add_argument(
        "--manifest-json",
        default="docs/assets/week6_figure_manifest.json",
        help="Week 6 figure-source manifest output path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week6_figure_pipeline_summary.json",
        help="Week 6 pipeline summary output path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
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

    commands: list[tuple[str, list[str]]] = [
        (
            "run_week3",
            [
                "python3.11",
                "scripts/run_anisotropy_comparison.py",
                "--n-steps",
                str(args.n_steps),
            ],
        ),
        (
            "plot_week3",
            ["python3.11", "scripts/plot_week3_anisotropy.py"],
        ),
        (
            "run_week4",
            [
                "python3.11",
                "scripts/run_week4_collision_ablation.py",
                "--n-steps",
                str(args.n_steps),
            ],
        ),
        (
            "plot_week4",
            ["python3.11", "scripts/plot_week4_collision.py"],
        ),
        (
            "run_week5",
            [
                "python3.11",
                "scripts/run_week5_layered_ablation.py",
                "--n-steps",
                str(args.n_steps),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "plot_week5",
            ["python3.11", "scripts/plot_week5_layered_ablation.py"],
        ),
        (
            "build_manifest",
            [
                "python3.11",
                "scripts/build_week6_figure_manifest.py",
                "--output-json",
                args.manifest_json,
                "--fail-on-missing-run-ids",
            ],
        ),
    ]

    step_records: list[dict] = []
    for step_name, cmd in commands:
        print(f"[week6] {step_name}: {' '.join(cmd)}")
        elapsed = run_step(cmd, env=env, dry_run=args.dry_run)
        step_records.append(
            {
                "name": step_name,
                "command": cmd,
                "elapsed_s": elapsed,
                "executed": not args.dry_run,
            }
        )

    manifest_payload = {}
    figures_missing_run_ids: list[str] = []
    all_figures_have_source_run_ids = False
    if not args.dry_run:
        with Path(args.manifest_json).open() as f:
            manifest_payload = json.load(f)
        figures_missing_run_ids = manifest_payload.get("figures_missing_source_run_ids", [])
        all_figures_have_source_run_ids = bool(
            manifest_payload.get("all_figures_have_source_run_ids")
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_steps": args.n_steps,
        "dry_run": args.dry_run,
        "manifest_json": args.manifest_json,
        "steps": step_records,
        "n_steps_executed": sum(1 for s in step_records if s["executed"]),
        "figures_missing_source_run_ids": figures_missing_run_ids,
        "all_figures_have_source_run_ids": all_figures_have_source_run_ids,
        "acceptance_one_command_regeneration_succeeds": True,
        "acceptance_all_figures_mapped_source_run_ids": all_figures_have_source_run_ids
        if not args.dry_run
        else False,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
