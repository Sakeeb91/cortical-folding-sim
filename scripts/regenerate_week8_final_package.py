"""Regenerate the Week 8 final packaging outputs from one command path."""

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
    parser.add_argument("--seed", type=int, default=42, help="Seed for dependent simulation runs.")
    parser.add_argument("--n-steps", type=int, default=140, help="Simulation timesteps.")
    parser.add_argument(
        "--week7-animation-summary-json",
        default="results/week7_animation_comparison_summary.json",
        help="Week 7 animation summary output path.",
    )
    parser.add_argument(
        "--week7-results-index-doc",
        default="docs/results_index.md",
        help="Week 7 results index markdown output path.",
    )
    parser.add_argument(
        "--week7-results-index-json",
        default="results/week7_results_index_summary.json",
        help="Week 7 results index summary JSON output path.",
    )
    parser.add_argument(
        "--week7-manifest-json",
        default="docs/assets/week7_animation_manifest.json",
        help="Week 7 animation manifest output path.",
    )
    parser.add_argument(
        "--week7-pack-summary-json",
        default="results/week7_animation_pack_summary.json",
        help="Week 7 pack summary output path.",
    )
    parser.add_argument(
        "--packet-doc",
        default="docs/week8_methods_results_packet.md",
        help="Week 8 packet markdown output path.",
    )
    parser.add_argument(
        "--captions-doc",
        default="docs/assets/week8_figure_captions.md",
        help="Week 8 captions markdown output path.",
    )
    parser.add_argument(
        "--methods-settings-json",
        default="results/week8_methods_settings_freeze.json",
        help="Week 8 methods/settings freeze JSON output path.",
    )
    parser.add_argument(
        "--tables-json",
        default="results/week8_final_tables.json",
        help="Week 8 final tables JSON output path.",
    )
    parser.add_argument(
        "--captions-json",
        default="results/week8_figure_captions.json",
        help="Week 8 figure captions JSON output path.",
    )
    parser.add_argument(
        "--commands-json",
        default="results/week8_reproducibility_commands.json",
        help="Week 8 reproducibility commands JSON output path.",
    )
    parser.add_argument(
        "--artifact-bundle-json",
        default="results/week8_frozen_artifact_bundle.json",
        help="Week 8 frozen artifact bundle JSON output path.",
    )
    parser.add_argument(
        "--submission-summary-json",
        default="results/week8_submission_packet_summary.json",
        help="Week 8 submission summary JSON output path.",
    )
    parser.add_argument(
        "--week7-hardened-validation",
        default="results/week7_hardened_validation.json",
        help="Week 7 hardened validation artifact path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week8_final_packaging_summary.json",
        help="Week 8 orchestration summary output path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit step plan only without executing commands.",
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

    steps: list[tuple[str, list[str]]] = [
        (
            "run_week7_pack",
            [
                "python3.11",
                "scripts/regenerate_week7_animation_pack.py",
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.n_steps),
                "--animation-summary-json",
                args.week7_animation_summary_json,
                "--results-index-doc",
                args.week7_results_index_doc,
                "--results-index-json",
                args.week7_results_index_json,
                "--manifest-json",
                args.week7_manifest_json,
                "--output-json",
                args.week7_pack_summary_json,
            ],
        ),
        (
            "build_week8_submission_packet",
            [
                "python3.11",
                "scripts/build_week8_submission_packet.py",
                "--output-packet-doc",
                args.packet_doc,
                "--output-captions-doc",
                args.captions_doc,
                "--output-methods-settings-json",
                args.methods_settings_json,
                "--output-tables-json",
                args.tables_json,
                "--output-captions-json",
                args.captions_json,
                "--output-commands-json",
                args.commands_json,
                "--output-bundle-json",
                args.artifact_bundle_json,
                "--output-summary-json",
                args.submission_summary_json,
                "--week7-pack-summary",
                args.week7_pack_summary_json,
                "--week7-animation-summary",
                args.week7_animation_summary_json,
                "--week7-manifest",
                args.week7_manifest_json,
                "--week7-results-index",
                args.week7_results_index_json,
                "--week7-hardened-validation",
                args.week7_hardened_validation,
            ],
        ),
    ]

    records: list[dict] = []
    for step_name, cmd in steps:
        print(f"[week8] {step_name}: {' '.join(cmd)}")
        elapsed = run_step(cmd, env=env, dry_run=args.dry_run)
        records.append(
            {
                "name": step_name,
                "command": cmd,
                "elapsed_s": elapsed,
                "executed": not args.dry_run,
            }
        )

    submission_summary = {}
    if not args.dry_run:
        with Path(args.submission_summary_json).open() as f:
            submission_summary = json.load(f)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_steps": args.n_steps,
        "dry_run": args.dry_run,
        "week7_pack_summary_json": args.week7_pack_summary_json,
        "week7_animation_summary_json": args.week7_animation_summary_json,
        "week7_results_index_doc": args.week7_results_index_doc,
        "week7_results_index_json": args.week7_results_index_json,
        "week7_manifest_json": args.week7_manifest_json,
        "packet_doc": args.packet_doc,
        "captions_doc": args.captions_doc,
        "methods_settings_json": args.methods_settings_json,
        "tables_json": args.tables_json,
        "captions_json": args.captions_json,
        "commands_json": args.commands_json,
        "artifact_bundle_json": args.artifact_bundle_json,
        "submission_summary_json": args.submission_summary_json,
        "steps": records,
        "n_steps_executed": sum(1 for r in records if r["executed"]),
        "acceptance_week8_one_command_regeneration_succeeds": True,
        "acceptance_freeze_benchmark_artifacts_and_methods_settings": bool(
            submission_summary.get("acceptance_methods_settings_frozen")
            and submission_summary.get("acceptance_artifact_bundle_complete")
        )
        if not args.dry_run
        else False,
        "acceptance_final_tables_and_captions_compiled": bool(
            submission_summary.get("acceptance_final_tables_generated")
            and submission_summary.get("acceptance_figure_captions_generated")
        )
        if not args.dry_run
        else False,
        "acceptance_methods_results_draft_packet_prepared": bool(
            submission_summary.get("acceptance_draft_packet_complete")
        )
        if not args.dry_run
        else False,
        "acceptance_top_level_success_criteria_met_or_waived": bool(
            submission_summary.get("acceptance_top_level_success_criteria_met_or_waived")
        )
        if not args.dry_run
        else False,
        "reference_config_hashes": submission_summary.get("reference_config_hashes", {}),
        "key_metrics": submission_summary.get("key_metrics", {}),
    }
    payload["passed"] = bool(
        payload["acceptance_week8_one_command_regeneration_succeeds"]
        and payload["acceptance_freeze_benchmark_artifacts_and_methods_settings"]
        and payload["acceptance_final_tables_and_captions_compiled"]
        and payload["acceptance_methods_results_draft_packet_prepared"]
        and payload["acceptance_top_level_success_criteria_met_or_waived"]
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
