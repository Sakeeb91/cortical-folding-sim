"""Run hardened validation gates for Week 8 final packaging outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week5-config",
        default="configs/week5_layered_ablation.json",
        help="Week 5 config path used for matrix checks.",
    )
    parser.add_argument(
        "--week7-baseline-validation",
        default="results/week7_hardened_validation.json",
        help="Prior-week hardened validation artifact used for runtime baseline.",
    )
    parser.add_argument(
        "--week7-baseline-animation-summary",
        default="results/week7_animation_comparison_summary.json",
        help="Prior-week animation comparison summary used for regression checks.",
    )
    parser.add_argument(
        "--week7-baseline-pack-summary",
        default="results/week7_animation_pack_summary.json",
        help="Prior-week pack summary used for baseline acceptance flags.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week8_hardened_validation.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--matrix-output-csv",
        default="results/week8_matrix_check.csv",
        help="Output CSV path for matrix records.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility checks.")
    parser.add_argument(
        "--matrix-seeds",
        default="11,23,37",
        help="Comma-separated seeds for matrix checks.",
    )
    parser.add_argument(
        "--matrix-labels",
        default="layered_off_reference,layered_on_reference",
        help="Comma-separated run labels evaluated in matrix checks.",
    )
    parser.add_argument("--n-steps", type=int, default=140, help="Simulation steps for checks.")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for reproducibility metric checks.",
    )
    parser.add_argument(
        "--runtime-budget-ratio-vs-week7",
        type=float,
        default=1.25,
        help="Runtime ratio budget for week8 runtime vs week7 runtime.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def load_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(v) -> float:
    return float(v)


def to_int(v) -> int:
    return int(float(v))


def compare_repro(a: dict, b: dict, keys: list[str], atol: float) -> list[str]:
    mismatches: list[str] = []
    for key in keys:
        va = to_float(a.get(key, math.nan))
        vb = to_float(b.get(key, math.nan))
        if abs(va - vb) > atol:
            mismatches.append(
                f"metric_mismatch:{key}:first={va:.9f}:second={vb:.9f}:tol={atol:.1e}"
            )
    return mismatches


def _pipeline_runtime(summary: dict) -> float:
    return sum(float(step.get("elapsed_s", 0.0)) for step in summary.get("steps", []))


def run_week8(seed: int, n_steps: int, prefix: str) -> dict[str, str]:
    week7_pack_summary = f"{prefix}_week7_animation_pack_summary.json"
    week7_animation_summary = f"{prefix}_week7_animation_summary.json"
    week7_results_doc = f"{prefix}_results_index.md"
    week7_results_json = f"{prefix}_results_index_summary.json"
    week7_manifest = f"{prefix}_week7_animation_manifest.json"
    packet_doc = f"{prefix}_methods_results_packet.md"
    captions_doc = f"{prefix}_figure_captions.md"
    methods_json = f"{prefix}_methods_settings_freeze.json"
    tables_json = f"{prefix}_final_tables.json"
    captions_json = f"{prefix}_figure_captions.json"
    commands_json = f"{prefix}_repro_commands.json"
    bundle_json = f"{prefix}_frozen_artifact_bundle.json"
    submission_summary = f"{prefix}_submission_packet_summary.json"
    packaging_summary = f"{prefix}_final_packaging_summary.json"

    cmd = [
        "python3.11",
        "scripts/regenerate_week8_final_package.py",
        "--seed",
        str(seed),
        "--n-steps",
        str(n_steps),
        "--week7-animation-summary-json",
        week7_animation_summary,
        "--week7-results-index-doc",
        week7_results_doc,
        "--week7-results-index-json",
        week7_results_json,
        "--week7-manifest-json",
        week7_manifest,
        "--week7-pack-summary-json",
        week7_pack_summary,
        "--packet-doc",
        packet_doc,
        "--captions-doc",
        captions_doc,
        "--methods-settings-json",
        methods_json,
        "--tables-json",
        tables_json,
        "--captions-json",
        captions_json,
        "--commands-json",
        commands_json,
        "--artifact-bundle-json",
        bundle_json,
        "--submission-summary-json",
        submission_summary,
        "--output-json",
        packaging_summary,
    ]
    subprocess.check_call(cmd)
    return {
        "week7_pack_summary": week7_pack_summary,
        "week7_animation_summary": week7_animation_summary,
        "week7_results_doc": week7_results_doc,
        "week7_results_json": week7_results_json,
        "week7_manifest": week7_manifest,
        "packet_doc": packet_doc,
        "captions_doc": captions_doc,
        "methods_json": methods_json,
        "tables_json": tables_json,
        "captions_json": captions_json,
        "commands_json": commands_json,
        "bundle_json": bundle_json,
        "submission_summary": submission_summary,
        "packaging_summary": packaging_summary,
    }


def write_matrix_csv(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_ci_command(name: str, command: list[str]) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "elapsed_s": elapsed,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> None:
    args = parse_args()
    matrix_seeds = [int(s.strip()) for s in args.matrix_seeds.split(",") if s.strip()]
    matrix_labels = [s.strip() for s in args.matrix_labels.split(",") if s.strip()]

    baseline_validation = load_json(args.week7_baseline_validation)
    baseline_animation = load_json(args.week7_baseline_animation_summary)
    baseline_pack = load_json(args.week7_baseline_pack_summary)

    run1 = run_week8(args.seed, args.n_steps, "results/week8_repro_run1")
    run2 = run_week8(args.seed, args.n_steps, "results/week8_repro_run2")

    run1_submission = load_json(run1["submission_summary"])
    run2_submission = load_json(run2["submission_summary"])
    run1_packaging = load_json(run1["packaging_summary"])
    run2_packaging = load_json(run2["packaging_summary"])
    run1_week7 = load_json(run1["week7_animation_summary"])
    run2_week7 = load_json(run2["week7_animation_summary"])
    run1_bundle = load_json(run1["bundle_json"])
    run2_bundle = load_json(run2["bundle_json"])

    repro_mismatches: list[str] = []
    if run1_submission.get("reference_config_hashes") != run2_submission.get("reference_config_hashes"):
        repro_mismatches.append("config_hash_mismatch:week8_reference_config_hashes")

    key_metric_keys = [
        "forward_stability_rate",
        "forward_gi_plausible_rate",
        "week5_delta_gi_layered_minus_baseline",
        "week5_delta_outside_skull_frac_layered_minus_baseline",
        "week5_delta_disp_p95_layered_minus_baseline",
        "week7_delta_gi_improved_minus_baseline",
        "week7_delta_disp_p95_improved_minus_baseline",
        "week7_delta_outside_skull_frac_improved_minus_baseline",
    ]
    repro_mismatches.extend(
        compare_repro(
            run1_submission.get("key_metrics", {}),
            run2_submission.get("key_metrics", {}),
            key_metric_keys,
            atol=args.atol,
        )
    )

    week7_metric_keys = [
        "delta_gi_improved_minus_baseline",
        "delta_disp_p95_improved_minus_baseline",
        "delta_outside_skull_frac_improved_minus_baseline",
    ]
    repro_mismatches.extend(compare_repro(run1_week7, run2_week7, week7_metric_keys, atol=args.atol))

    acceptance_flags = [
        "acceptance_top_level_success_criteria_met_or_waived",
        "acceptance_draft_packet_complete",
        "acceptance_methods_settings_frozen",
        "acceptance_final_tables_generated",
        "acceptance_figure_captions_generated",
        "acceptance_submission_repro_commands_ready",
        "acceptance_artifact_bundle_complete",
    ]
    for key in acceptance_flags:
        if bool(run1_submission.get(key)) != bool(run2_submission.get(key)):
            repro_mismatches.append(f"acceptance_flag_mismatch:submission:{key}")

    pack_flags = [
        "acceptance_week8_one_command_regeneration_succeeds",
        "acceptance_freeze_benchmark_artifacts_and_methods_settings",
        "acceptance_final_tables_and_captions_compiled",
        "acceptance_methods_results_draft_packet_prepared",
        "acceptance_top_level_success_criteria_met_or_waived",
    ]
    for key in pack_flags:
        if bool(run1_packaging.get(key)) != bool(run2_packaging.get(key)):
            repro_mismatches.append(f"acceptance_flag_mismatch:packaging:{key}")

    if bool(run1_bundle.get("all_artifacts_present")) != bool(run2_bundle.get("all_artifacts_present")):
        repro_mismatches.append("acceptance_flag_mismatch:bundle_all_artifacts_present")

    reproducibility = {
        "seed": args.seed,
        "atol": args.atol,
        "first_packaging_summary": run1["packaging_summary"],
        "second_packaging_summary": run2["packaging_summary"],
        "first_submission_summary": run1["submission_summary"],
        "second_submission_summary": run2["submission_summary"],
        "mismatches": repro_mismatches,
        "passed": len(repro_mismatches) == 0,
    }

    matrix_records: list[dict] = []
    for seed in matrix_seeds:
        matrix_prefix = f"results/week8_matrix_seed{seed}"
        matrix_csv = f"{matrix_prefix}.csv"
        matrix_summary = f"{matrix_prefix}_summary.json"
        matrix_manifest = f"{matrix_prefix}_manifest.json"
        cmd = [
            "python3.11",
            "scripts/run_forward_sweep.py",
            "--config-path",
            args.week5_config,
            "--seed",
            str(seed),
            "--n-steps",
            str(args.n_steps),
            "--output-csv",
            matrix_csv,
            "--output-summary",
            matrix_summary,
            "--output-manifest",
            matrix_manifest,
        ]
        subprocess.check_call(cmd)
        row_by_label = {r["label"]: r for r in load_csv_rows(matrix_csv)}
        for label in matrix_labels:
            row = row_by_label[label]
            matrix_records.append(
                {
                    "seed": seed,
                    "label": label,
                    "stable": to_int(row["stable"]),
                    "fail_reason": row["fail_reason"],
                    "gi": to_float(row["gi"]),
                    "disp_p95": to_float(row["disp_p95"]),
                    "outside_skull_frac": to_float(row["outside_skull_frac"]),
                    "runtime_s": to_float(row["runtime_s"]),
                }
            )

    write_matrix_csv(args.matrix_output_csv, matrix_records)
    total_matrix = len(matrix_records)
    stable_matrix = sum(int(r["stable"]) for r in matrix_records)
    failure_reason_counts: dict[str, int] = {}
    for record in matrix_records:
        if record["stable"] == 0:
            reason = record["fail_reason"]
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

    matrix = {
        "seeds": matrix_seeds,
        "labels": matrix_labels,
        "n_records": total_matrix,
        "stability_rate": stable_matrix / max(total_matrix, 1),
        "failure_reason_counts": failure_reason_counts,
        "matrix_csv": args.matrix_output_csv,
        "passed": total_matrix >= 6 and (stable_matrix / max(total_matrix, 1)) >= 0.95,
    }

    checks = []

    def add_check(name: str, baseline_value: float, week8_value: float, max_increase: float) -> None:
        delta = week8_value - baseline_value
        passed = delta <= max_increase
        checks.append(
            {
                "name": name,
                "week7": baseline_value,
                "week8": week8_value,
                "delta": delta,
                "max_increase_allowed": max_increase,
                "passed": passed,
                "status": "acceptable" if passed else "regression",
            }
        )

    add_check(
        "layered_outside_skull_frac",
        to_float(baseline_animation["improved_metrics"]["outside_skull_frac"]),
        to_float(run1_week7["improved_metrics"]["outside_skull_frac"]),
        0.03,
    )
    add_check(
        "layered_disp_p95",
        to_float(baseline_animation["improved_metrics"]["disp_p95"]),
        to_float(run1_week7["improved_metrics"]["disp_p95"]),
        0.08,
    )

    gi_delta = to_float(run1_week7["improved_metrics"]["gi"]) - to_float(
        baseline_animation["improved_metrics"]["gi"]
    )
    gi_passed = gi_delta >= -0.40
    checks.append(
        {
            "name": "layered_gi",
            "week7": to_float(baseline_animation["improved_metrics"]["gi"]),
            "week8": to_float(run1_week7["improved_metrics"]["gi"]),
            "delta": gi_delta,
            "max_drop_allowed": -0.40,
            "passed": gi_passed,
            "status": "acceptable" if gi_passed else "regression",
        }
    )

    regression = {
        "week7_reference_validation": args.week7_baseline_validation,
        "week7_reference_animation_summary": args.week7_baseline_animation_summary,
        "week7_reference_pack_summary": args.week7_baseline_pack_summary,
        "week8_reference_packaging_summary": run1["packaging_summary"],
        "week8_reference_submission_summary": run1["submission_summary"],
        "checks": checks,
        "passed": all(bool(c["passed"]) for c in checks)
        and bool(run1_submission.get("acceptance_top_level_success_criteria_met_or_waived"))
        and bool(run1_packaging.get("acceptance_final_tables_and_captions_compiled"))
        and bool(baseline_pack.get("acceptance_animation_regeneration_succeeds", True)),
    }

    week7_runtime_s = float(
        baseline_validation.get("runtime_budget_vs_week6", {}).get("week7_reference_runtime_s", 0.0)
    )
    week8_runtime_s = _pipeline_runtime(run1_packaging)
    runtime_ratio = week8_runtime_s / max(week7_runtime_s, 1e-12)
    runtime_budget = {
        "week7_reference_runtime_s": week7_runtime_s,
        "week8_reference_runtime_s": week8_runtime_s,
        "runtime_ratio_week8_over_week7": runtime_ratio,
        "runtime_delta_s": week8_runtime_s - week7_runtime_s,
        "budget_ratio": args.runtime_budget_ratio_vs_week7,
        "passed": runtime_ratio <= args.runtime_budget_ratio_vs_week7,
    }

    ci_parity_checks = [
        run_ci_command("validation_quick", ["./scripts/run_validation_quick.sh"]),
        run_ci_command("validation_full", ["./scripts/run_validation_full.sh"]),
    ]
    ci_parity = {
        "checks": ci_parity_checks,
        "passed": all(bool(c["passed"]) for c in ci_parity_checks),
    }

    report = {
        "week8_packaging_summary": run1["packaging_summary"],
        "week8_submission_summary": run1["submission_summary"],
        "week8_frozen_bundle": run1["bundle_json"],
        "week8_week7_animation_summary": run1["week7_animation_summary"],
        "reproducibility": reproducibility,
        "matrix": matrix,
        "regression_vs_week7": regression,
        "runtime_budget_vs_week7": runtime_budget,
        "ci_parity": ci_parity,
    }
    report["passed"] = bool(
        reproducibility["passed"]
        and matrix["passed"]
        and regression["passed"]
        and runtime_budget["passed"]
        and ci_parity["passed"]
        and bool(run1_packaging.get("acceptance_week8_one_command_regeneration_succeeds"))
        and bool(run1_packaging.get("acceptance_freeze_benchmark_artifacts_and_methods_settings"))
        and bool(run1_packaging.get("acceptance_final_tables_and_captions_compiled"))
        and bool(run1_packaging.get("acceptance_methods_results_draft_packet_prepared"))
        and bool(run1_submission.get("acceptance_top_level_success_criteria_met_or_waived"))
        and bool(run1_submission.get("acceptance_draft_packet_complete"))
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
