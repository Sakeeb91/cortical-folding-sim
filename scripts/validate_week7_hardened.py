"""Run hardened validation gates for Week 7 animation comparison pack."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
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
        "--week6-baseline-comparison",
        default="results/week6_repro_run1_week5_comparison.json",
        help="Prior-week comparison artifact used as regression baseline.",
    )
    parser.add_argument(
        "--week6-baseline-summary",
        default="results/week6_figure_pipeline_summary.json",
        help="Prior-week pipeline summary artifact used for runtime baseline.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week7_hardened_validation.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--matrix-output-csv",
        default="results/week7_matrix_check.csv",
        help="Output CSV for matrix records.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility checks.",
    )
    parser.add_argument(
        "--matrix-seeds",
        default="11,23,37",
        help="Comma-separated seeds for matrix check.",
    )
    parser.add_argument(
        "--matrix-labels",
        default="layered_off_reference,layered_on_reference",
        help="Comma-separated run labels evaluated in matrix check.",
    )
    parser.add_argument("--n-steps", type=int, default=140, help="Simulation steps for checks.")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for reproducibility metric checks.",
    )
    parser.add_argument(
        "--runtime-budget-ratio-vs-week6",
        type=float,
        default=1.25,
        help="Runtime ratio budget for week7 pipeline runtime vs week6 pipeline runtime.",
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


def run_week7(seed: int, n_steps: int, prefix: str) -> dict[str, str]:
    pack_summary_path = f"{prefix}_animation_pack_summary.json"
    animation_summary_path = f"{prefix}_animation_summary.json"
    results_index_doc_path = f"{prefix}_results_index.md"
    results_index_json_path = f"{prefix}_results_index_summary.json"
    manifest_path = f"{prefix}_animation_manifest.json"

    cmd = [
        "python3.11",
        "scripts/regenerate_week7_animation_pack.py",
        "--seed",
        str(seed),
        "--n-steps",
        str(n_steps),
        "--output-json",
        pack_summary_path,
        "--animation-summary-json",
        animation_summary_path,
        "--results-index-doc",
        results_index_doc_path,
        "--results-index-json",
        results_index_json_path,
        "--manifest-json",
        manifest_path,
    ]
    subprocess.check_call(cmd)

    snap = {
        "pack_summary": pack_summary_path,
        "animation_summary": animation_summary_path,
        "results_index_doc": results_index_doc_path,
        "results_index_json": results_index_json_path,
        "manifest": manifest_path,
        "week5_comparison": f"{prefix}_week5_comparison.json",
    }
    shutil.copyfile("results/week5_layered_comparison.json", snap["week5_comparison"])
    return snap


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

    baseline_week6_comparison = load_json(args.week6_baseline_comparison)
    baseline_week6_summary = load_json(args.week6_baseline_summary)

    run1 = run_week7(args.seed, args.n_steps, "results/week7_repro_run1")
    run2 = run_week7(args.seed, args.n_steps, "results/week7_repro_run2")

    run1_pack = load_json(run1["pack_summary"])
    run2_pack = load_json(run2["pack_summary"])
    run1_anim = load_json(run1["animation_summary"])
    run2_anim = load_json(run2["animation_summary"])
    run1_manifest = load_json(run1["manifest"])
    run2_manifest = load_json(run2["manifest"])
    run1_results_index = load_json(run1["results_index_json"])
    run2_results_index = load_json(run2["results_index_json"])

    repro_mismatches: list[str] = []
    if run1_anim.get("sweep_config_hash") != run2_anim.get("sweep_config_hash"):
        repro_mismatches.append("config_hash_mismatch:week7_sweep_config_hash")

    metric_keys = [
        "delta_gi_improved_minus_baseline",
        "delta_disp_p95_improved_minus_baseline",
        "delta_outside_skull_frac_improved_minus_baseline",
    ]
    repro_mismatches.extend(compare_repro(run1_anim, run2_anim, metric_keys, atol=args.atol))

    baseline_metric_keys = ["gi", "disp_p95", "outside_skull_frac"]
    repro_mismatches.extend(
        compare_repro(
            run1_anim.get("baseline_metrics", {}),
            run2_anim.get("baseline_metrics", {}),
            baseline_metric_keys,
            atol=args.atol,
        )
    )
    repro_mismatches.extend(
        compare_repro(
            run1_anim.get("improved_metrics", {}),
            run2_anim.get("improved_metrics", {}),
            baseline_metric_keys,
            atol=args.atol,
        )
    )

    acceptance_flags = [
        "acceptance_animation_regeneration_succeeds",
        "acceptance_results_index_claims_linked",
        "acceptance_animation_outputs_mapped",
        "animation_acceptance_both_runs_stable",
    ]
    for key in acceptance_flags:
        if bool(run1_pack.get(key)) != bool(run2_pack.get(key)):
            repro_mismatches.append(f"acceptance_flag_mismatch:{key}")

    if bool(run1_manifest.get("all_assets_have_source_run_ids")) != bool(
        run2_manifest.get("all_assets_have_source_run_ids")
    ):
        repro_mismatches.append("acceptance_flag_mismatch:manifest_all_assets_have_source_run_ids")

    if bool(run1_results_index.get("all_claims_have_linked_artifacts")) != bool(
        run2_results_index.get("all_claims_have_linked_artifacts")
    ):
        repro_mismatches.append("acceptance_flag_mismatch:results_index_all_claims_linked")

    reproducibility = {
        "seed": args.seed,
        "atol": args.atol,
        "first_pack_summary": run1["pack_summary"],
        "second_pack_summary": run2["pack_summary"],
        "first_animation_summary": run1["animation_summary"],
        "second_animation_summary": run2["animation_summary"],
        "first_manifest": run1["manifest"],
        "second_manifest": run2["manifest"],
        "mismatches": repro_mismatches,
        "passed": len(repro_mismatches) == 0,
    }

    matrix_records: list[dict] = []
    for seed in matrix_seeds:
        matrix_prefix = f"results/week7_matrix_seed{seed}"
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
    for rec in matrix_records:
        if rec["stable"] == 0:
            reason = rec["fail_reason"]
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

    def add_check(name: str, baseline_value: float, week7_value: float, max_increase: float) -> None:
        delta = week7_value - baseline_value
        passed = delta <= max_increase
        checks.append(
            {
                "name": name,
                "week6": baseline_value,
                "week7": week7_value,
                "delta": delta,
                "max_increase_allowed": max_increase,
                "passed": passed,
                "status": "acceptable" if passed else "regression",
            }
        )

    add_check(
        "layered_outside_skull_frac",
        to_float(baseline_week6_comparison["layered_outside_skull_frac"]),
        to_float(run1_anim["improved_metrics"]["outside_skull_frac"]),
        0.03,
    )
    add_check(
        "layered_disp_p95",
        to_float(baseline_week6_comparison["layered_disp_p95"]),
        to_float(run1_anim["improved_metrics"]["disp_p95"]),
        0.08,
    )

    gi_delta = to_float(run1_anim["improved_metrics"]["gi"]) - to_float(
        baseline_week6_comparison["layered_gi"]
    )
    gi_passed = gi_delta >= -0.40
    checks.append(
        {
            "name": "layered_gi",
            "week6": to_float(baseline_week6_comparison["layered_gi"]),
            "week7": to_float(run1_anim["improved_metrics"]["gi"]),
            "delta": gi_delta,
            "max_drop_allowed": -0.40,
            "passed": gi_passed,
            "status": "acceptable" if gi_passed else "regression",
        }
    )

    regression = {
        "week6_reference_summary": args.week6_baseline_summary,
        "week6_reference_comparison": args.week6_baseline_comparison,
        "week7_comparison": run1["animation_summary"],
        "checks": checks,
        "passed": all(bool(c["passed"]) for c in checks),
    }

    week6_runtime_s = _pipeline_runtime(baseline_week6_summary)
    week7_runtime_s = _pipeline_runtime(run1_pack)
    runtime_ratio = week7_runtime_s / max(week6_runtime_s, 1e-12)
    runtime_budget = {
        "week6_reference_runtime_s": week6_runtime_s,
        "week7_reference_runtime_s": week7_runtime_s,
        "runtime_ratio_week7_over_week6": runtime_ratio,
        "runtime_delta_s": week7_runtime_s - week6_runtime_s,
        "budget_ratio": args.runtime_budget_ratio_vs_week6,
        "passed": runtime_ratio <= args.runtime_budget_ratio_vs_week6,
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
        "week7_pack_summary": run1["pack_summary"],
        "week7_animation_summary": run1["animation_summary"],
        "week7_animation_manifest": run1["manifest"],
        "week7_results_index_summary": run1["results_index_json"],
        "reproducibility": reproducibility,
        "matrix": matrix,
        "regression_vs_week6": regression,
        "runtime_budget_vs_week6": runtime_budget,
        "ci_parity": ci_parity,
    }
    report["passed"] = bool(
        reproducibility["passed"]
        and matrix["passed"]
        and regression["passed"]
        and runtime_budget["passed"]
        and ci_parity["passed"]
        and bool(run1_pack.get("acceptance_animation_regeneration_succeeds"))
        and bool(run1_pack.get("acceptance_results_index_claims_linked"))
        and bool(run1_pack.get("acceptance_animation_outputs_mapped"))
        and bool(run1_manifest.get("all_assets_have_source_run_ids"))
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
