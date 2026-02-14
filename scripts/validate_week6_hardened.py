"""Run hardened validation gates for Week 6 figure pipeline outputs."""

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
        "--week5-baseline-comparison",
        default="results/week5_layered_comparison.json",
        help="Prior week baseline comparison JSON path.",
    )
    parser.add_argument(
        "--week5-baseline-summary",
        default="results/week5_layered_ablation_summary.json",
        help="Prior week baseline summary JSON path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week6_hardened_validation.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--matrix-output-csv",
        default="results/week6_matrix_check.csv",
        help="Output CSV for matrix records.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility checks.")
    parser.add_argument(
        "--matrix-seeds",
        default="11,23,37",
        help="Comma-separated seeds for matrix check.",
    )
    parser.add_argument(
        "--matrix-labels",
        default="layered_on_reference,layered_clip_tight",
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
        "--runtime-budget-ratio-vs-week5",
        type=float,
        default=1.35,
        help="Runtime ratio budget for week6 layered runtime vs week5 baseline runtime.",
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


def run_week6(seed: int, n_steps: int, prefix: str) -> dict[str, str]:
    summary_path = f"{prefix}_pipeline_summary.json"
    manifest_path = f"{prefix}_figure_manifest.json"
    cmd = [
        "python3.11",
        "scripts/regenerate_week6_figures.py",
        "--seed",
        str(seed),
        "--n-steps",
        str(n_steps),
        "--output-json",
        summary_path,
        "--manifest-json",
        manifest_path,
    ]
    subprocess.check_call(cmd)

    snap = {
        "summary": summary_path,
        "manifest": manifest_path,
        "week3_comparison": f"{prefix}_week3_comparison.json",
        "week4_comparison": f"{prefix}_week4_comparison.json",
        "week5_comparison": f"{prefix}_week5_comparison.json",
        "week5_summary": f"{prefix}_week5_summary.json",
    }
    shutil.copyfile("results/week3_anisotropy_comparison.json", snap["week3_comparison"])
    shutil.copyfile("results/week4_collision_comparison.json", snap["week4_comparison"])
    shutil.copyfile("results/week5_layered_comparison.json", snap["week5_comparison"])
    shutil.copyfile("results/week5_layered_ablation_summary.json", snap["week5_summary"])
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

    baseline_week5_comparison = load_json(args.week5_baseline_comparison)
    baseline_week5_summary = load_json(args.week5_baseline_summary)

    run1 = run_week6(args.seed, args.n_steps, "results/week6_repro_run1")
    run2 = run_week6(args.seed, args.n_steps, "results/week6_repro_run2")

    run1_summary = load_json(run1["summary"])
    run2_summary = load_json(run2["summary"])
    run1_manifest = load_json(run1["manifest"])
    run2_manifest = load_json(run2["manifest"])
    run1_week3 = load_json(run1["week3_comparison"])
    run2_week3 = load_json(run2["week3_comparison"])
    run1_week4 = load_json(run1["week4_comparison"])
    run2_week4 = load_json(run2["week4_comparison"])
    run1_week5 = load_json(run1["week5_comparison"])
    run2_week5 = load_json(run2["week5_comparison"])
    run1_week5_summary = load_json(run1["week5_summary"])
    run2_week5_summary = load_json(run2["week5_summary"])

    repro_mismatches: list[str] = []
    if run1_week5_summary.get("sweep_config_hash") != run2_week5_summary.get("sweep_config_hash"):
        repro_mismatches.append("config_hash_mismatch:week5_sweep_config_hash")

    week3_keys = ["delta_gi", "delta_curv_p90", "delta_disp_p95"]
    week4_keys = [
        "reduction_overlap_count_vs_sampled",
        "spatial_hash_collision_overlap_count",
    ]
    week5_keys = ["layered_gi", "layered_disp_p95", "robust_region_count"]
    repro_mismatches.extend(compare_repro(run1_week3, run2_week3, week3_keys, atol=args.atol))
    repro_mismatches.extend(compare_repro(run1_week4, run2_week4, week4_keys, atol=args.atol))
    repro_mismatches.extend(compare_repro(run1_week5, run2_week5, week5_keys, atol=args.atol))

    acceptance_flags = [
        "acceptance_one_command_regeneration_succeeds",
        "acceptance_all_figures_mapped_source_run_ids",
    ]
    for key in acceptance_flags:
        if bool(run1_summary.get(key)) != bool(run2_summary.get(key)):
            repro_mismatches.append(f"acceptance_flag_mismatch:{key}")
    if bool(run1_manifest.get("all_figures_have_source_run_ids")) != bool(
        run2_manifest.get("all_figures_have_source_run_ids")
    ):
        repro_mismatches.append("acceptance_flag_mismatch:manifest_all_figures_have_source_run_ids")

    reproducibility = {
        "seed": args.seed,
        "atol": args.atol,
        "first_summary": run1["summary"],
        "second_summary": run2["summary"],
        "first_manifest": run1["manifest"],
        "second_manifest": run2["manifest"],
        "first_week5_summary": run1["week5_summary"],
        "second_week5_summary": run2["week5_summary"],
        "mismatches": repro_mismatches,
        "passed": len(repro_mismatches) == 0,
    }

    matrix_records: list[dict] = []
    for seed in matrix_seeds:
        matrix_prefix = f"results/week6_matrix_seed{seed}"
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

    by_label: dict[str, dict[str, float]] = {}
    for label in matrix_labels:
        subset = [r for r in matrix_records if r["label"] == label]
        by_label[label] = {
            "n": len(subset),
            "stable_rate": sum(int(r["stable"]) for r in subset) / max(len(subset), 1),
            "mean_gi": sum(float(r["gi"]) for r in subset) / max(len(subset), 1),
            "mean_disp_p95": sum(float(r["disp_p95"]) for r in subset) / max(len(subset), 1),
        }

    matrix = {
        "seeds": matrix_seeds,
        "labels": matrix_labels,
        "n_records": total_matrix,
        "stability_rate": stable_matrix / max(total_matrix, 1),
        "failure_reason_counts": failure_reason_counts,
        "by_label": by_label,
        "matrix_csv": args.matrix_output_csv,
        "passed": total_matrix >= 6 and (stable_matrix / max(total_matrix, 1)) >= 0.95,
    }

    checks = []

    def add_check(name: str, baseline_value: float, week6_value: float, max_increase: float) -> None:
        delta = week6_value - baseline_value
        passed = delta <= max_increase
        checks.append(
            {
                "name": name,
                "week5": baseline_value,
                "week6": week6_value,
                "delta": delta,
                "max_increase_allowed": max_increase,
                "passed": passed,
                "status": "acceptable" if passed else "regression",
            }
        )

    add_check(
        "layered_outside_skull_frac",
        to_float(baseline_week5_comparison["layered_outside_skull_frac"]),
        to_float(run1_week5["layered_outside_skull_frac"]),
        0.03,
    )
    add_check(
        "layered_disp_p95",
        to_float(baseline_week5_comparison["layered_disp_p95"]),
        to_float(run1_week5["layered_disp_p95"]),
        0.08,
    )

    gi_delta = to_float(run1_week5["layered_gi"]) - to_float(baseline_week5_comparison["layered_gi"])
    gi_passed = gi_delta >= -0.40
    checks.append(
        {
            "name": "layered_gi",
            "week5": to_float(baseline_week5_comparison["layered_gi"]),
            "week6": to_float(run1_week5["layered_gi"]),
            "delta": gi_delta,
            "max_drop_allowed": -0.40,
            "passed": gi_passed,
            "status": "acceptable" if gi_passed else "regression",
        }
    )

    robust_delta = int(run1_week5["robust_region_count"]) - int(
        baseline_week5_comparison["robust_region_count"]
    )
    robust_passed = robust_delta >= 0
    checks.append(
        {
            "name": "robust_region_count",
            "week5": int(baseline_week5_comparison["robust_region_count"]),
            "week6": int(run1_week5["robust_region_count"]),
            "delta": robust_delta,
            "passed": robust_passed,
            "status": "acceptable" if robust_passed else "regression",
        }
    )

    regression = {
        "week5_reference_summary": args.week5_baseline_summary,
        "week5_reference_comparison": args.week5_baseline_comparison,
        "week6_comparison": run1["week5_comparison"],
        "checks": checks,
        "passed": all(bool(c["passed"]) for c in checks),
    }

    runtime_ratio = to_float(run1_week5["layered_runtime_s"]) / max(
        to_float(baseline_week5_comparison["layered_runtime_s"]), 1e-12
    )
    runtime_budget = {
        "week5_reference_runtime_s": to_float(baseline_week5_comparison["layered_runtime_s"]),
        "week6_reference_runtime_s": to_float(run1_week5["layered_runtime_s"]),
        "runtime_ratio_week6_over_week5": runtime_ratio,
        "runtime_delta_s": to_float(run1_week5["layered_runtime_s"])
        - to_float(baseline_week5_comparison["layered_runtime_s"]),
        "budget_ratio": args.runtime_budget_ratio_vs_week5,
        "passed": runtime_ratio <= args.runtime_budget_ratio_vs_week5,
        "week6_pipeline_total_runtime_s": sum(
            float(step.get("elapsed_s", 0.0)) for step in run1_summary.get("steps", [])
        ),
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
        "week6_pipeline_summary": run1["summary"],
        "week6_figure_manifest": run1["manifest"],
        "reproducibility": reproducibility,
        "matrix": matrix,
        "regression_vs_week5": regression,
        "runtime_budget_vs_week5": runtime_budget,
        "ci_parity": ci_parity,
        "baseline_snapshot_week5_summary": baseline_week5_summary,
    }
    report["passed"] = bool(
        reproducibility["passed"]
        and matrix["passed"]
        and regression["passed"]
        and runtime_budget["passed"]
        and ci_parity["passed"]
        and bool(run1_summary.get("acceptance_one_command_regeneration_succeeds"))
        and bool(run1_summary.get("acceptance_all_figures_mapped_source_run_ids"))
        and bool(run1_manifest.get("all_figures_have_source_run_ids"))
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
