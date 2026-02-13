"""Run hardened validation gates for Week 5 layered ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week5-config",
        default="configs/week5_layered_ablation.json",
        help="Week 5 config path.",
    )
    parser.add_argument(
        "--week4-baseline-csv",
        default="results/week4_collision_ablation.csv",
        help="Week 4 baseline CSV used for regression/runtime comparisons.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week5_hardened_validation.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--matrix-output-csv",
        default="results/week5_matrix_check.csv",
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
    parser.add_argument("--n-steps", type=int, default=150, help="Simulation steps for all checks.")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for reproducibility metric checks.",
    )
    parser.add_argument(
        "--runtime-budget-ratio-vs-week4",
        type=float,
        default=1.35,
        help="Runtime ratio budget for week5 layered_on / week4 spatial_hash.",
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


def run_week5(seed: int, n_steps: int, config_path: str, prefix: str) -> dict[str, str]:
    csv_path = f"{prefix}_ablation.csv"
    summary_path = f"{prefix}_ablation_summary.json"
    manifest_path = f"{prefix}_ablation_manifest.json"
    cmp_path = f"{prefix}_comparison.json"
    cmd = [
        "python3.11",
        "scripts/run_week5_layered_ablation.py",
        "--config-path",
        config_path,
        "--seed",
        str(seed),
        "--n-steps",
        str(n_steps),
        "--output-csv",
        csv_path,
        "--output-summary",
        summary_path,
        "--output-manifest",
        manifest_path,
        "--output-json",
        cmp_path,
    ]
    subprocess.check_call(cmd)
    return {
        "csv": csv_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "comparison": cmp_path,
    }


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


def write_matrix_csv(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    matrix_seeds = [int(s.strip()) for s in args.matrix_seeds.split(",") if s.strip()]
    matrix_labels = [s.strip() for s in args.matrix_labels.split(",") if s.strip()]

    run1 = run_week5(args.seed, args.n_steps, args.week5_config, "results/week5_repro_run1")
    run2 = run_week5(args.seed, args.n_steps, args.week5_config, "results/week5_repro_run2")

    run1_summary = load_json(run1["summary"])
    run2_summary = load_json(run2["summary"])
    run1_cmp = load_json(run1["comparison"])
    run2_cmp = load_json(run2["comparison"])

    repro_mismatches: list[str] = []
    if run1_summary.get("sweep_config_hash") != run2_summary.get("sweep_config_hash"):
        repro_mismatches.append("config_hash_mismatch:sweep_config_hash")

    key_metrics = [
        "layered_stability_rate",
        "layered_gi",
        "layered_disp_p95",
        "layered_outside_skull_frac",
        "robust_region_count",
    ]
    repro_mismatches.extend(compare_repro(run1_cmp, run2_cmp, key_metrics, atol=args.atol))

    acceptance_flags = [
        "acceptance_layered_mode_stable_on_reduced_grid",
        "acceptance_robust_parameter_region_found",
        "acceptance_week5_passed",
    ]
    for key in acceptance_flags:
        if bool(run1_cmp.get(key)) != bool(run2_cmp.get(key)):
            repro_mismatches.append(f"acceptance_flag_mismatch:{key}")

    reproducibility = {
        "seed": args.seed,
        "atol": args.atol,
        "first_summary": run1["summary"],
        "second_summary": run2["summary"],
        "first_comparison": run1["comparison"],
        "second_comparison": run2["comparison"],
        "mismatches": repro_mismatches,
        "passed": len(repro_mismatches) == 0,
    }

    matrix_records: list[dict] = []
    for seed in matrix_seeds:
        matrix_prefix = f"results/week5_matrix_seed{seed}"
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

    week4_rows = load_csv_rows(args.week4_baseline_csv)
    week4_by_label = {r["label"]: r for r in week4_rows}
    if "collision_spatial_hash" in week4_by_label:
        week4_ref = week4_by_label["collision_spatial_hash"]
    else:
        week4_ref = week4_rows[0]

    week5_rows = load_csv_rows(run1["csv"])
    week5_ref = {r["label"]: r for r in week5_rows}["layered_on_reference"]

    checks = []

    def add_check(name: str, week4_value: float, week5_value: float, max_increase: float) -> None:
        delta = week5_value - week4_value
        passed = delta <= max_increase
        checks.append(
            {
                "name": name,
                "week4": week4_value,
                "week5": week5_value,
                "delta": delta,
                "max_increase_allowed": max_increase,
                "passed": passed,
                "status": "acceptable" if passed else "regression",
            }
        )

    add_check(
        "outside_skull_frac",
        to_float(week4_ref["outside_skull_frac"]),
        to_float(week5_ref["outside_skull_frac"]),
        0.03,
    )
    add_check(
        "skull_penetration_p95",
        to_float(week4_ref["skull_penetration_p95"]),
        to_float(week5_ref["skull_penetration_p95"]),
        0.03,
    )
    add_check(
        "disp_p95",
        to_float(week4_ref["disp_p95"]),
        to_float(week5_ref["disp_p95"]),
        0.08,
    )

    gi_delta = to_float(week5_ref["gi"]) - to_float(week4_ref["gi"])
    gi_passed = gi_delta >= -0.40
    checks.append(
        {
            "name": "gi",
            "week4": to_float(week4_ref["gi"]),
            "week5": to_float(week5_ref["gi"]),
            "delta": gi_delta,
            "max_drop_allowed": -0.40,
            "passed": gi_passed,
            "status": "acceptable" if gi_passed else "regression",
        }
    )

    stable_passed = to_int(week5_ref["stable"]) >= to_int(week4_ref["stable"])
    checks.append(
        {
            "name": "stable",
            "week4": to_int(week4_ref["stable"]),
            "week5": to_int(week5_ref["stable"]),
            "delta": to_int(week5_ref["stable"]) - to_int(week4_ref["stable"]),
            "passed": stable_passed,
            "status": "acceptable" if stable_passed else "regression",
        }
    )

    regression = {
        "week4_reference_label": week4_ref["label"],
        "week5_reference_label": week5_ref["label"],
        "checks": checks,
        "passed": all(bool(c["passed"]) for c in checks),
    }

    runtime_ratio = to_float(week5_ref["runtime_s"]) / max(to_float(week4_ref["runtime_s"]), 1e-12)
    runtime_budget = {
        "week4_reference_runtime_s": to_float(week4_ref["runtime_s"]),
        "week5_reference_runtime_s": to_float(week5_ref["runtime_s"]),
        "runtime_ratio_week5_over_week4": runtime_ratio,
        "runtime_delta_s": to_float(week5_ref["runtime_s"]) - to_float(week4_ref["runtime_s"]),
        "budget_ratio": args.runtime_budget_ratio_vs_week4,
        "passed": runtime_ratio <= args.runtime_budget_ratio_vs_week4,
    }

    report = {
        "week5_comparison": run1["comparison"],
        "week5_summary": run1["summary"],
        "reproducibility": reproducibility,
        "matrix": matrix,
        "regression_vs_week4": regression,
        "runtime_budget_vs_week4": runtime_budget,
    }
    report["passed"] = bool(
        reproducibility["passed"]
        and matrix["passed"]
        and regression["passed"]
        and runtime_budget["passed"]
        and bool(run1_cmp.get("acceptance_week5_passed"))
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
