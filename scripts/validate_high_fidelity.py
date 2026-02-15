"""Run hardened validation gates for high-fidelity simulation mode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from cortical_folding.benchmarking import config_hash, load_grid_config
from cortical_folding.reproducibility import compare_summary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-config",
        default="configs/forward_sweep_baseline.json",
        help="Baseline sweep config for regression/runtime comparisons.",
    )
    parser.add_argument(
        "--high-fidelity-config",
        default="configs/high_fidelity_forward_sweep.json",
        help="High-fidelity sweep config.",
    )
    parser.add_argument(
        "--matrix-config",
        default="configs/high_fidelity_matrix.json",
        help="Config used for matrix stability checks.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/high_fidelity",
        help="Output directory for reports and run artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility runs.")
    parser.add_argument(
        "--matrix-seeds",
        default="11,23,37",
        help="Comma-separated seeds for matrix runs.",
    )
    parser.add_argument(
        "--matrix-labels",
        default="hf_matrix_reference,hf_matrix_layered",
        help="Comma-separated labels used in matrix checks.",
    )
    parser.add_argument("--n-steps", type=int, default=120, help="Simulation timesteps.")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for reproducibility checks.",
    )
    parser.add_argument(
        "--runtime-budget-ratio",
        type=float,
        default=2.0,
        help="Max allowed runtime ratio (high_fidelity / baseline).",
    )
    parser.add_argument(
        "--skip-ci-parity",
        action="store_true",
        help="Skip CI parity script execution.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit planned report contract without executing simulation commands.",
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


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> dict:
    t0 = time.perf_counter()
    if dry_run:
        return {
            "command": cmd,
            "returncode": 0,
            "passed": True,
            "elapsed_s": 0.0,
            "stdout_tail": "",
            "stderr_tail": "",
            "executed": False,
        }
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "elapsed_s": time.perf_counter() - t0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "executed": True,
    }


def sum_runtime_from_csv(path: str | Path) -> float:
    rows = load_csv_rows(path)
    return float(sum(to_float(row.get("runtime_s", 0.0)) for row in rows))


def _stable_metric_mean(rows: list[dict], key: str) -> float:
    stable_rows = [row for row in rows if to_int(row.get("stable", 0)) == 1]
    if not stable_rows:
        return float("nan")
    return float(sum(to_float(row.get(key, 0.0)) for row in stable_rows) / len(stable_rows))


def regression_checks(
    baseline_summary: dict,
    hf_summary: dict,
    baseline_rows: list[dict],
    hf_rows: list[dict],
) -> list[dict]:
    checks: list[dict] = []

    def add_min_check(name: str, baseline_key: str, hf_key: str, max_drop: float) -> None:
        b = to_float(baseline_summary.get(baseline_key, math.nan))
        h = to_float(hf_summary.get(hf_key, math.nan))
        delta = h - b
        passed = delta >= -abs(max_drop)
        checks.append(
            {
                "name": name,
                "baseline": b,
                "high_fidelity": h,
                "delta": delta,
                "max_drop_allowed": -abs(max_drop),
                "status": "pass" if passed else "fail",
                "passed": passed,
            }
        )

    add_min_check("stability_rate", "stability_rate", "stability_rate", max_drop=0.05)
    baseline_disp = _stable_metric_mean(baseline_rows, "disp_p95")
    hf_disp = _stable_metric_mean(hf_rows, "disp_p95")
    disp_delta = hf_disp - baseline_disp
    disp_passed = disp_delta <= 0.05
    checks.append(
        {
            "name": "disp_p95_mean",
            "baseline": baseline_disp,
            "high_fidelity": hf_disp,
            "delta": disp_delta,
            "max_increase_allowed": 0.05,
            "status": "pass" if disp_passed else "fail",
            "passed": disp_passed,
        }
    )

    baseline_outside = _stable_metric_mean(baseline_rows, "outside_skull_frac")
    hf_outside = _stable_metric_mean(hf_rows, "outside_skull_frac")
    outside_delta = hf_outside - baseline_outside
    outside_passed = outside_delta <= 0.05
    checks.append(
        {
            "name": "outside_skull_frac_mean",
            "baseline": baseline_outside,
            "high_fidelity": hf_outside,
            "delta": outside_delta,
            "max_increase_allowed": 0.05,
            "status": "pass" if outside_passed else "fail",
            "passed": outside_passed,
        }
    )

    gi_value = to_float(hf_summary.get("gi_mean", math.nan))
    gi_passed = gi_value >= 0.60
    checks.append(
        {
            "name": "gi_mean_absolute_floor",
            "baseline": to_float(baseline_summary.get("gi_mean", math.nan)),
            "high_fidelity": gi_value,
            "delta": gi_value - to_float(baseline_summary.get("gi_mean", math.nan)),
            "min_allowed": 0.60,
            "status": "pass" if gi_passed else "fail",
            "passed": gi_passed,
        }
    )
    return checks


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = out_dir / "baseline_forward_sweep.csv"
    baseline_summary = out_dir / "baseline_forward_sweep_summary.json"
    baseline_manifest = out_dir / "baseline_forward_sweep_manifest.json"

    hf_csv = out_dir / "high_fidelity_forward_sweep.csv"
    hf_summary = out_dir / "high_fidelity_forward_sweep_summary.json"
    hf_manifest = out_dir / "high_fidelity_forward_sweep_manifest.json"

    hf_rerun_csv = out_dir / "high_fidelity_forward_sweep_rerun.csv"
    hf_rerun_summary = out_dir / "high_fidelity_forward_sweep_rerun_summary.json"
    hf_rerun_manifest = out_dir / "high_fidelity_forward_sweep_rerun_manifest.json"

    matrix_csv = out_dir / "matrix_stability.csv"
    runtime_stats_json = out_dir / "runtime_stats.json"
    config_hashes_json = out_dir / "config_hashes.json"
    report_json = out_dir / "validation_report.json"

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    matrix_seeds = [int(s.strip()) for s in args.matrix_seeds.split(",") if s.strip()]
    matrix_labels = [s.strip() for s in args.matrix_labels.split(",") if s.strip()]

    command_records: list[dict] = []
    command_records.append(
        run_command(
            [
                "python3.11",
                "scripts/run_forward_sweep.py",
                "--config-path",
                args.baseline_config,
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.n_steps),
                "--output-csv",
                str(baseline_csv),
                "--output-summary",
                str(baseline_summary),
                "--output-manifest",
                str(baseline_manifest),
            ],
            env=env,
            dry_run=args.dry_run,
        )
    )
    command_records.append(
        run_command(
            [
                "python3.11",
                "scripts/run_forward_sweep.py",
                "--mode",
                "high_fidelity",
                "--config-path",
                args.high_fidelity_config,
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.n_steps),
                "--output-csv",
                str(hf_csv),
                "--output-summary",
                str(hf_summary),
                "--output-manifest",
                str(hf_manifest),
            ],
            env=env,
            dry_run=args.dry_run,
        )
    )
    command_records.append(
        run_command(
            [
                "python3.11",
                "scripts/run_forward_sweep.py",
                "--mode",
                "high_fidelity",
                "--config-path",
                args.high_fidelity_config,
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.n_steps),
                "--output-csv",
                str(hf_rerun_csv),
                "--output-summary",
                str(hf_rerun_summary),
                "--output-manifest",
                str(hf_rerun_manifest),
            ],
            env=env,
            dry_run=args.dry_run,
        )
    )

    matrix_records: list[dict] = []
    for seed in matrix_seeds:
        matrix_seed_csv = out_dir / f"matrix_seed_{seed}.csv"
        matrix_seed_summary = out_dir / f"matrix_seed_{seed}_summary.json"
        matrix_seed_manifest = out_dir / f"matrix_seed_{seed}_manifest.json"
        command_records.append(
            run_command(
                [
                    "python3.11",
                    "scripts/run_forward_sweep.py",
                    "--mode",
                    "high_fidelity",
                    "--config-path",
                    args.matrix_config,
                    "--seed",
                    str(seed),
                    "--n-steps",
                    str(args.n_steps),
                    "--output-csv",
                    str(matrix_seed_csv),
                    "--output-summary",
                    str(matrix_seed_summary),
                    "--output-manifest",
                    str(matrix_seed_manifest),
                ],
                env=env,
                dry_run=args.dry_run,
            )
        )
        if not args.dry_run:
            by_label = {row["label"]: row for row in load_csv_rows(matrix_seed_csv)}
            for label in matrix_labels:
                row = by_label[label]
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

    ci_checks: list[dict] = []
    if not args.skip_ci_parity:
        for name, command in (
            ("validation_quick", ["./scripts/run_validation_quick.sh"]),
            ("validation_full", ["./scripts/run_validation_full.sh"]),
        ):
            if Path(command[0]).exists():
                ci_checks.append(run_command(command, env=env, dry_run=args.dry_run))
            else:
                ci_checks.append(
                    {
                        "name": name,
                        "command": command,
                        "returncode": 0,
                        "passed": True,
                        "elapsed_s": 0.0,
                        "stdout_tail": "",
                        "stderr_tail": "",
                        "executed": False,
                        "status": "skipped_missing_script",
                    }
                )

    required_commands_passed = all(bool(c["passed"]) for c in command_records)

    if (not args.dry_run) and required_commands_passed:
        baseline_summary_payload = load_json(baseline_summary)
        hf_summary_payload = load_json(hf_summary)
        hf_rerun_summary_payload = load_json(hf_rerun_summary)
        repro_keys = [
            "stability_rate",
            "gi_mean",
            "gi_std",
            "gi_plausible_rate",
            "collision_force_share_mean",
        ]
        repro_mismatches = compare_summary_metrics(
            hf_summary_payload, hf_rerun_summary_payload, repro_keys, atol=args.atol
        )
        reproducibility = {
            "seed": args.seed,
            "atol": args.atol,
            "first_summary": str(hf_summary),
            "second_summary": str(hf_rerun_summary),
            "mismatch_count": len(repro_mismatches),
            "mismatches": repro_mismatches,
            "passed": len(repro_mismatches) == 0,
        }

        matrix_n = len(matrix_records)
        matrix_stable = sum(int(r["stable"]) for r in matrix_records)
        matrix = {
            "seeds": matrix_seeds,
            "labels": matrix_labels,
            "n_records": matrix_n,
            "stability_rate": matrix_stable / max(matrix_n, 1),
            "passed": matrix_n >= 6 and (matrix_stable / max(matrix_n, 1)) >= 0.95,
            "matrix_csv": str(matrix_csv),
        }
        if matrix_records:
            with matrix_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(matrix_records[0].keys()))
                writer.writeheader()
                writer.writerows(matrix_records)

        baseline_rows = load_csv_rows(baseline_csv)
        hf_rows = load_csv_rows(hf_csv)
        reg_checks = regression_checks(
            baseline_summary_payload,
            hf_summary_payload,
            baseline_rows,
            hf_rows,
        )
        regression = {
            "baseline_summary": str(baseline_summary),
            "high_fidelity_summary": str(hf_summary),
            "checks": reg_checks,
            "passed": all(bool(c["passed"]) for c in reg_checks),
        }

        baseline_runtime = sum_runtime_from_csv(baseline_csv)
        hf_runtime = sum_runtime_from_csv(hf_csv)
        runtime_budget = {
            "baseline_runtime_s": baseline_runtime,
            "high_fidelity_runtime_s": hf_runtime,
            "runtime_ratio_hf_over_baseline": hf_runtime / max(baseline_runtime, 1e-12),
            "budget_ratio": args.runtime_budget_ratio,
        }
        runtime_budget["passed"] = bool(
            runtime_budget["runtime_ratio_hf_over_baseline"] <= args.runtime_budget_ratio
        )

        ci_parity = {
            "checks": ci_checks,
            "passed": all(bool(c["passed"]) for c in ci_checks) if ci_checks else True,
        }
        runtime_stats = {
            "baseline_runtime_s": baseline_runtime,
            "high_fidelity_runtime_s": hf_runtime,
            "runtime_ratio_hf_over_baseline": runtime_budget["runtime_ratio_hf_over_baseline"],
            "command_elapsed_s_total": sum(float(c["elapsed_s"]) for c in command_records),
        }
        with runtime_stats_json.open("w") as f:
            json.dump(runtime_stats, f, indent=2)
    else:
        blocked_reason = "dry_run" if args.dry_run else "upstream_command_failure"
        reproducibility = {
            "seed": args.seed,
            "atol": args.atol,
            "mismatch_count": 0,
            "mismatches": [],
            "passed": False,
            "status": blocked_reason,
        }
        matrix = {
            "seeds": matrix_seeds,
            "labels": matrix_labels,
            "n_records": len(matrix_seeds) * len(matrix_labels),
            "stability_rate": 0.0,
            "passed": False,
            "matrix_csv": str(matrix_csv),
            "status": blocked_reason,
        }
        regression = {
            "checks": [],
            "passed": False,
            "status": blocked_reason,
        }
        runtime_budget = {
            "baseline_runtime_s": 0.0,
            "high_fidelity_runtime_s": 0.0,
            "runtime_ratio_hf_over_baseline": 0.0,
            "budget_ratio": args.runtime_budget_ratio,
            "passed": False,
            "status": blocked_reason,
        }
        ci_parity = {
            "checks": ci_checks,
            "passed": False if ci_checks else True,
            "status": blocked_reason,
        }

    config_hashes = {
        "baseline_config": config_hash(load_grid_config(args.baseline_config)),
        "high_fidelity_config": config_hash(load_grid_config(args.high_fidelity_config)),
        "matrix_config": config_hash(load_grid_config(args.matrix_config)),
    }
    with config_hashes_json.open("w") as f:
        json.dump(config_hashes, f, indent=2)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "n_steps": args.n_steps,
        "output_dir": str(out_dir),
        "artifacts": {
            "baseline_csv": str(baseline_csv),
            "baseline_summary": str(baseline_summary),
            "baseline_manifest": str(baseline_manifest),
            "high_fidelity_csv": str(hf_csv),
            "high_fidelity_summary": str(hf_summary),
            "high_fidelity_manifest": str(hf_manifest),
            "high_fidelity_rerun_summary": str(hf_rerun_summary),
            "runtime_stats": str(runtime_stats_json),
            "config_hashes": str(config_hashes_json),
            "matrix_csv": str(matrix_csv),
        },
        "reproducibility": reproducibility,
        "matrix": matrix,
        "regression_vs_baseline": regression,
        "runtime_budget": runtime_budget,
        "ci_parity": ci_parity,
        "commands": command_records,
    }
    report["passed"] = bool(
        reproducibility["passed"]
        and matrix["passed"]
        and regression["passed"]
        and runtime_budget["passed"]
        and ci_parity["passed"]
    )

    with report_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_json}")


if __name__ == "__main__":
    main()
