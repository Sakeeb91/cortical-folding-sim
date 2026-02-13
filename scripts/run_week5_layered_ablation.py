"""Run Week 5 layered approximation ablation and summarize acceptance metrics."""

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
        default="configs/week5_layered_ablation.json",
        help="Week 5 ablation config JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/week5_layered_ablation.csv",
        help="Sweep CSV output path.",
    )
    parser.add_argument(
        "--output-summary",
        default="results/week5_layered_ablation_summary.json",
        help="Sweep summary output path.",
    )
    parser.add_argument(
        "--output-manifest",
        default="results/week5_layered_ablation_manifest.json",
        help="Sweep manifest output path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week5_layered_comparison.json",
        help="Week 5 comparison summary output path.",
    )
    parser.add_argument("--n-steps", type=int, default=150, help="Simulation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed metadata.")
    parser.add_argument(
        "--runtime-budget-ratio",
        type=float,
        default=1.25,
        help="Maximum allowed runtime ratio (layered_on / layered_off).",
    )
    parser.add_argument(
        "--robust-max-outside-skull-frac",
        type=float,
        default=0.34,
        help="Maximum outside-skull fraction for robust region eligibility.",
    )
    parser.add_argument(
        "--robust-max-disp-p95",
        type=float,
        default=0.80,
        help="Maximum displacement p95 for robust region eligibility.",
    )
    parser.add_argument(
        "--robust-min-gi",
        type=float,
        default=2.0,
        help="Minimum GI for robust region eligibility.",
    )
    return parser.parse_args()


def load_csv_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def to_int(row: dict, key: str) -> int:
    return int(float(row[key]))


def robust_region_labels(
    rows: list[dict],
    *,
    max_outside_skull_frac: float,
    max_disp_p95: float,
    min_gi: float,
) -> list[str]:
    labels: list[str] = []
    for row in rows:
        if to_int(row, "stable") != 1:
            continue
        if to_int(row, "gi_plausible") != 1:
            continue
        if to_float(row, "outside_skull_frac") > max_outside_skull_frac:
            continue
        if to_float(row, "disp_p95") > max_disp_p95:
            continue
        if to_float(row, "gi") < min_gi:
            continue
        labels.append(row["label"])
    return labels


def summarize_group(rows: list[dict], labels: set[str]) -> dict:
    subset = [r for r in rows if r["label"] in labels]
    if not subset:
        return {
            "n_runs": 0,
            "stable_rate": 0.0,
            "best_label": "none",
            "best_gi": float("nan"),
            "worst_disp_p95": float("nan"),
        }
    stable_rows = [r for r in subset if to_int(r, "stable") == 1]
    best = max(stable_rows or subset, key=lambda r: to_float(r, "gi"))
    return {
        "n_runs": len(subset),
        "stable_rate": len(stable_rows) / len(subset),
        "best_label": best["label"],
        "best_gi": to_float(best, "gi"),
        "worst_disp_p95": max(to_float(r, "disp_p95") for r in subset),
    }


def main() -> None:
    args = parse_args()
    cmd = [
        "python3.11",
        "scripts/run_forward_sweep.py",
        "--config-path",
        args.config_path,
        "--n-steps",
        str(args.n_steps),
        "--seed",
        str(args.seed),
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

    baseline = row_by_label["layered_off_reference"]
    layered = row_by_label["layered_on_reference"]
    layered_rows = [r for r in rows if to_int(r, "enable_two_layer_approx") == 1]

    robust_labels = robust_region_labels(
        layered_rows,
        max_outside_skull_frac=args.robust_max_outside_skull_frac,
        max_disp_p95=args.robust_max_disp_p95,
        min_gi=args.robust_min_gi,
    )

    failure_reason_counts: dict[str, int] = {}
    for row in rows:
        reason = row["fail_reason"]
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

    layered_runtime_ratio = to_float(layered, "runtime_s") / max(to_float(baseline, "runtime_s"), 1e-12)
    reduced_grid_stable = all(to_int(r, "stable") == 1 for r in rows)

    payload = {
        "n_runs": len(rows),
        "seed": args.seed,
        "config_path": args.config_path,
        "baseline_label": "layered_off_reference",
        "layered_label": "layered_on_reference",
        "n_layered_runs": len(layered_rows),
        "layered_stability_rate": (
            sum(to_int(r, "stable") for r in layered_rows) / max(len(layered_rows), 1)
        ),
        "reduced_grid_all_stable": reduced_grid_stable,
        "baseline_gi": to_float(baseline, "gi"),
        "layered_gi": to_float(layered, "gi"),
        "delta_gi_layered_minus_baseline": to_float(layered, "gi") - to_float(baseline, "gi"),
        "baseline_outside_skull_frac": to_float(baseline, "outside_skull_frac"),
        "layered_outside_skull_frac": to_float(layered, "outside_skull_frac"),
        "delta_outside_skull_frac_layered_minus_baseline": (
            to_float(layered, "outside_skull_frac") - to_float(baseline, "outside_skull_frac")
        ),
        "baseline_disp_p95": to_float(baseline, "disp_p95"),
        "layered_disp_p95": to_float(layered, "disp_p95"),
        "delta_disp_p95_layered_minus_baseline": (
            to_float(layered, "disp_p95") - to_float(baseline, "disp_p95")
        ),
        "baseline_runtime_s": to_float(baseline, "runtime_s"),
        "layered_runtime_s": to_float(layered, "runtime_s"),
        "runtime_ratio_layered_over_baseline": layered_runtime_ratio,
        "runtime_budget_ratio": args.runtime_budget_ratio,
        "robust_region_definition": {
            "stable": 1,
            "gi_plausible": 1,
            "max_outside_skull_frac": args.robust_max_outside_skull_frac,
            "max_disp_p95": args.robust_max_disp_p95,
            "min_gi": args.robust_min_gi,
        },
        "robust_region_count": len(robust_labels),
        "robust_region_labels": robust_labels,
        "failure_reason_counts": failure_reason_counts,
        "clipping_ablation": summarize_group(
            rows, labels={"layered_clip_tight", "layered_clip_relaxed"}
        ),
        "damping_ablation": summarize_group(
            rows, labels={"layered_damping_low", "layered_damping_high"}
        ),
        "dt_ablation": summarize_group(
            rows, labels={"layered_dt_small", "layered_dt_large"}
        ),
    }

    payload["acceptance_layered_mode_stable_on_reduced_grid"] = bool(
        payload["reduced_grid_all_stable"]
    )
    payload["acceptance_robust_parameter_region_found"] = bool(
        payload["robust_region_count"] >= 1
    )
    payload["acceptance_runtime_within_budget"] = bool(
        payload["runtime_ratio_layered_over_baseline"] <= args.runtime_budget_ratio
    )
    payload["acceptance_week5_passed"] = bool(
        payload["acceptance_layered_mode_stable_on_reduced_grid"]
        and payload["acceptance_robust_parameter_region_found"]
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
