"""Validate forward sweep outputs against quality gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from cortical_folding.validation import (
    build_gate_report,
    compute_gate_metrics,
    evaluate_gate_checks,
    load_gate_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="results/forward_sweep.csv",
        help="CSV file produced by run_forward_sweep.py",
    )
    parser.add_argument(
        "--input-summary",
        default="results/forward_sweep_summary.json",
        help="Summary JSON produced by run_forward_sweep.py",
    )
    parser.add_argument(
        "--gate-config",
        default="configs/validation_gates_default.json",
        help="Validation gate threshold config JSON.",
    )
    parser.add_argument(
        "--output-report",
        default="results/validation_gate_report.json",
        help="Path to write gate report JSON.",
    )
    parser.add_argument(
        "--fail-on-failure",
        action="store_true",
        help="Return non-zero exit code if any gate fails.",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: str) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    summary = load_json(args.input_summary)
    thresholds = load_gate_thresholds(args.gate_config)
    metrics = compute_gate_metrics(rows, summary)
    checks = evaluate_gate_checks(metrics, thresholds)
    report = build_gate_report(
        rows=rows,
        summary=summary,
        thresholds=thresholds,
        checks=checks,
        metadata={
            "input_csv": args.input_csv,
            "input_summary": args.input_summary,
            "gate_config": args.gate_config,
        },
    )
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {output_path}")
    for check in report["checks"]:
        print(check["message"])
    if report["passed"]:
        print("All validation gates passed.")
    else:
        print(
            "Validation gates failed. Review report and adjust physics "
            "params, thresholds, or model stability."
        )
        if args.fail_on_failure:
            sys.exit(1)


if __name__ == "__main__":
    main()
