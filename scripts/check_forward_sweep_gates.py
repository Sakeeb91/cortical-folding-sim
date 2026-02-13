"""Validate forward sweep outputs against quality gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    gate_config = load_json(args.gate_config)
    print(
        f"Loaded rows={len(rows)} summary_keys={len(summary)} "
        f"gate_keys={len(gate_config)}"
    )


if __name__ == "__main__":
    main()
