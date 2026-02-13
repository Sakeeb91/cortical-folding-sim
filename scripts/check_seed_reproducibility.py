"""Check reproducibility between two seeded benchmark summary files."""

from __future__ import annotations

import argparse
import sys

from cortical_folding.reproducibility import compare_summary_metrics, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="First summary JSON path.")
    parser.add_argument("--second", required=True, help="Second summary JSON path.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = load_json(args.first)
    second = load_json(args.second)
    keys = [
        "stability_rate",
        "gi_mean",
        "gi_std",
        "gi_plausible_rate",
    ]
    mismatches = compare_summary_metrics(first, second, keys=keys, atol=args.atol)
    if mismatches:
        print("Seeded reproducibility check failed:")
        for m in mismatches:
            print(m)
        sys.exit(1)
    print("Seeded reproducibility check passed.")


if __name__ == "__main__":
    main()
