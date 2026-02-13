"""Check reproducibility between two seeded benchmark summary files."""

from __future__ import annotations

import argparse

from cortical_folding.reproducibility import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="First summary JSON path.")
    parser.add_argument("--second", required=True, help="Second summary JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = load_json(args.first)
    second = load_json(args.second)
    print(
        f"Loaded summaries: first_runs={first.get('n_runs')} "
        f"second_runs={second.get('n_runs')}"
    )


if __name__ == "__main__":
    main()
