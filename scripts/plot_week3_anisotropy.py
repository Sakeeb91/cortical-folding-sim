"""Plot Week 3 anisotropy comparison metrics."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default="results/week3_anisotropy_comparison.json",
        help="Comparison JSON path.",
    )
    parser.add_argument(
        "--output-png",
        default="docs/assets/week3_anisotropy_delta.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Week 3 plotting scaffold: {args.input_json} -> {args.output_png}")


if __name__ == "__main__":
    main()
