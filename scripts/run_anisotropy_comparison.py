"""Run isotropic vs anisotropic comparison and summarize morphology deltas."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default="configs/week3_anisotropy_ab.json",
        help="A/B config JSON path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week3_anisotropy_comparison.json",
        help="Comparison summary output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Week 3 anisotropy comparison scaffold. config={args.config_path}")


if __name__ == "__main__":
    main()
