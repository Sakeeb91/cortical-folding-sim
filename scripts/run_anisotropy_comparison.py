"""Run isotropic vs anisotropic comparison and summarize morphology deltas."""

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
        default="configs/week3_anisotropy_ab.json",
        help="A/B config JSON path.",
    )
    parser.add_argument(
        "--output-json",
        default="results/week3_anisotropy_comparison.json",
        help="Comparison summary output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/week3_anisotropy_ab.csv",
        help="Sweep CSV output path for A/B runs.",
    )
    parser.add_argument("--n-steps", type=int, default=140, help="Simulation steps.")
    return parser.parse_args()


def load_csv_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    cmd = [
        "python3.11",
        "scripts/run_forward_sweep.py",
        "--config-path",
        args.config_path,
        "--n-steps",
        str(args.n_steps),
        "--output-csv",
        args.output_csv,
        "--output-summary",
        "results/week3_anisotropy_ab_summary.json",
        "--output-manifest",
        "results/week3_anisotropy_ab_manifest.json",
    ]
    subprocess.check_call(cmd)

    rows = load_csv_rows(args.output_csv)
    row_by_label = {r["label"]: r for r in rows}
    base = row_by_label["isotropic_baseline"]
    aniso = row_by_label["anisotropic_z_bias"]

    comparison = {
        "n_runs": len(rows),
        "baseline_label": "isotropic_baseline",
        "anisotropic_label": "anisotropic_z_bias",
        "baseline_stable": int(base["stable"]),
        "anisotropic_stable": int(aniso["stable"]),
        "baseline_gi": to_float(base, "gi"),
        "anisotropic_gi": to_float(aniso, "gi"),
        "delta_gi": to_float(aniso, "gi") - to_float(base, "gi"),
        "baseline_curv_p90": to_float(base, "mean_curv_p90"),
        "anisotropic_curv_p90": to_float(aniso, "mean_curv_p90"),
        "delta_curv_p90": to_float(aniso, "mean_curv_p90") - to_float(base, "mean_curv_p90"),
        "baseline_disp_p95": to_float(base, "disp_p95"),
        "anisotropic_disp_p95": to_float(aniso, "disp_p95"),
        "delta_disp_p95": to_float(aniso, "disp_p95") - to_float(base, "disp_p95"),
        "stability_rate": (int(base["stable"]) + int(aniso["stable"])) / 2.0,
        "morphology_difference_detected": abs(
            to_float(aniso, "mean_curv_p90") - to_float(base, "mean_curv_p90")
        ) > 1e-4,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
