"""One-command regeneration for high-fidelity simulation and render artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic runs.")
    parser.add_argument("--n-steps", type=int, default=120, help="Steps for sweep/validation.")
    parser.add_argument("--render-steps", type=int, default=180, help="Steps for render pipeline.")
    parser.add_argument(
        "--output-json",
        default="results/high_fidelity/package_summary.json",
        help="Output orchestration summary JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit planned commands only.",
    )
    return parser.parse_args()


def run_step(cmd: list[str], env: dict[str, str], dry_run: bool) -> dict:
    t0 = time.perf_counter()
    if dry_run:
        return {"command": cmd, "elapsed_s": 0.0, "returncode": 0, "passed": True, "executed": False}
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    return {
        "command": cmd,
        "elapsed_s": time.perf_counter() - t0,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "executed": True,
    }


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    steps = [
        (
            "high_fidelity_sweep",
            [
                "python3.11",
                "scripts/run_forward_sweep.py",
                "--mode",
                "high_fidelity",
                "--config-path",
                "configs/high_fidelity_forward_sweep.json",
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.n_steps),
                "--output-csv",
                "results/high_fidelity/forward_sweep.csv",
                "--output-summary",
                "results/high_fidelity/forward_sweep_summary.json",
                "--output-manifest",
                "results/high_fidelity/forward_sweep_manifest.json",
            ],
        ),
        (
            "publication_render",
            [
                "python3.11",
                "scripts/generate_high_fidelity_publication_render.py",
                "--config-path",
                "configs/high_fidelity_publication_render.json",
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.render_steps),
                "--with-metric-overlays",
                "--output-gif",
                "docs/assets/high_fidelity/publication_comparison.gif",
                "--output-mp4",
                "docs/assets/high_fidelity/publication_comparison.mp4",
                "--output-summary",
                "results/high_fidelity/publication_render_summary.json",
                "--output-manifest",
                "results/high_fidelity/publication_render_manifest.json",
            ],
        ),
        (
            "hardened_validation",
            [
                "python3.11",
                "scripts/validate_high_fidelity.py",
                "--seed",
                str(args.seed),
                "--n-steps",
                str(args.n_steps),
                "--output-dir",
                "results/high_fidelity",
            ],
        ),
    ]

    records = []
    for step_name, cmd in steps:
        print(f"[high_fidelity] {step_name}: {' '.join(cmd)}")
        rec = run_step(cmd, env=env, dry_run=args.dry_run)
        rec["name"] = step_name
        records.append(rec)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_steps": args.n_steps,
        "render_steps": args.render_steps,
        "dry_run": args.dry_run,
        "steps": records,
    }
    report["passed"] = bool(all(bool(step["passed"]) for step in records))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
