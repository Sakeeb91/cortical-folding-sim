#!/usr/bin/env bash
set -euo pipefail

python3.11 -m pytest tests -q
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py --n-steps 120
python3.11 scripts/check_forward_sweep_gates.py \
  --input-csv results/forward_sweep.csv \
  --input-summary results/forward_sweep_summary.json \
  --gate-config configs/validation_gates_default.json \
  --output-report results/validation_gate_report.json \
  --fail-on-failure

# Seeded reproducibility sanity check (repeat same run + compare summaries)
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py --n-steps 120 --output-summary results/forward_sweep_summary_seed_repeat.json
python3.11 scripts/check_seed_reproducibility.py \
  --first results/forward_sweep_summary.json \
  --second results/forward_sweep_summary_seed_repeat.json
