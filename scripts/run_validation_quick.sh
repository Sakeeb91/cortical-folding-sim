#!/usr/bin/env bash
set -euo pipefail

python3.11 -m pytest tests -q
MPLBACKEND=Agg python3.11 scripts/run_forward_sweep.py \
  --quick --n-steps 80 --max-runs 4 \
  --gi-plausible-min 0.45 \
  --gi-plausible-max 1.20
python3.11 scripts/check_forward_sweep_gates.py \
  --input-csv results/forward_sweep.csv \
  --input-summary results/forward_sweep_summary.json \
  --gate-config configs/validation_gates_default.json \
  --output-report results/validation_gate_report.json \
  --fail-on-failure
