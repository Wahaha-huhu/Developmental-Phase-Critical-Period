#!/usr/bin/env bash
set -euo pipefail

mkdir -p solid_results/logs solid_results/configs

E1_160M_CSV=${E1_160M_CSV:-results/e1_dense_160m_all_checkpoints/raw/e1_spectral_metrics.csv}
E1_1P4B_CSV=${E1_1P4B_CSV:-solid_results/e1_dense_1p4b_early_sparse_late/raw/e1_spectral_metrics.csv}

if [[ -f "$E1_160M_CSV" ]]; then
  echo "[plot] 160M: $E1_160M_CSV"
  python scripts/plot_solid_e1_dense_indicator_panels.py \
    --e1-metrics "$E1_160M_CSV" \
    --out solid_results/e1_dense_indicator_panels_160m \
    --model-tag 160m \
    --exclude-step0 \
    2>&1 | tee solid_results/logs/plot_e1_160m.log
else
  echo "[skip] missing 160M CSV: $E1_160M_CSV"
fi

if [[ -f "$E1_1P4B_CSV" ]]; then
  echo "[plot] 1.4B: $E1_1P4B_CSV"
  python scripts/plot_solid_e1_dense_indicator_panels.py \
    --e1-metrics "$E1_1P4B_CSV" \
    --out solid_results/e1_dense_indicator_panels_1p4b \
    --model-tag 1p4b \
    --exclude-step0 \
    2>&1 | tee solid_results/logs/plot_e1_1p4b.log
else
  echo "[skip] missing 1.4B CSV: $E1_1P4B_CSV"
  echo "Set E1_1P4B_CSV=/path/to/e1_spectral_metrics.csv after running/locating the 1.4B E1 sweep."
fi
