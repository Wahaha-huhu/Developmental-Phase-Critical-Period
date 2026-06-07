#!/usr/bin/env bash
set -euo pipefail

mkdir -p solid_results/logs solid_results/configs

MODE="${MODE:-early_sparse_late}"
TEMPLATE="${TEMPLATE:-configs/e1_phase_identification.yaml}"
CONFIG="${CONFIG:-solid_results/configs/e1_dense_1b_${MODE}.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-solid_results/e1_dense_1b_${MODE}}"
MODEL="${MODEL:-EleutherAI/pythia-1b-deduped}"
DROP_STEP0="${DROP_STEP0:-0}"
CHEAP_METRICS="${CHEAP_METRICS:-0}"
RUN_COLLECT="${RUN_COLLECT:-0}"
CSV="${CSV:-${OUTPUT_ROOT}/raw/e1_spectral_metrics.csv}"
PLOT_OUT="${PLOT_OUT:-solid_results/e1_dense_indicator_panels_1b}"

ARGS=(
  --template "$TEMPLATE"
  --out "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --model "$MODEL"
  --mode "$MODE"
)
if [[ "$DROP_STEP0" == "1" ]]; then
  ARGS+=(--drop-step0)
fi
if [[ "$CHEAP_METRICS" == "1" ]]; then
  ARGS+=(--cheap-metrics)
fi

python solid_script/generate_e1_1b_solid_config.py "${ARGS[@]}"

if [[ "$RUN_COLLECT" == "1" ]]; then
  PYTHONUNBUFFERED=1 python -u scripts/run_e1_collect_spectra.py \
    --config "$CONFIG" \
    2>&1 | tee "solid_results/logs/e1_dense_1b_${MODE}_collect.log"
fi

if [[ ! -f "$CSV" ]]; then
  echo "Metrics CSV not found: $CSV" >&2
  echo "Set CSV=/path/to/e1_spectral_metrics.csv if the run output is elsewhere." >&2
  exit 2
fi

python solid_script/plot_solid_e1_dense_indicator_panels.py \
  --e1-metrics "$CSV" \
  --out "$PLOT_OUT" \
  --model-tag 1b \
  --exclude-step0

echo "Done. Plots in $PLOT_OUT"
