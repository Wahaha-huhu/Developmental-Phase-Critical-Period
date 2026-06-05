#!/usr/bin/env bash
set -euo pipefail

# Run Pythia 12B E1 sparse-key sweep. This is intentionally sparse by default
# because 12B is the riskiest model on a 48GB A40 + 200GB disk.
# Usage from repo root:
#   bash scripts/run_e1_large_12b_a40_sparse.sh
# Optional dense early mode, only if you are comfortable with time/storage risk:
#   MODE=dense_early_sparse_late bash scripts/run_e1_large_12b_a40_sparse.sh
# Optional keep cache:
#   DELETE_CACHE=0 bash scripts/run_e1_large_12b_a40_sparse.sh

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
DELETE_CACHE="${DELETE_CACHE:-1}"
MODE="${MODE:-sparse_key}"

mkdir -p logs results/e1_large_scale_backups "$HF_HUB_CACHE"

python scripts/generate_e1_large_scale_configs.py \
  --template configs/e1_phase_identification.yaml \
  --model EleutherAI/pythia-12b-deduped \
  --mode "$MODE" \
  --early-max-step 8000

label="12b"
repo="EleutherAI/pythia-12b-deduped"
cfg="configs/e1_large_${label}_${MODE}.yaml"
root="results/e1_large_scale/${label}_${MODE}"
model_cache="$HF_HUB_CACHE/models--EleutherAI--pythia-12b-deduped"

echo "=============================="
echo "Running $repo"
echo "Mode: $MODE"
echo "Config: $cfg"
echo "Output: $root"
echo "HF cache: $HF_HUB_CACHE"
echo "=============================="

echo "12B warning: if this OOMs, stop and do not treat it as blocking."
python scripts/run_e1_collect_spectra.py --config "$cfg" 2>&1 | tee "logs/e1_large_${label}_${MODE}.log"

mkdir -p results/e1_large_scale_backups
cp "$root/raw/e1_spectral_metrics.csv" "results/e1_large_scale_backups/e1_large_${label}_${MODE}.csv"

python scripts/analyze_e1_dense_boundary.py \
  --metrics "$root/raw/e1_spectral_metrics.csv" \
  --out "results/e1_large_scale/${label}_${MODE}_boundary_analysis" \
  --model "$repo" 2>&1 | tee "logs/e1_large_${label}_${MODE}_analysis.log"

if [[ "$DELETE_CACHE" == "1" ]]; then
  echo "Deleting cache for $repo"
  rm -rf "$model_cache" || true
  find "$HF_HUB_CACHE" -type d -name "*.incomplete" -exec rm -rf {} + || true
  du -sh "$HF_HOME" || true
else
  echo "Keeping cache for $repo because DELETE_CACHE=$DELETE_CACHE"
fi

zip -r "e1_large_12b_${MODE}_for_review.zip" \
  "results/e1_large_scale/${label}_${MODE}_boundary_analysis" \
  "results/e1_large_scale_backups/e1_large_${label}_${MODE}.csv" \
  "configs/e1_large_${label}_${MODE}.yaml" \
  "logs/e1_large_${label}_${MODE}.log" "logs/e1_large_${label}_${MODE}_analysis.log"

echo "Done. Review package: e1_large_12b_${MODE}_for_review.zip"
