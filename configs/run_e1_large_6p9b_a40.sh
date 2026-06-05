#!/usr/bin/env bash
set -euo pipefail

# Run Pythia 6.9B E1 dense-early + sparse-late cheap-metric sweep.
# Usage from repo root:
#   bash scripts/run_e1_large_6p9b_a40.sh
# Optional:
#   DELETE_CACHE=0 bash scripts/run_e1_large_6p9b_a40.sh

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
DELETE_CACHE="${DELETE_CACHE:-1}"

mkdir -p logs results/e1_large_scale_backups "$HF_HUB_CACHE"

python scripts/generate_e1_large_scale_configs.py \
  --template configs/e1_phase_identification.yaml \
  --model EleutherAI/pythia-6.9b-deduped \
  --mode dense_early_sparse_late \
  --early-max-step 8000

label="6p9"
repo="EleutherAI/pythia-6.9b-deduped"
cfg="configs/e1_large_${label}_dense_early_sparse_late.yaml"
root="results/e1_large_scale/${label}_dense_early_sparse_late"
model_cache="$HF_HUB_CACHE/models--EleutherAI--pythia-6.9b-deduped"

echo "=============================="
echo "Running $repo"
echo "Config: $cfg"
echo "Output: $root"
echo "HF cache: $HF_HUB_CACHE"
echo "=============================="

python scripts/run_e1_collect_spectra.py --config "$cfg" 2>&1 | tee "logs/e1_large_${label}.log"

mkdir -p results/e1_large_scale_backups
cp "$root/raw/e1_spectral_metrics.csv" "results/e1_large_scale_backups/e1_large_${label}_dense_early_sparse_late.csv"

python scripts/analyze_e1_dense_boundary.py \
  --metrics "$root/raw/e1_spectral_metrics.csv" \
  --out "results/e1_large_scale/${label}_boundary_analysis" \
  --model "$repo" 2>&1 | tee "logs/e1_large_${label}_analysis.log"

if [[ "$DELETE_CACHE" == "1" ]]; then
  echo "Deleting cache for $repo"
  rm -rf "$model_cache" || true
  find "$HF_HUB_CACHE" -type d -name "*.incomplete" -exec rm -rf {} + || true
  du -sh "$HF_HOME" || true
else
  echo "Keeping cache for $repo because DELETE_CACHE=$DELETE_CACHE"
fi

zip -r e1_large_6p9b_for_review.zip \
  results/e1_large_scale/${label}_boundary_analysis \
  results/e1_large_scale_backups/e1_large_${label}_dense_early_sparse_late.csv \
  configs/e1_large_${label}_dense_early_sparse_late.yaml \
  logs/e1_large_${label}.log logs/e1_large_${label}_analysis.log

echo "Done. Review package: e1_large_6p9b_for_review.zip"
