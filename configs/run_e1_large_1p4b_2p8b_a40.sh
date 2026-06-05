#!/usr/bin/env bash
set -euo pipefail

# Run Pythia 1.4B and 2.8B E1 dense-early + sparse-late sweeps on an A40-style server.
# Usage from repo root:
#   bash scripts/run_e1_large_1p4b_2p8b_a40.sh
# Optional:
#   DELETE_CACHE=0 bash scripts/run_e1_large_1p4b_2p8b_a40.sh

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
DELETE_CACHE="${DELETE_CACHE:-1}"

mkdir -p logs results/e1_large_scale_backups "$HF_HUB_CACHE"

python scripts/generate_e1_large_scale_configs.py \
  --template configs/e1_phase_identification.yaml \
  --model EleutherAI/pythia-1.4b-deduped \
  --model EleutherAI/pythia-2.8b-deduped \
  --mode dense_early_sparse_late \
  --early-max-step 8000

run_one() {
  local label="$1"
  local repo="$2"
  local cfg="configs/e1_large_${label}_dense_early_sparse_late.yaml"
  local root="results/e1_large_scale/${label}_dense_early_sparse_late"
  local model_cache="$HF_HUB_CACHE/models--EleutherAI--pythia-${label/p/.}b-deduped"

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
}

run_one "1p4" "EleutherAI/pythia-1.4b-deduped"
run_one "2p8" "EleutherAI/pythia-2.8b-deduped"

zip -r e1_large_1p4b_2p8b_for_review.zip \
  results/e1_large_scale/1p4_boundary_analysis \
  results/e1_large_scale/2p8_boundary_analysis \
  results/e1_large_scale_backups/e1_large_1p4_dense_early_sparse_late.csv \
  results/e1_large_scale_backups/e1_large_2p8_dense_early_sparse_late.csv \
  configs/e1_large_1p4_dense_early_sparse_late.yaml \
  configs/e1_large_2p8_dense_early_sparse_late.yaml \
  logs/e1_large_1p4.log logs/e1_large_2p8.log \
  logs/e1_large_1p4_analysis.log logs/e1_large_2p8_analysis.log

echo "Done. Review package: e1_large_1p4b_2p8b_for_review.zip"
