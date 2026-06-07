#!/usr/bin/env bash
set -euo pipefail

INPUTS=${INPUTS:-results/step1_factual_dense_160m_32ckpt}
OUT=${OUT:-solid_results/step1_dense_durability_analysis}

mkdir -p solid_results/logs
python solid_script/analyze_step1_dense_durability.py \
  --inputs "$INPUTS" \
  --out "$OUT" \
  2>&1 | tee solid_results/logs/step1_dense_analysis.log
