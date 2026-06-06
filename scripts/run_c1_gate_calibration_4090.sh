#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/c1_gate_calibration_160m_step1000_4090_fast.yaml}
LOG_DIR=${LOG_DIR:-logs/c1}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/c1_gate_calibration_4090_${STAMP}.log"

export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export PYTHONUNBUFFERED=1

echo "[run] config=$CONFIG"
echo "[run] log=$LOG_FILE"

python -u scripts/run_c1_gate_calibration_4090.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"

echo "[run] finished"
echo "[run] package command:"
echo "zip -r c1_gate_calibration_4090_for_review.zip results/c1_gate_calibration_160m_step1000_4090* configs/c1_gate_calibration_160m_step1000_4090*.yaml logs/c1 scripts/run_c1_gate_calibration_4090.py"
