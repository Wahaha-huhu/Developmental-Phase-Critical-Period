#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/c1_alignment_160m_4090_step1000_to_step2000_scale0p001.yaml"
  "configs/c1_alignment_160m_4090_step1000_to_step2000_scale0p005.yaml"
  "configs/c1_alignment_160m_4090_step1000_to_step2000_scale0p01.yaml"
)

mkdir -p logs/c1
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

for CFG in "${CONFIGS[@]}"; do
  ROOT=$(python - <<PY
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path("$CFG").read_text())
print(cfg["outputs"]["root"])
PY
)
  ENDPOINT=$(python - <<PY
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path("$CFG").read_text())
print(str(cfg.get("endpoint_step", "step2000")).replace("step", ""))
PY
)
  NAME=$(basename "$CFG" .yaml)
  RESOLVED="configs/${NAME}_resolved.yaml"
  AUDIT="${ROOT}/continuation_budget_audit.csv"
  mkdir -p "$ROOT"
  echo "=== Planning $CFG ==="
  python scripts/plan_c1_continuation_token_budgets.py \
    --config "$CFG" \
    --out-config "$RESOLVED" \
    --out-csv "$AUDIT" \
    --endpoint-step "$ENDPOINT" \
    --pythia-tokens-per-step 2097152
  echo "=== Budget audit ==="
  cat "$AUDIT"
  echo "=== Running $RESOLVED ==="
  PYTHONUNBUFFERED=1 python -u scripts/run_c1_alignment_consolidation.py \
    --config "$RESOLVED" \
    2>&1 | tee "logs/c1/${NAME}.log"
  echo "=== Analyzing $ROOT ==="
  python scripts/analyze_c1_alignment_consolidation.py --root "$ROOT" || true
  echo "=== Completed $ROOT ==="
  echo
  echo "Intermediate zip: c1_4090_${NAME}_for_review.zip"
  zip -r "c1_4090_${NAME}_for_review.zip" "$ROOT" "$CFG" "$RESOLVED" "logs/c1/${NAME}.log" || true
  echo
  echo "Proceeding to next config..."
  echo
  sleep 3
done

zip -r c1_4090_continuation_pilots_for_review.zip \
  results/c1_alignment_160m_4090_step1000_to_step2000_scale0p001 \
  results/c1_alignment_160m_4090_step1000_to_step2000_scale0p005 \
  results/c1_alignment_160m_4090_step1000_to_step2000_scale0p01 \
  configs/c1_alignment_160m_4090_step1000_to_step2000_scale0p*.yaml \
  logs/c1 || true
