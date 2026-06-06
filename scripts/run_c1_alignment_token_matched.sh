#!/usr/bin/env bash
set -euo pipefail

CONFIG_IN=${CONFIG_IN:-configs/c1_alignment_160m_pilot.yaml}
CONFIG_OUT=${CONFIG_OUT:-configs/c1_alignment_160m_pilot_token_matched_resolved.yaml}
BUDGET_CSV=${BUDGET_CSV:-results/c1_alignment_160m_pilot_token_matched/continuation_budget_audit.csv}
ROOT=${ROOT:-results/c1_alignment_160m_pilot_token_matched}
ENDPOINT_STEP=${ENDPOINT_STEP:-8000}
PYTHIA_TOKENS_PER_STEP=${PYTHIA_TOKENS_PER_STEP:-2097152}

mkdir -p "$(dirname "$BUDGET_CSV")" logs/c1

python scripts/plan_c1_continuation_token_budgets.py \
  --config "$CONFIG_IN" \
  --out-config "$CONFIG_OUT" \
  --out-csv "$BUDGET_CSV" \
  --endpoint-step "$ENDPOINT_STEP" \
  --pythia-tokens-per-step "$PYTHIA_TOKENS_PER_STEP"

# Ensure output root is distinct if the runner reads outputs.root.
python - <<PY
from pathlib import Path
import yaml
p = Path("$CONFIG_OUT")
cfg = yaml.safe_load(p.read_text())
cfg.setdefault("outputs", {})["root"] = "$ROOT"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Resolved run root:", "$ROOT")
PY

python scripts/run_c1_alignment_consolidation.py \
  --config "$CONFIG_OUT" \
  2>&1 | tee logs/c1/$(basename "$ROOT")_run.log

python scripts/analyze_c1_alignment_consolidation.py \
  --root "$ROOT" \
  2>&1 | tee logs/c1/$(basename "$ROOT")_analysis.log
