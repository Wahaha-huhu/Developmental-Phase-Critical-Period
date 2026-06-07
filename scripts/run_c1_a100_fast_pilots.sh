#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs/c1
PYTHIA_TOKENS_PER_STEP=${PYTHIA_TOKENS_PER_STEP:-2097152}
RUN_SCALE0P005=${RUN_SCALE0P005:-0}
RUN_ATTACK=${RUN_ATTACK:-0}

get_root() {
  python -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c['outputs']['root'])" "$1"
}

run_one() {
  local cfg="$1"
  local endpoint="$2"
  local tag="$3"
  local resolved="${cfg%.yaml}_resolved.yaml"
  local root
  root=$(get_root "$cfg")
  mkdir -p "$root"
  echo "[run] planning budget for $cfg -> $resolved"
  python scripts/plan_c1_continuation_token_budgets.py \
    --config "$cfg" \
    --out-config "$resolved" \
    --out-csv "$root/continuation_budget_audit.csv" \
    --endpoint-step "$endpoint" \
    --pythia-tokens-per-step "$PYTHIA_TOKENS_PER_STEP"
  echo "[run] budget audit"
  cat "$root/continuation_budget_audit.csv" || true
  echo "[run] launching $tag"
  PYTHONUNBUFFERED=1 python -u scripts/run_c1_alignment_consolidation.py \
    --config "$resolved" \
    2>&1 | tee "logs/c1/${tag}.log"
  echo "[run] analyzing $tag"
  python scripts/analyze_c1_alignment_consolidation.py --root "$root" || true
}

run_one configs/c1_alignment_160m_a100_fast_scale0p001.yaml 2000 c1_a100_fast_scale0p001

if [[ "$RUN_SCALE0P005" == "1" ]]; then
  run_one configs/c1_alignment_160m_a100_fast_scale0p005.yaml 2000 c1_a100_fast_scale0p005
fi

if [[ "$RUN_ATTACK" == "1" ]]; then
  run_one configs/c1_alignment_160m_a100_fast_attack_only.yaml 2000 c1_a100_fast_attack_only
fi

zip -r c1_a100_fast_pilots_for_review.zip \
  results/c1_alignment_160m_a100_fast_* \
  configs/c1_alignment_160m_a100_fast_*.yaml \
  logs/c1 || true
