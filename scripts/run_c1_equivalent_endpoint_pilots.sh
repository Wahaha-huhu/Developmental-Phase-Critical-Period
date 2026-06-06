#!/usr/bin/env bash
set -euo pipefail

# Equivalent-endpoint C1 pilot runner.
# Generates endpoint configs, resolves token budgets with the existing planner,
# runs each resolved config, analyses it, and packages partial/completed outputs.

TEMPLATE=${TEMPLATE:-configs/c1_alignment_160m_pilot.yaml}
INJECT_STEP=${INJECT_STEP:-step1000}
ENDPOINTS=${ENDPOINTS:-"step2000 step3000 step4000"}
BUDGET_SCALE=${BUDGET_SCALE:-0.01}
BATCH_SIZE=${BATCH_SIZE:-64}
SEQ_LEN=${SEQ_LEN:-512}
GRAD_ACCUM=${GRAD_ACCUM:-16}
PYTHIA_TOKENS_PER_STEP=${PYTHIA_TOKENS_PER_STEP:-2097152}
SEEDS=${SEEDS:-"0"}

mkdir -p logs/c1_equiv results/c1_equiv_budget_audits

python scripts/plan_c1_equivalent_endpoint_pilots.py \
  --template "$TEMPLATE" \
  --inject-step "$INJECT_STEP" \
  --endpoints $ENDPOINTS \
  --budget-scale "$BUDGET_SCALE" \
  --batch-size "$BATCH_SIZE" \
  --sequence-length "$SEQ_LEN" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --pythia-tokens-per-step "$PYTHIA_TOKENS_PER_STEP" \
  --seeds $SEEDS

for cfg in configs/c1_alignment_160m_equiv_${INJECT_STEP}_to_step*_scale$(echo "$BUDGET_SCALE" | tr . p).yaml; do
  [ -f "$cfg" ] || continue
  name=$(basename "$cfg" .yaml)
  resolved="configs/${name}_resolved.yaml"
  audit="results/c1_equiv_budget_audits/${name}_budget_audit.csv"
  echo "=== Planning token budget for $cfg ==="
  python scripts/plan_c1_continuation_token_budgets.py \
    --config "$cfg" \
    --out-config "$resolved" \
    --out-csv "$audit" \
    --endpoint-step "${cfg##*to_step}"
  # Some planner versions cannot parse endpoint from filename; retry robustly in Python if needed is left to user.
  echo "=== Running $resolved ==="
  python scripts/run_c1_alignment_consolidation.py --config "$resolved" \
    2>&1 | tee "logs/c1_equiv/${name}.log"

  root=$(python - <<PY
import yaml
cfg=yaml.safe_load(open('$resolved'))
print(cfg.get('outputs',{}).get('root', cfg.get('output_root')))
PY
)
  echo "=== Analysing $root ==="
  python scripts/analyze_c1_alignment_consolidation.py --root "$root" \
    2>&1 | tee "logs/c1_equiv/${name}_analysis.log"
done

zip -r c1_equivalent_endpoint_pilots_for_review.zip \
  results/c1_alignment_160m_equiv_* \
  results/c1_equiv_budget_audits \
  configs/c1_alignment_160m_equiv_* \
  logs/c1_equiv || true
