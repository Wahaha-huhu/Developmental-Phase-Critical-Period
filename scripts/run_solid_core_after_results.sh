#!/usr/bin/env bash
set -euo pipefail

# Solid-core helper. This script does NOT assume a specific E3 runner; set
# E3_RUNNER to your working factual-cell runner if you want it to execute the
# sweep. Otherwise it just generates the config and analysis commands.

TEMPLATE=${TEMPLATE:-configs/e3_critical_period_intervention_final_v4.yaml}
CONFIG=${CONFIG:-configs/step1_factual_dense_160m_32ckpt.yaml}
OUTROOT=${OUTROOT:-results/step1_factual_dense_160m_32ckpt}
E3_RUNNER=${E3_RUNNER:-}

mkdir -p logs/solid_core

python scripts/generate_step1_dense_factual_configs.py \
  --template "$TEMPLATE" \
  --out "$CONFIG" \
  --output-root "$OUTROOT"

if [[ -n "$E3_RUNNER" ]]; then
  echo "Running Step 1 dense factual sweep with $E3_RUNNER"
  PYTHONUNBUFFERED=1 python -u "$E3_RUNNER" --config "$CONFIG" \
    2>&1 | tee logs/solid_core/step1_dense_factual_sweep.log
else
  echo "E3_RUNNER not set. Generated config only: $CONFIG"
  echo "Example: E3_RUNNER=scripts/run_e3_factual_cell_v3.py bash scripts/run_solid_core_after_results.sh"
fi

cat <<EOF

After the sweep finishes, run analysis with the actual summary CSV/root, e.g.:

python scripts/analyze_step1_dense_durability.py \
  --inputs $OUTROOT \
  --out results/step1_dense_durability_analysis

Then build artifact overlay using dense E1 and Step-1 summary, e.g.:

python scripts/plot_step0_artifact_overlay.py \
  --e1-metrics results/e1_dense_160m_all_checkpoints/raw/e1_spectral_metrics.csv \
  --durability results/step1_dense_durability_analysis/tables/step1_stage_summary.csv \
  --out results/step0_artifact_overlay
EOF
