# Solid-core implementation runbook

This patch implements the revised thesis structure's solid path:

1. Step 0 — engineering-artifact overlay.
2. Step 1 — dense short-horizon factual durability sweep.
3. Step 1 analysis — uptake/retention/degradation curves, window-vs-late bootstrap, break-vs-monotone test.
4. Step 2 mechanism linkage — use the Step 0 overlay and existing dense E1 results as the geometry concordance base.

C1 alignment continue-to-consolidation is intentionally not part of this solid-core path. It remains Step 4: exploratory, feasibility-gated, and expensive.

## Apply

```bash
unzip -o solid_core_steps_patch.zip
chmod +x scripts/generate_step1_dense_factual_configs.py \
         scripts/plot_step0_artifact_overlay.py \
         scripts/analyze_step1_dense_durability.py \
         scripts/run_solid_core_after_results.sh
pip install -e .
```

## Step 1: generate the dense factual sweep config

```bash
python scripts/generate_step1_dense_factual_configs.py \
  --template configs/e3_critical_period_intervention_final_v4.yaml \
  --out configs/step1_factual_dense_160m_32ckpt.yaml \
  --output-root results/step1_factual_dense_160m_32ckpt
```

If the template path differs, pass the E3 factual config that already worked in your repo. The generator writes redundant aliases (`model`, `models`, `checkpoints`, `stages`, `revisions`) to reduce schema mismatch risk.

Checkpoint grid:

```text
step0, step1, step2, step4, step8, step16, step32, step64, step128, step256, step512,
step1000, step2000, ..., step9000,
step10000, step13000, step16000, step23000, step32000, step44000, step64000,
step89000, step100000, step110000, step128000, step143000
```

## Step 1: run the factual sweep

Use the E3 factual runner that is known to work in your repo. For example:

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_e3_factual_cell_v3.py \
  --config configs/step1_factual_dense_160m_32ckpt.yaml \
  2>&1 | tee logs/solid_core/step1_dense_factual_sweep.log
```

If your working runner has another name, use that name. The generated config is intended to be compatible with the existing E3 factual runner, not to replace it.

## Step 1: analyze

```bash
python scripts/analyze_step1_dense_durability.py \
  --inputs results/step1_factual_dense_160m_32ckpt \
  --out results/step1_dense_durability_analysis
```

Outputs:

```text
results/step1_dense_durability_analysis/figures/durability_sweep.png
results/step1_dense_durability_analysis/figures/break_test.png
results/step1_dense_durability_analysis/tables/step1_stage_summary.csv
results/step1_dense_durability_analysis/tables/window_vs_late_bootstrap.csv
results/step1_dense_durability_analysis/tables/segmented_vs_monotone_aic.csv
results/step1_dense_durability_analysis/reports/step1_dense_durability_report.md
```

## Step 0: artifact overlay

Use existing dense E1 160M plus the Step-1 durability summary:

```bash
python scripts/plot_step0_artifact_overlay.py \
  --e1-metrics results/e1_dense_160m_all_checkpoints/raw/e1_spectral_metrics.csv \
  --durability results/step1_dense_durability_analysis/tables/step1_stage_summary.csv \
  --out results/step0_artifact_overlay \
  --warmup-end 1400
```

Outputs:

```text
results/step0_artifact_overlay/figures/artifact_overlay.png
results/step0_artifact_overlay/tables/artifact_overlay_values.csv
results/step0_artifact_overlay/reports/artifact_overlay_report.md
```

## Review ZIP

```bash
zip -r solid_core_for_review.zip \
  configs/step1_factual_dense_160m_32ckpt.yaml \
  configs/step1_dense_32_checkpoints.json \
  results/step1_dense_durability_analysis \
  results/step0_artifact_overlay \
  logs/solid_core
```

## Decision rules

The solid sensitive-window claim is supported if:

- retention/degradation peak in the early reorganisation window and decline later;
- the segmented/break model beats monotone alternatives;
- window-vs-late bootstrap difference is positive;
- the effect remains after uptake-covariate checks;
- the consolidation/durability transition is offset from the warmup landmark.

If Step 1 becomes monotone, the thesis should report an honest negative for the period claim and keep E1/E2/C1 as exploratory. If Step 0 shows consolidation geometry exactly at warmup-end, mechanism should be hedged as schedule-entangled.
