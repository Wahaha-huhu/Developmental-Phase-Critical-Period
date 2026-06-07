# Step 1 dense factual durability analysis

## Input mapping
- Source CSV: `results/step1_factual_dense_160m_32ckpt/raw/e3_factual_cell_summary.csv`
- Uptake column: `uptake_margin_delta`
- Retention column: `normalized_retention_margin`
- Degradation-AUC column: `degradation_auc_margin`
- Exclude step0: `True`
- Clean cell rows: 155
- Duplicate stage/seed rows dropped: 0

## Summary
- Checkpoints in main analysis: 31
- Peak retention checkpoint: step 1000
- Peak retention mean: 0.7551
- Uptake at peak: 1.2284

## Generated figures
- `figures/step1_uptake_retention_curve_logx.png`
- `figures/step1_uptake_retention_curve_linear.png`
- `figures/step1_degradation_auc_curve_logx.png`
- `figures/step1_degradation_auc_curve_linear.png`
- `figures/step1_break_vs_monotone_positive_x.png`

AUC is plotted separately because its magnitude and interpretation differ from uptake/retention.
