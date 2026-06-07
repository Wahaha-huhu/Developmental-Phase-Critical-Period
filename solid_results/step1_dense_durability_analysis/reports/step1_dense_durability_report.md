# Step 1 Dense Factual Durability Analysis

## Input
- CSV: `results/step1_factual_dense_160m_32ckpt/raw/e3_factual_cell_summary.csv`
- Rows loaded: 106
- Rows after filtering: 101
- Exclude step0: True

## Plotting choices
- Uptake and retention are plotted together.
- Degradation/AUC is plotted separately because its magnitude is not comparable to uptake/retention.
- Break-vs-monotone uses positive injection steps only.

## Metric columns detected
- uptake_col: `uptake_margin_delta`
- retention_col: `None`
- degradation_auc_col: `None`
- degradation_margin_auc_col: `None`

## Output figures
- `solid_results/step1_dense_durability_analysis/figures/step1_uptake_retention_curve_linear.png`
- `solid_results/step1_dense_durability_analysis/figures/step1_uptake_retention_curve_logx.png`
