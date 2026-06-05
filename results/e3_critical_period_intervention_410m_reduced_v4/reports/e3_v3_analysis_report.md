# E3 factual v3 analysis report

- Root: `results/e3_critical_period_intervention_410m_reduced_v4`
- Cells: 24
- Positive uptake cells by margin: 23/24
- Mean uptake margin delta: 2.0566

## Interpretation guardrails

A cell is interpretable for durability only if uptake is positive and preferably comparable across stages. If uptake is weak or highly variable, normalize retention/degradation by uptake and treat absolute durability cautiously.

## Output files

- `tables/e3_v3_cell_summary.csv`
- `tables/e3_v3_stage_summary.csv`
- `figures/e3_v3_uptake_margin_delta.png`
- `figures/e3_v3_normalized_retention_margin.png`
- `figures/e3_v3_k_star_accuracy_threshold.png`
- `figures/e3_v3_degradation_auc_accuracy.png`