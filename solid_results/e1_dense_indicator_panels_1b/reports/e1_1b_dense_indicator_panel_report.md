# E1 dense indicator panels — 1b

- Source CSV: `solid_results/e1_dense_1b_early_sparse_late/raw/e1_spectral_metrics.csv`
- Output root: `solid_results/e1_dense_indicator_panels_1b`
- Rows after normalization: 34,560
- Step range: 1 → 143000
- Number of steps: 30
- Step0 excluded: True
- Indicators plotted: stable_rank, effective_rank, spectral_norm, frobenius_norm, alpha_tail_frac_0.2, alpha_tail_frac_0.3, alpha_tail_frac_0.5, cols, mean_singular_value, median_eigenvalue, mp_edge, mp_outliers_x1, mp_outliers_x1.1, mp_outliers_x1.25, nuclear_norm, num_singular_values, rows, subspace_stability_topk
- Module families: attention, mlp
- Module roles: attention_out, attention_qkv, mlp_in, mlp_out

## Main generated figures

- `figures/e1_1b_indicator_overview_logx.png`
- `figures/e1_1b_indicator_overview_linear.png`
- `figures/e1_1b_attention_vs_mlp_rank_logx.png`
- `figures/e1_1b_attention_vs_mlp_rank_linear.png`

## Appendix-style figures

- `figures/e1_1b_module_role_<indicator>_logx.png`
- `figures/e1_1b_module_role_<indicator>_linear.png`
- `figures/e1_1b_layer_heatmap_<indicator>_attention.png`
- `figures/e1_1b_layer_heatmap_<indicator>_mlp.png`
