# E1 dense indicator panel report

- Rows after normalisation: 80,784
- Checkpoints: 153 (1 → 143000)
- Indicators plotted: stable_rank, effective_rank, spectral_norm, frobenius_norm
- Module families: Attention, MLP
- Module roles: Attention output, MLP in, MLP out, QKV

## Generated artifacts
- `tables/e1_metrics_long_normalized.csv`
- `tables/e1_indicator_overall_summary.csv`
- `tables/e1_indicator_by_module_family.csv`
- `tables/e1_indicator_by_layer_family.csv`
- `figures/e1_dense_indicator_overview_logx.png`
- `figures/e1_attention_vs_mlp_rank_logx.png`
- `figures/e1_module_role_stable_rank_logx.png`
- `figures/e1_module_role_effective_rank_logx.png`
- `figures/e1_module_role_spectral_norm_logx.png`
- `figures/e1_module_role_frobenius_norm_logx.png`
- `figures/e1_dense_indicator_overview_linear.png`
- `figures/e1_attention_vs_mlp_rank_linear.png`
- `figures/e1_module_role_stable_rank_linear.png`
- `figures/e1_module_role_effective_rank_linear.png`
- `figures/e1_module_role_spectral_norm_linear.png`
- `figures/e1_module_role_frobenius_norm_linear.png`
- `figures/e1_layer_heatmap_stable_rank_attention.png`
- `figures/e1_layer_heatmap_stable_rank_mlp.png`
- `figures/e1_layer_heatmap_effective_rank_attention.png`
- `figures/e1_layer_heatmap_effective_rank_mlp.png`

## Intended thesis use

- Main text: use the compact attention-vs-MLP rank comparison when discussing module-level differences.
- Appendix: use the full dense overview, per-module-role curves, and layerwise heatmaps.
- Avoid treating norm-only plots as primary evidence for phase boundaries; use them as corroborative diagnostics.
