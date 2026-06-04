# E2 larger-model extension protocol

This extension runs E2-lite on Pythia-410M and Pythia-1B using the same checkpoint grid and local probe set as the 70M/160M pilot. The purpose is not to establish a causal sensitive period, but to check whether the functional correlates of the E1 spectral window replicate at larger model scale.

Recommended order:

1. Run `configs/e2_functional_grounding_410m_1b.yaml` if time permits. This preserves comparability with the 70M/160M E2 run.
2. If runtime is a problem, use `configs/e2_functional_grounding_410m_1b_core.yaml`, which keeps only fixed-text NLL and syntax-regularity probes.
3. Run `scripts/analyze_e2_results.py` to regenerate ordinary E2 tables/plots.
4. Run `scripts/summarize_e2_multimodel.py` to create cross-model summary figures suitable for thesis review.

The most important outputs are:

- `results/e2_functional_grounding/reports/e2_multimodel_summary.md`
- `results/e2_functional_grounding/figures/e2_multimodel_fixed_text_nll.png`
- `results/e2_functional_grounding/figures/e2_multimodel_syntax_margin.png`
- `results/e2_functional_grounding/tables/e2_multimodel_metric_summary_by_checkpoint.csv`
- `results/e2_functional_grounding/processed/e2_fixed_text_nll_progress.csv`
- `results/e2_functional_grounding/processed/e2_syntax_margin_summary.csv`

Interpretation rule:

- If fixed-text NLL and syntax margins improve consistently inside or immediately around 128--2000 across 410M/1B, E2 supports a replicated lightweight functional correlate of E1.
- If only fixed-text NLL improves but probes remain noisy, E2 should be framed as weak/partial functional grounding.
- If larger models move the functional changes later, E2 should report a scale-dependent mismatch between geometric and behavioural probes rather than force alignment.
