# E1 after-collection checklist

Run this after `scripts/run_e1_collect_spectra.py` finishes.

```bash
python scripts/analyze_e1_results.py \
  --metrics results/e1_phase_identification/raw/e1_spectral_metrics.csv
```

Then inspect:

```text
results/e1_phase_identification/tables/e1_checkpoint_completeness.csv
results/e1_phase_identification/tables/e1_metric_summary_by_model_step_module.csv
results/e1_phase_identification/processed/e1_boundary_consensus_table.csv
results/e1_phase_identification/reports/e1_validation_report.md
results/e1_phase_identification/figures/
```

Interpretation rules:

1. Do not infer a boundary from a single metric.
2. Treat norm growth as corroborative only.
3. Check linear-step plots before relying on log-step plots.
4. The 70M run is a pipeline validation. The phase claim should be replicated on 160M and ideally 410M.
5. The critical/sensitive-period claim is not established by E1. It requires E3 behavioural durability.

Recommended next run order:

1. Analyze 70M E1 output.
2. If the output looks complete, run 160M on the same checkpoint grid.
3. If 160M replicates the broad transition, run 410M or a reduced 410M checkpoint grid.
4. Start T1 toy calibration while 160M/410M downloads are running.
