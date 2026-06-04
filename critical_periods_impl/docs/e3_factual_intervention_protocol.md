# E3 factual-association intervention MVP

This patch implements the first E3 causal pipeline. It intentionally runs only the primary
synthetic factual-association signal before the full E3/E4/E6 grid is attempted.

## Scientific role

E1 identified a candidate developmental window from weight-space indicators. E2 showed that
this window overlaps with rapid fixed-text NLL and syntax-margin improvement. E3 now asks
whether the same checkpoint window matters causally: is a synthetic signal injected at one
stage more durably retained or harder to degrade than the same signal injected at another
stage?

This MVP measures **checkpoint fine-tuning amenability and short-horizon durability**, not
lifetime pretraining imprint. It uses full fine-tuning from public Pythia checkpoints.

## Recommended order

1. Run the smoke test:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python scripts/run_e3_factual_intervention.py --config configs/e3_factual_160m_smoke.yaml
python scripts/analyze_e3_results.py --root results/e3_critical_period_intervention
```

2. Inspect uptake. The smoke test is useful only if post-injection accuracy and/or margin
   improves over base.

3. Run the 160M MVP:

```bash
python scripts/run_e3_factual_intervention.py --config configs/e3_factual_160m_mvp.yaml
python scripts/analyze_e3_results.py --root results/e3_critical_period_intervention
```

4. Only after the 160M result is interpretable, optionally run the 410M reduced scale check:

```bash
python scripts/run_e3_factual_intervention.py --config configs/e3_factual_410m_reduced.yaml
python scripts/analyze_e3_results.py --root results/e3_critical_period_intervention
```

## Output files

Summary metrics:

```text
results/e3_critical_period_intervention/raw/e3_factual_metrics.csv
```

Item-level probe metrics:

```text
results/e3_critical_period_intervention/raw/e3_factual_item_metrics.csv
```

Analysis outputs:

```text
results/e3_critical_period_intervention/processed/e3_factual_cell_summary.csv
results/e3_critical_period_intervention/tables/e3_factual_stage_summary.csv
results/e3_critical_period_intervention/reports/e3_factual_intervention_report.md
results/e3_critical_period_intervention/figures/
```

## Packaging for review

```bash
zip -r e3_factual_intervention_for_review.zip results/e3_critical_period_intervention
```

## Interpretation rules

- First check uptake. A stage with weak uptake is not interpretable as a durability test.
- Report both absolute and uptake-normalized retention.
- A positive sensitive-period result requires a boundary-like difference near the E1/E2 window,
  not merely a smooth monotone early-to-late decline.
- Degradation-resistance is an application-relevant durability measure; clean continuation is
  the simpler scientific retention measure.
