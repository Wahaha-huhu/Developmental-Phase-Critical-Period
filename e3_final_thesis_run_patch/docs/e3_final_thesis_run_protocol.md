# E3 final thesis run protocol

This patch freezes the final E3 factual-signal design around three requirements:

1. Probes query the same taught synthetic facts using held-out templates.
2. Clean-retention uses one fixed continuation corpus, sampled once and reused unchanged across all stages.
3. Degradation uses nested conflicting-fact budgets; every poison budget restarts from the same post-injection model.

## Fixed continuation corpus

Preferred final-thesis corpus: local held-out / validation Pile JSONL or another frozen held-out text corpus.
Create it once:

```bash
python scripts/build_e3_fixed_continuation_corpus.py \
  --source local_jsonl \
  --input /path/to/pile_validation_or_heldout.jsonl \
  --text-key text \
  --tokenizer EleutherAI/pythia-160m-deduped \
  --target-sequences 12000 \
  --max-tokens 256 \
  --min-tokens 64 \
  --seed 0 \
  --output data/e3_continuation/fixed_continuation_seed0.jsonl
```

Debug fallback only:

```bash
python scripts/build_e3_fixed_continuation_corpus.py \
  --source synthetic_generic \
  --target-sequences 12000 \
  --output data/e3_continuation/fixed_continuation_seed0.jsonl
```

The synthetic fallback is useful for checking plumbing but should not be the final thesis result.

## Recommended run order

1. Build the fixed continuation corpus.
2. Run the final calibration config.
3. Inspect uptake comparability.
4. Run the final 160M five-seed thesis sweep.
5. Run the 410M reduced scale check only if 160M is interpretable.

## Commands

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python scripts/run_e3_factual_cell_v3.py --config configs/e3_factual_160m_final_calibration_v4.yaml
python scripts/analyze_e3_v3_results.py --root results/e3_critical_period_intervention_final_calibration_v4
```

If uptake is positive and not wildly stage-confounded:

```bash
python scripts/run_e3_factual_cell_v3.py --config configs/e3_factual_160m_final_thesis_v4.yaml
python scripts/analyze_e3_v3_results.py --root results/e3_critical_period_intervention_final_v4
```

Then package:

```bash
zip -r e3_final_thesis_run_for_review.zip \
  results/e3_critical_period_intervention_final_calibration_v4 \
  results/e3_critical_period_intervention_final_v4 \
  configs/e3_factual_160m_final_calibration_v4.yaml \
  configs/e3_factual_160m_final_thesis_v4.yaml \
  data/e3_continuation/fixed_continuation_seed0.jsonl.meta.json
```

## Interpretation rule

Do not interpret durability for a cell with weak or negative uptake. The final thesis should report:

- absolute uptake;
- uptake-normalized clean retention;
- degradation AUC / k-star;
- sensitivity analyses excluding diagnostic-only or weak-uptake stages.

Step0 and step128 are useful diagnostic stages, but the primary stage-shape test should focus on the region where uptake is reliably positive.
