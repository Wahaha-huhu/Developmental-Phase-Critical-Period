# C1 4090 continuation pilots

These configs assume the C1 label-gate injection calibration has already found a stable, effective setting:

- Pythia-160M at `step1000`
- `force_float32: true`
- injection LR `1e-6`
- 100 injection steps
- structured `REFUSE` / `COMPLY` label gate

The purpose of these pilots is not the final critical-period test. They answer:

1. Does the learned gate survive a small amount of generic LM continuation?
2. Are continuation losses finite under the safe recipe?
3. Are endpoint scores, jailbreak probes, poison attacks, and delta-persistence logs produced?
4. Does post-hoc injection at the matched endpoint behave differently from an in-window carried signal?

Run in order: 0.001 -> 0.005 -> 0.01. Stop if any run shows NaN, complete gate erasure, or over-refusal.

## Token budgets

For endpoint `step2000` and injection `step1000`, the full Pythia-equivalent interval is:

`1000 * 2,097,152 = 2,097,152,000 tokens`.

The configs use scaled budgets:

- `0.001`: ~2.10M tokens
- `0.005`: ~10.49M tokens
- `0.01`: ~20.97M tokens

With `batch_size=16`, `sequence_length=512`, and `gradient_accumulation_steps=8`, local tokens/update are `65,536`, so expected in-window continuation updates are roughly:

- `0.001`: 32 updates
- `0.005`: 160 updates
- `0.01`: 320 updates

The post-hoc arm injects at the endpoint and has zero long continuation.

## Run

```bash
unzip -o c1_4090_continuation_pilots_patch.zip
chmod +x scripts/run_c1_4090_continuation_pilots.sh
pip install -e .

CONFIG=configs/c1_alignment_160m_4090_step1000_to_step2000_scale0p001.yaml
python scripts/plan_c1_continuation_token_budgets.py \
  --config "$CONFIG" \
  --out-config configs/tmp_resolved.yaml \
  --out-csv results/tmp_budget_audit.csv \
  --endpoint-step 2000 \
  --pythia-tokens-per-step 2097152
```

To run all three:

```bash
bash scripts/run_c1_4090_continuation_pilots.sh
```

## Review

Upload `c1_4090_continuation_pilots_for_review.zip`, or an intermediate per-config zip if you stop early.
