# C1 Equivalent-Endpoint Pilot Notes

This pilot tests whether an alignment-style signal injected at `step1000` survives a smaller amount of continuation before committing to the expensive `step1000 -> step8000` run.

For each endpoint `E`, compare:

- **Carried arm:** load `step1000`, inject signal, continue for the Pythia-token-equivalent interval `1000 -> E`, then evaluate and attack.
- **Post-hoc arm:** load `stepE`, inject the same signal, no long continuation, then evaluate and attack.

This isolates whether a signal that is present during the interval leading to endpoint `E` becomes more durable than a signal inserted post-hoc at the same endpoint.

Recommended first endpoints:

- `step2000`: tests whether the signal exists after a short continuation through the main E1 boundary.
- `step3000`: tests trailing consolidation.
- `step4000`: tests early post-window carry-through.
- `step8000`: full consolidation endpoint, run only if the smaller pilots work.

Use a scaled token budget first (`budget_scale=0.01`) because full Pythia-token matching from 1000 to 8000 is approximately 14.68B tokens.

Suggested A100 pilot settings:

```yaml
sequence_length: 512
batch_size: 64
gradient_accumulation_steps: 16
budget_scale: 0.01
```

This gives 524,288 local tokens per optimizer update, so the 1% `step1000 -> step2000` pilot is about 40 local updates, and `step1000 -> step8000` is about 280 updates.

Go/no-go:

1. The Zorblax refusal gate must improve on trained instances.
2. It must generalize to held-out Zorblax instances.
3. It must preserve compliance on Florblax near-miss instances.
4. Weight-poison or jailbreak attacks should produce measurable degradation.

If the gate does not generalize at 160M, do not run the long full endpoint; either simplify the signal or escalate the pilot to 410M.
