# E2-lite functional grounding protocol

E2-lite asks whether the weight-space transition identified in E1 has any behavioural/log-likelihood correlate over the same checkpoints. It is not a causal experiment. The causal sensitive-period claim remains reserved for E3.

The first implementation uses lightweight local probes so it can run on the same 4090 environment without downloading external evaluation suites:

1. fixed short text snippets scored by average next-token negative log-likelihood;
2. small multiple-choice probes scored by conditional answer log probability;
3. checkpoint-wise plots aligned to the E1 candidate window.

The result should be interpreted conservatively. If model loss or probe scores improve sharply near the E1 window, E2 supports functional grounding. If the curves are smooth or noisy, E1 remains a geometric result and the thesis should say so.

Recommended first run:

```bash
python scripts/run_e2_functional_grounding.py --config configs/e2_functional_grounding.yaml
python scripts/analyze_e2_results.py --root results/e2_functional_grounding
```

For a fast smoke test, use only `EleutherAI/pythia-70m-deduped` and checkpoints `step0`, `step512`, `step2000`.
