# E2 multi-model functional grounding summary

## Coverage

- `EleutherAI/pythia-160m-deduped`: 410 rows, 10 checkpoints.
- `EleutherAI/pythia-1b-deduped`: 410 rows, 10 checkpoints.
- `EleutherAI/pythia-410m-deduped`: 410 rows, 10 checkpoints.
- `EleutherAI/pythia-70m-deduped`: 410 rows, 10 checkpoints.

## Fixed-text NLL progress

- `EleutherAI/pythia-160m-deduped`: step0 NLL=11.059, step2000 NLL=5.207, fraction of total reduction by step2000=0.909.
- `EleutherAI/pythia-1b-deduped`: step0 NLL=11.042, step2000 NLL=4.892, fraction of total reduction by step2000=0.850.
- `EleutherAI/pythia-410m-deduped`: step0 NLL=10.995, step2000 NLL=5.119, fraction of total reduction by step2000=0.844.
- `EleutherAI/pythia-70m-deduped`: step0 NLL=11.036, step2000 NLL=5.333, fraction of total reduction by step2000=0.934.

## Syntax margin

- `EleutherAI/pythia-160m-deduped`: step0=-0.055; step128=-0.535; step512=2.165; step1000=2.703; step2000=4.839; step3000=4.545; step8000=5.519; step143000=7.001.
- `EleutherAI/pythia-1b-deduped`: step0=-0.208; step128=0.034; step512=2.243; step1000=4.155; step2000=7.440; step3000=7.106; step8000=5.904; step143000=7.728.
- `EleutherAI/pythia-410m-deduped`: step0=-0.693; step128=0.180; step512=1.911; step1000=4.169; step2000=5.712; step3000=6.855; step8000=6.965; step143000=7.576.
- `EleutherAI/pythia-70m-deduped`: step0=-0.073; step128=-0.363; step512=1.844; step1000=3.850; step2000=4.200; step3000=4.494; step8000=5.497; step143000=4.998.

## Interpretation

Use this summary to judge whether the lightweight functional measures move in the same early interval as E1. A consistent result across 410M/1B would strengthen the claim that E1's weight-space reorganisation has functional correlates. This remains observational evidence, not a causal sensitive-period test.
