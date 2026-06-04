# E2-lite functional grounding report
## Coverage
- `EleutherAI/pythia-160m-deduped`: 410 metric rows, 10 checkpoints.
- `EleutherAI/pythia-70m-deduped`: 410 metric rows, 10 checkpoints.

## Interpretation guide
E2-lite is observational. It asks whether lightweight behavioural/log-likelihood measures change near the E1 candidate window (approximately 128--2000 steps). It does not establish a sensitive or critical period.

## Largest adjacent changes
- `EleutherAI/pythia-160m-deduped` `fixed_text` `nll`: 0→128, delta=-3.138.
- `EleutherAI/pythia-70m-deduped` `fixed_text` `nll`: 0→128, delta=-3.083.
- `EleutherAI/pythia-160m-deduped` `syntax_regularities` `gold_logprob_margin`: 128→512, delta=2.701.
- `EleutherAI/pythia-70m-deduped` `syntax_regularities` `gold_logprob_margin`: 128→512, delta=2.207.
- `EleutherAI/pythia-160m-deduped` `syntax_regularities` `gold_logprob_margin`: 1000→2000, delta=2.136.
- `EleutherAI/pythia-70m-deduped` `syntax_regularities` `gold_logprob_margin`: 512→1000, delta=2.007.
- `EleutherAI/pythia-70m-deduped` `fixed_text` `nll`: 128→512, delta=-1.655.
- `EleutherAI/pythia-160m-deduped` `fixed_text` `nll`: 128→512, delta=-1.638.
- `EleutherAI/pythia-70m-deduped` `syntax_regularities` `gold_logprob_margin`: 8000→16000, delta=1.467.
- `EleutherAI/pythia-70m-deduped` `syntax_regularities` `gold_logprob_margin`: 64000→143000, delta=-1.456.
- `EleutherAI/pythia-160m-deduped` `syntax_regularities` `gold_logprob_margin`: 8000→16000, delta=1.187.
- `EleutherAI/pythia-70m-deduped` `syntax_regularities` `gold_logprob_margin`: 3000→8000, delta=1.003.

## Thesis status
Use this report to decide whether the E1 geometric phase has a functional correlate. Strong support would require systematic loss/probe improvement near the E1 window rather than only at the final checkpoint. Smooth or noisy curves should be reported as weak/partial functional grounding.
