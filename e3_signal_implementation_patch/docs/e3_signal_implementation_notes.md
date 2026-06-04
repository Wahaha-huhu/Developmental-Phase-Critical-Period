# E3 Signal Implementation Notes

This patch implements the synthetic signal layer used by E3 and later E4/E6.

## Smoke test signal generation

```bash
python scripts/smoke_e3_signal_generation.py --signal factual --n-items 300 --n-probe 200 --seed 0
```

The output preview should show:

- fictional names with bounded tokenization length;
- train templates that do not include the probe template;
- probes with a leading-space continuation value;
- controls with unseen entity/value pairs.

## Core design choices

- Factual signal uses full-text causal-LM loss.
- Procedural and alignment signals use response-only masked loss.
- Scoring should use conditional log-probability margins, not free generation.
- E3-MVP should start with the factual signal only.
