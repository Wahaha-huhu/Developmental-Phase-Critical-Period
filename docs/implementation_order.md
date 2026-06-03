# Implementation Order

## M1 — E1 phase identification

Goal: validate whether the preliminary weight-space phase structure is real.

Run order:

1. Pythia-70M-deduped with the default checkpoint grid.
2. Inspect raw CSV for missing values and impossible values.
3. Generate linear-step and log-step plots.
4. Add Pythia-160M-deduped.
5. Add Pythia-410M-deduped if memory/time are acceptable.
6. Write a short E1 report under `results/e1_phase_identification/reports/`.

Do not start E3 until E1 boundaries are validated on a linear step axis and MP/alpha sensitivity has been inspected.

## M2 — T1 toy calibration

Goal: check whether the indicator panel detects a known transition in a controlled setting.

## M3 — E2-lite functional grounding

Goal: align E1 boundaries with loss/probe curves over the same checkpoints.

## M4 — E3 minimal intervention

Goal: test whether injected-signal retention changes around the indicator-defined consolidation boundary.

Initial scope:

- model: Pythia-70M-deduped
- checkpoints: step0, step512, step1000, step2000, step3000, step8000, step64000
- signal: one synthetic factual association signal
- seeds: 3
- fixed injection steps and fixed washout steps
- normalized retention as the primary metric

## M5+ — E4/E5/E6 extensions

Only expand to generality, module-specificity, and alignment proxy after M4 gives a measurable retention effect.
