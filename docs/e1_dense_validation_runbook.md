# E1 dense checkpoint validation runbook

Purpose: strengthen the stage-localisation claim by running a dense checkpoint sweep on Pythia-160M, and cheaper early-window dense sweeps on larger models.

Recommended design:

- `160M`: all available Pythia checkpoint revisions, full metrics. This is the main dense validation because E3's primary causal run is 160M.
- `410M`: all available checkpoint revisions up to step8000, cheaper metrics. This checks the early boundary shape at larger scale.
- `1B`: optional early-window cheap sweep if compute/download time is acceptable.

The dense run is not meant to move the goalposts after E3. It is a robustness check: the E3 stage grid already brackets the E1 window. Dense E1 should confirm or refine the boundary estimate, not redefine the causal result after the fact.

Important file hygiene:

- Use separate output roots for each dense run.
- Back up each raw CSV after the run.
- Do not mix dense outputs with previous sparse E1 outputs without a separate combined analysis script.
