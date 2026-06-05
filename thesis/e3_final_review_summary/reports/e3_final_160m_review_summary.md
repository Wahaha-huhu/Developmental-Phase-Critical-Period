# E3 final 160M review summary

Data root: `results/e3_critical_period_intervention_final_v4`.

## Coverage

- Cells: 40 = 8 stages x 5 seeds.
- Stages: step0, step128, step512, step1000, step2000, step3000, step8000, step143000.
- Positive uptake by margin: 40/40.
- Mean uptake margin delta: 1.785.

## Main result

The factual signal is learned at every stage, but uptake is not perfectly matched: late checkpoints, especially `step143000`, have much larger post-injection margin. Therefore raw post-injection and raw degradation scores should be treated cautiously; the primary durability readouts should be uptake-normalized.

Clean retention gives the clearest sensitive-window evidence. The E1/E2 window stages (`step512`, `step1000`, `step2000`, `step3000`) have mean normalized margin retention 0.666, compared with 0.352 for late stages (`step8000`, `step143000`).

Normalized degradation also favours the window stages over late stages, but is more gradual/monotone than clean retention. Mean normalized degradation AUC is 0.879 in the window versus 0.678 late.

## Interpretation

This is supportive evidence for a short-horizon sensitive period: the injected factual signal enters the model at all stages, but the fraction retained after fixed Pile continuation is highest around the E1/E2 reorganisation window and lower after consolidation. The degradation result supports the same direction but looks more like a gradual loss of overwrite resistance than a sharp boundary.

## Guardrails

- `step0` and `step128` should be treated as diagnostic stages, not primary boundary evidence.
- Because uptake is larger at late checkpoints, thesis figures should show uptake next to normalized retention/degradation.
- Claim "short-horizon sensitive-period evidence", not a fully established biological-style critical period.
