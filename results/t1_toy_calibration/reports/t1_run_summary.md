# T1 toy calibration run summary

- Device: `cuda`
- Prime: `97`
- Base tasks: `6`
- Base train examples: `264`
- Base held-out examples: `318`
- Training steps: `6000`
- Evaluation interval: `100`
- Spectral checkpoint interval: `200`
- Intervention enabled: `True`

Outputs:

- `raw/t1_training_curve.csv`
- `raw/t1_spectral_metrics.csv`
- `raw/t1_intervention_retention.csv` if intervention is enabled

Interpretation note: this experiment calibrates the indicator pipeline in a controlled task; it is not evidence that Pythia shares the toy mechanism.
