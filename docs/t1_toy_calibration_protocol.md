# T1 toy calibration protocol

T1 is a controlled calibration experiment. Its role is to check whether the weight-spectral indicators used in E1 behave sensibly around a known behavioural transition.

## Task

The model receives a prompted pair `(task_id, x)` and predicts `y`. For each task, the target is a modular affine map:

```text
y = (a_t x + b_t) mod p
```

The training set contains a fixed subset of `x` values for each task. The held-out set contains the remaining `x` values. This gives a behavioural generalisation measure separate from training accuracy.

## Outputs

The script writes:

```text
results/t1_toy_calibration/raw/t1_training_curve.csv
results/t1_toy_calibration/raw/t1_spectral_metrics.csv
results/t1_toy_calibration/raw/t1_intervention_retention.csv
results/t1_toy_calibration/reports/t1_run_summary.md
```

The analysis script writes:

```text
results/t1_toy_calibration/figures/
results/t1_toy_calibration/tables/
results/t1_toy_calibration/reports/t1_analysis_report.md
```

## Interpretation

A useful T1 result is not simply high accuracy. The useful result is alignment between:

1. a behavioural transition in held-out accuracy;
2. a change in spectral indicators such as stable rank, effective rank, or subspace stability;
3. a stage-dependent difference in inject-then-washout retention.

A negative T1 result is also useful: it would tell us that the indicator panel needs more careful calibration before being used to motivate E3.
