# Next implementation steps after E1-70M

The completed Pythia-70M E1 run provides a usable first phase-identification result, but it should be treated as model-size-specific until replicated.

## Immediate next steps

1. Generate thesis-oriented early-window E1 plots:

```bash
python scripts/plot_e1_early_and_compare.py \
  --metrics results/e1_phase_identification/raw/e1_spectral_metrics.csv \
  --max-step 8000
```

2. Run the 160M replication:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python scripts/run_e1_collect_spectra.py \
  --config configs/e1_phase_identification_160m.yaml
```

3. Re-run the E1 analysis scripts after 160M finishes:

```bash
python scripts/analyze_e1_results.py \
  --metrics results/e1_phase_identification/raw/e1_spectral_metrics.csv

python scripts/plot_e1_early_and_compare.py \
  --metrics results/e1_phase_identification/raw/e1_spectral_metrics.csv \
  --max-step 8000
```

4. If 160M agrees with 70M, run the reduced 410M grid:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python scripts/run_e1_collect_spectra.py \
  --config configs/e1_phase_identification_410m_reduced.yaml
```

## Start T1 in parallel if the GPU is free

T1 is cheap and does not require Hugging Face downloads:

```bash
python scripts/run_t1_toy_calibration.py \
  --config configs/t1_toy_calibration.yaml

python scripts/analyze_t1_toy_results.py \
  --root results/t1_toy_calibration
```

## Thesis update rule

- After 70M only: write "E1 completed for Pythia-70M; strongest consensus around 1000--2000 steps; model-size replication pending."
- After 70M + 160M agreement: write "replicated across small Pythia sizes".
- After 410M agreement: make the E1 phase-identification result a major thesis result.
- Do not write "critical period" as a result until E3 intervention supports it.
