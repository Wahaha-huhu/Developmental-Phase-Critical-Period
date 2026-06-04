# E1 multi-model artifact protocol

After running each Pythia model size, preserve its raw CSV separately:

```bash
mkdir -p results/e1_phase_identification/model_raw_backups
cp results/e1_phase_identification/raw/e1_spectral_metrics.csv \
  results/e1_phase_identification/model_raw_backups/e1_spectral_metrics_<MODEL>.csv
```

Then merge and regenerate multi-model artifacts:

```bash
python scripts/merge_e1_model_raws.py --root results/e1_phase_identification
```

If the 70M CSV is only available from an earlier ZIP, extract it and pass it explicitly:

```bash
python scripts/merge_e1_model_raws.py \
  --root results/e1_phase_identification \
  --extra-csv /path/to/e1_spectral_metrics_70m.csv
```

The output `raw/e1_spectral_metrics_combined.csv` should contain one row set per model size. Use this combined file for final thesis figures.
