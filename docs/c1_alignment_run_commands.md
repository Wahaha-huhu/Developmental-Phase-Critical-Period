# Run commands

Apply the patch:

```bash
unzip -o c1_alignment_continue_to_consolidation_patch.zip
pip install -e .
```

Make sure the fixed Pile continuation corpus exists:

```bash
ls -lh data/e3_continuation/fixed_pile_val_seed0.jsonl
```

Run the one-seed pilot:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

python scripts/run_c1_alignment_consolidation.py \
  --config configs/c1_alignment_160m_pilot.yaml

python scripts/analyze_c1_alignment_consolidation.py \
  --root results/c1_alignment_160m_pilot
```

Inspect:

```bash
cat results/c1_alignment_160m_pilot/reports/c1_alignment_analysis_report.md
cat results/c1_alignment_160m_pilot/tables/c1_endpoint_retention_summary.csv
cat results/c1_alignment_160m_pilot/tables/c1_attack_normalized_summary.csv
cat results/c1_alignment_160m_pilot/tables/c1_trajectory_summary.csv
```

If the pilot passes, run the three-seed follow-up:

```bash
python scripts/run_c1_alignment_consolidation.py \
  --config configs/c1_alignment_160m_three_seed.yaml

python scripts/analyze_c1_alignment_consolidation.py \
  --root results/c1_alignment_160m_three_seed
```

Package:

```bash
zip -r c1_alignment_for_review.zip \
  results/c1_alignment_160m_pilot \
  results/c1_alignment_160m_three_seed \
  configs/c1_alignment_160m_pilot.yaml \
  configs/c1_alignment_160m_three_seed.yaml
```

To run only one arm or seed for debugging:

```bash
python scripts/run_c1_alignment_consolidation.py \
  --config configs/c1_alignment_160m_pilot.yaml \
  --arm in_window \
  --seed 0
```
