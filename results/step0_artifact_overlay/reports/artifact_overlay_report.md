# Step 0 artifact overlay report

- E1 metrics: `results/e1_dense_160m_all_checkpoints/raw/e1_spectral_metrics.csv`
- Durability file: `results/step1_dense_durability_analysis/tables/step1_stage_summary.csv`
- Warmup-end marker: step 1400
- Geometry trace used: SV/subspace stability

Interpretation checklist:

1. Does the durability decline occur at/after the geometry re-stabilisation region rather than exactly at warmup end?
2. Are directional indicators (SV/subspace stability or stable rank) offset from the warmup landmark?
3. If geometry and durability sit directly on warmup-end, hedge mechanism as schedule-entangled.

Outputs:

- `figures/artifact_overlay.png`
- `tables/artifact_overlay_values.csv`
- `tables/pythia_lr_schedule_overlay.csv`
