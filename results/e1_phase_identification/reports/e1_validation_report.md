# E1 phase-identification validation report

## Data coverage

- Rows: 480
- Models: 1
- Checkpoints: 10
- Layers: 12
- Modules: 4

## Checkpoint completeness

- `EleutherAI/pythia-160m-deduped` `step0` (step 0): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step128` (step 128): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step512` (step 512): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step1000` (step 1000): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step2000` (step 2000): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step3000` (step 3000): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step8000` (step 8000): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step16000` (step 16000): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step64000` (step 64000): 48 rows, 12 layers, 4 modules.
- `EleutherAI/pythia-160m-deduped` `step143000` (step 143000): 48 rows, 12 layers, 4 modules.

## Candidate boundary consensus

- `EleutherAI/pythia-160m-deduped` 512→1000: 16 votes across 7 metrics and 4 modules. Metrics: alpha_tail_frac_0.3; effective_rank; mp_outliers_x1; mp_outliers_x1.1; mp_outliers_x1.25; stable_rank; subspace_stability_topk. Modules: attention.dense; attention.query_key_value; mlp.dense_4h_to_h; mlp.dense_h_to_4h.
- `EleutherAI/pythia-160m-deduped` 128→512: 15 votes across 5 metrics and 4 modules. Metrics: mp_outliers_x1; mp_outliers_x1.1; mp_outliers_x1.25; stable_rank; subspace_stability_topk. Modules: attention.dense; attention.query_key_value; mlp.dense_4h_to_h; mlp.dense_h_to_4h.
- `EleutherAI/pythia-160m-deduped` 1000→2000: 11 votes across 6 metrics and 4 modules. Metrics: alpha_tail_frac_0.3; effective_rank; mp_outliers_x1.1; mp_outliers_x1.25; stable_rank; subspace_stability_topk. Modules: attention.dense; attention.query_key_value; mlp.dense_4h_to_h; mlp.dense_h_to_4h.
- `EleutherAI/pythia-160m-deduped` 0→128: 7 votes across 3 metrics and 3 modules. Metrics: mp_outliers_x1; mp_outliers_x1.1; mp_outliers_x1.25. Modules: attention.dense; attention.query_key_value; mlp.dense_4h_to_h.
- `EleutherAI/pythia-160m-deduped` 64000→143000: 3 votes across 3 metrics and 2 modules. Metrics: alpha_tail_frac_0.3; effective_rank; stable_rank. Modules: attention.dense; attention.query_key_value.
- `EleutherAI/pythia-160m-deduped` 3000→8000: 2 votes across 2 metrics and 2 modules. Metrics: alpha_tail_frac_0.3; effective_rank. Modules: attention.dense; mlp.dense_h_to_4h.
- `EleutherAI/pythia-160m-deduped` 16000→64000: 2 votes across 2 metrics and 1 modules. Metrics: effective_rank; subspace_stability_topk. Modules: attention.query_key_value.

## Interpretation guide

- This report is observational. It supports phase identification only, not the causal sensitive-period claim.
- A strong E1 result requires multiple independent indicators to concentrate around similar checkpoint intervals.
- Module-staggered boundaries should be retained as an empirical finding, not smoothed into a single global transition.
- The next validation steps are model-size replication and behavioural grounding.

## Files generated

- `tables/e1_checkpoint_completeness.csv`
- `tables/e1_metric_summary_by_checkpoint.csv`
- `processed/e1_adjacent_checkpoint_changes.csv`
- `processed/e1_candidate_boundaries.csv`
- `processed/e1_boundary_consensus_table.csv`
- `figures/e1_candidate_transition_strength_heatmap.png` if matplotlib is available