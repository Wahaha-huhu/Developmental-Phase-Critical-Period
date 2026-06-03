# E1 phase-identification artifact report

This report is generated automatically from `e1_spectral_metrics.csv`. It is a diagnostic artifact, not a final interpretation. Candidate boundaries should be accepted only after visual inspection, threshold sensitivity checks, and replication across model sizes.

## Run coverage

- Models: EleutherAI/pythia-70m-deduped
- Steps: [0, 128, 512, 1000, 2000, 3000, 8000, 16000, 64000, 143000]
- Metric rows: 240
- Unique matrices: 24
- Unique modules: 4
- Unique layers: 6
- Metrics summarized: stable_rank, effective_rank, spectral_norm, frobenius_norm, subspace_stability_topk, mp_outliers_x1, mp_outliers_x1.1, mp_outliers_x1.25, alpha_tail_frac_0.3

## Completeness checks

The table `tables/e1_checkpoint_completeness.csv` records how many matrices, layers, and modules were measured at each checkpoint. Before interpreting phase boundaries, confirm that each checkpoint has the expected number of matrices.

## Strongest adjacent-change candidates

- `EleutherAI/pythia-70m-deduped` / `mlp.dense_4h_to_h` / `mp_outliers_x1`: strongest adjacent signal at `0→128` (strength=9, direction=increase).
- `EleutherAI/pythia-70m-deduped` / `attention.query_key_value` / `mp_outliers_x1`: strongest adjacent signal at `0→128` (strength=7, direction=increase).
- `EleutherAI/pythia-70m-deduped` / `mlp.dense_h_to_4h` / `mp_outliers_x1`: strongest adjacent signal at `128→512` (strength=3.333, direction=increase).
- `EleutherAI/pythia-70m-deduped` / `attention.query_key_value` / `stable_rank`: strongest adjacent signal at `128→512` (strength=0.7066, direction=decrease).
- `EleutherAI/pythia-70m-deduped` / `mlp.dense_h_to_4h` / `stable_rank`: strongest adjacent signal at `1000→2000` (strength=0.6603, direction=decrease).
- `EleutherAI/pythia-70m-deduped` / `attention.query_key_value` / `subspace_stability_topk`: strongest adjacent signal at `128→512` (strength=0.531, direction=low_stability).
- `EleutherAI/pythia-70m-deduped` / `attention.dense` / `subspace_stability_topk`: strongest adjacent signal at `512→1000` (strength=0.5098, direction=low_stability).
- `EleutherAI/pythia-70m-deduped` / `mlp.dense_h_to_4h` / `alpha_tail_frac_0.3`: strongest adjacent signal at `1000→2000` (strength=0.3768, direction=decrease).
- `EleutherAI/pythia-70m-deduped` / `mlp.dense_4h_to_h` / `subspace_stability_topk`: strongest adjacent signal at `1000→2000` (strength=0.3747, direction=low_stability).
- `EleutherAI/pythia-70m-deduped` / `mlp.dense_h_to_4h` / `subspace_stability_topk`: strongest adjacent signal at `1000→2000` (strength=0.3385, direction=low_stability).
- `EleutherAI/pythia-70m-deduped` / `mlp.dense_4h_to_h` / `stable_rank`: strongest adjacent signal at `3000→8000` (strength=0.3377, direction=decrease).
- `EleutherAI/pythia-70m-deduped` / `attention.dense` / `stable_rank`: strongest adjacent signal at `1000→2000` (strength=0.2942, direction=decrease).

## Interpretation protocol

1. Inspect both linear-step and log/symlog-step plots. A transition that appears only on a log axis should not be treated as a sharp phase boundary.
2. Check whether multiple independent indicators agree: stable/effective-rank change, subspace-stability drop, and MP-outlier emergence are stronger together than any single metric alone.
3. Check module consistency. A global phase claim requires several modules/layers to show the same broad transition; otherwise the result should be phrased as module-specific.
4. Check model-size replication. The initial 70M result is a pipeline validation; the thesis claim should rely on replication across 70M/160M and, if feasible, 410M.
5. Treat norm growth as corroborative only. Norms often move monotonically and are not by themselves evidence of a developmental phase.

## Thesis use

These artifacts support Chapter 4, E1: phase identification and validation. They should not yet be used to claim a critical period. The critical/sensitive-period claim requires E3: behavioural durability after checkpoint-specific intervention.
