# E3 410M reduced scale-check review

## Coverage

- Model: `EleutherAI/pythia-410m-deduped`
- Cells: 24 = 8 stages × 3 seeds
- Stages: step0, step128, step512, step1000, step2000, step3000, step8000, step143000
- Positive uptake cells by margin: 23/24
- Mean uptake margin delta: 2.057

Note: the `experiment_id` field still says `e3_factual_160m_final_thesis_v4`, but the `model` field is correctly `EleutherAI/pythia-410m-deduped`. Treat the experiment-id label as a copied-config naming issue, not a data/model issue.

## Main stage pattern

Mean normalized clean-retention margin:

| stage      |   n |   uptake_margin_mean |   post_inj_acc_mean |   norm_ret_margin_mean |   norm_ret_acc_mean |
|:-----------|----:|---------------------:|--------------------:|-----------------------:|--------------------:|
| step0      |   3 |            0.0991175 |            0.41     |              -0.943548 |            0.277645 |
| step128    |   3 |            0.746021  |            0.53     |              -0.081703 |            0.249881 |
| step512    |   3 |            1.46183   |            0.555    |               0.5371   |            0.405736 |
| step1000   |   3 |            1.12598   |            0.428333 |               0.744282 |            0.947133 |
| step2000   |   3 |            1.69452   |            0.615    |               0.75349  |            0.656176 |
| step3000   |   3 |            2.13476   |            0.758333 |               0.706326 |            0.618182 |
| step8000   |   3 |            3.32034   |            0.923333 |               0.514454 |            0.530744 |
| step143000 |   3 |            5.86985   |            1        |               0.413681 |            0.671175 |

Mean normalized degradation-resistance:

| stage      |   n |   norm_deg_auc_margin_mean |   norm_deg_auc_acc_mean |   final_norm_margin_remaining_mean |
|:-----------|----:|---------------------------:|------------------------:|-----------------------------------:|
| step0      |   3 |                  -0.174084 |                0.811402 |                        -1.12457    |
| step128    |   3 |                   0.501645 |                0.634383 |                         0.00171336 |
| step512    |   3 |                   0.803005 |                0.761522 |                         0.580788   |
| step1000   |   3 |                   0.863353 |                0.881211 |                         0.571208   |
| step2000   |   3 |                   0.816285 |                0.717432 |                         0.562295   |
| step3000   |   3 |                   0.792986 |                0.738912 |                         0.519982   |
| step8000   |   3 |                   0.688848 |                0.734304 |                         0.315537   |
| step143000 |   3 |                   0.533657 |                0.687623 |                         0.127951   |

## Window-vs-late comparison

Using the E1/E2 window as `step512, step1000, step2000, step3000` and late controls as `step8000, step143000`:

- Mean normalized clean-retention margin, window: 0.685
- Mean normalized clean-retention margin, late: 0.464
- Paired seed-level window-minus-late retention difference: 0.221

- Mean normalized degradation AUC, window: 0.819
- Mean normalized degradation AUC, late: 0.611
- Paired seed-level window-minus-late normalized degradation AUC difference: 0.208

## Interpretation

The 410M reduced run replicates the qualitative E3 pattern from the 160M run. The injected factual signal has positive uptake in nearly all cells, and the uptake-normalized clean-retention curve is highest around the independently identified E1/E2 window (`step1000` and `step2000`) and lower at late checkpoints. The normalized degradation-resistance curve is also higher in the window than at late checkpoints, although the decline is gradual.

The `step0` and `step128` cells should be treated as diagnostic baselines rather than central evidence because uptake is weak and normalized ratios become unstable.

## Thesis-level claim

This is best used as a scale-check replication, not the primary E3 evidence. Together with the full 160M five-seed run, it supports the statement that stage-dependent short-horizon durability is not unique to 160M: a reduced 410M sweep shows the same higher durability around the early reorganization window and lower durability after consolidation.
