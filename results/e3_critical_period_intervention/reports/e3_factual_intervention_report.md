# E3 factual-intervention MVP report

## Coverage

| model                          | stage      |   step |   n_seeds |
|:-------------------------------|:-----------|-------:|----------:|
| EleutherAI/pythia-160m-deduped | step0      |      0 |         3 |
| EleutherAI/pythia-160m-deduped | step1000   |   1000 |         3 |
| EleutherAI/pythia-160m-deduped | step128    |    128 |         3 |
| EleutherAI/pythia-160m-deduped | step143000 | 143000 |         3 |
| EleutherAI/pythia-160m-deduped | step2000   |   2000 |         3 |
| EleutherAI/pythia-160m-deduped | step3000   |   3000 |         3 |
| EleutherAI/pythia-160m-deduped | step512    |    512 |         3 |
| EleutherAI/pythia-160m-deduped | step8000   |   8000 |         3 |

## Headline stage summary

| model                          | stage      |   step |   n_seeds |   post_injection_accuracy_mean |   uptake_margin_delta_mean |   normalized_clean_retention_margin_mean |   normalized_degradation_margin_auc_logk_mean |   k_star_accuracy_below_0p5_mean |
|:-------------------------------|:-----------|-------:|----------:|-------------------------------:|---------------------------:|-----------------------------------------:|----------------------------------------------:|---------------------------------:|
| EleutherAI/pythia-160m-deduped | step0      |      0 |         3 |                     0.0133333  |                   -1.18595 |                                 0.427474 |                                       1.50109 |                                4 |
| EleutherAI/pythia-160m-deduped | step128    |    128 |         3 |                     0.00833333 |                   -2.16588 |                                 0.558142 |                                       1.79348 |                                4 |
| EleutherAI/pythia-160m-deduped | step512    |    512 |         3 |                     0.015      |                   -1.11476 |                                 0.546076 |                                       1.52734 |                                4 |
| EleutherAI/pythia-160m-deduped | step1000   |   1000 |         3 |                     0.0133333  |                   -1.34362 |                                 0.759213 |                                       1.67833 |                                4 |
| EleutherAI/pythia-160m-deduped | step2000   |   2000 |         3 |                     0.005      |                   -1.59972 |                                 0.841703 |                                       1.60641 |                                4 |
| EleutherAI/pythia-160m-deduped | step3000   |   3000 |         3 |                     0.00916667 |                   -1.52024 |                                 0.939467 |                                       1.50318 |                                4 |
| EleutherAI/pythia-160m-deduped | step8000   |   8000 |         3 |                     0.00833333 |                   -1.34043 |                                 1.01486  |                                       1.19868 |                                4 |
| EleutherAI/pythia-160m-deduped | step143000 | 143000 |         3 |                     0.00666667 |                   -4.86124 |                                 1.08823  |                                       1.80096 |                                4 |

## Diagnostic model comparison

| model                          | metric                                 |   boundary_step |   n_points |   log_step_sse |   fixed_boundary_sse |   log_step_aic |   fixed_boundary_aic |   delta_aic_boundary_minus_logstep | boundary_model_better   |
|:-------------------------------|:---------------------------------------|----------------:|-----------:|---------------:|---------------------:|---------------:|---------------------:|-----------------------------------:|:------------------------|
| EleutherAI/pythia-160m-deduped | normalized_clean_retention_margin_mean |            2000 |          8 |      0.0719114 |            0.0363467 |       -33.6941 |             -37.1527 |                           -3.45865 | True                    |

## Interpretation guide

This MVP should first be judged by uptake. Cells with weak post-injection uptake are not interpretable as durability tests. The main durability quantities are uptake-normalized clean retention and degradation-resistance AUC. A sensitive-period result would require a boundary-like change near the independently measured E1 window, not merely a smooth monotone early-to-late decline.