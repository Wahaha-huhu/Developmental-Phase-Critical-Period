# E3 factual-intervention MVP report

## Coverage

| model                          | stage    |   step |   n_seeds |
|:-------------------------------|:---------|-------:|----------:|
| EleutherAI/pythia-160m-deduped | step1000 |   1000 |         3 |

## Headline stage summary

| model                          | stage    |   step |   n_seeds |   post_injection_accuracy_mean |   uptake_margin_delta_mean |   normalized_clean_retention_margin_mean |   normalized_degradation_margin_auc_logk_mean |   k_star_accuracy_below_0p5_mean |
|:-------------------------------|:---------|-------:|----------:|-------------------------------:|---------------------------:|-----------------------------------------:|----------------------------------------------:|---------------------------------:|
| EleutherAI/pythia-160m-deduped | step1000 |   1000 |         3 |                        0.13125 |                   -0.44354 |                                 0.956786 |                                      0.646771 |                                4 |

## Diagnostic model comparison

Not enough points for the fixed-boundary-vs-log-step diagnostic.

## Interpretation guide

This MVP should first be judged by uptake. Cells with weak post-injection uptake are not interpretable as durability tests. The main durability quantities are uptake-normalized clean retention and degradation-resistance AUC. A sensitive-period result would require a boundary-like change near the independently measured E1 window, not merely a smooth monotone early-to-late decline.