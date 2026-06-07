# Step 1 dense factual durability analysis

Inputs: `['results/step1_factual_dense_160m_32ckpt']`

Retention column: `normalized_retention_margin`

Window: step512–step3000; late >= step8000

## Window-vs-late bootstrap

|   window_mean |   late_mean |       diff |    ci_low |    ci_high |   p_le_0 | metric                 |
|--------------:|------------:|-----------:|----------:|-----------:|---------:|:-----------------------|
|       0.66588 |    0.359909 |   0.305971 |   0.27779 |   0.333854 |        0 | retention_margin       |
|       1.46391 |    3.03731  |  -1.5734   |  -1.76589 |  -1.389    |        1 | uptake_margin          |
|     -73.1765  |    8.45777  | -81.6342   | -94.7402  | -69.2933   |        1 | degradation_auc_margin |

## Segmented-vs-monotone AIC

| model             |   break_step |       aic |     rss | params                                                                                             |
|:------------------|-------------:|----------:|--------:|:---------------------------------------------------------------------------------------------------|
| segmented_logstep |         2000 | -193.461  | 162.902 | [np.float64(-2.041401679315795), np.float64(0.8231131948154005), np.float64(-1.0715606273356564)]  |
| segmented_logstep |         1400 | -193.45   | 162.908 | [np.float64(-2.088029532338676), np.float64(0.8707673200297179), np.float64(-1.0757885901645574)]  |
| segmented_logstep |         1000 | -191.856  | 163.748 | [np.float64(-2.1197323564789596), np.float64(0.9073609557489375), np.float64(-1.0602856681062098)] |
| segmented_logstep |         3000 | -191.225  | 164.082 | [np.float64(-1.983788181273188), np.float64(0.7699803486434377), np.float64(-1.0685722462425715)]  |
| segmented_logstep |         4000 | -188.306  | 165.634 | [np.float64(-1.940210574462321), np.float64(0.7329590210847972), np.float64(-1.0669862026441213)]  |
| segmented_logstep |          512 | -183.01   | 168.488 | [np.float64(-2.1578291226078306), np.float64(0.9695040999809632), np.float64(-1.0185728083680792)] |
| segmented_logstep |         8000 | -178.467  | 170.975 | [np.float64(-1.8387909045051771), np.float64(0.6555363082642365), np.float64(-1.0993628205499126)] |
| monotone_logstep  |              | -118.106  | 209.073 | [np.float64(-1.4341792609920738), np.float64(0.434451434039614)]                                   |
| constant          |              |   33.4027 | 343.048 | [np.float64(-0.004777769818662368)]                                                                |

Outputs:

- `figures/durability_sweep.png`
- `figures/break_test.png`
- `tables/step1_stage_summary.csv`
- `tables/window_vs_late_bootstrap.csv`
- `tables/segmented_vs_monotone_aic.csv`
