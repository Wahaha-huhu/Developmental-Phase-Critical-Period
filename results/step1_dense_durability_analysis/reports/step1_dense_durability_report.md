# Step 1 dense factual durability analysis

Inputs: `['results/step1_factual_dense_160m_32ckpt']`

Retention column: `normalized_retention_margin`

Window: step512–step3000; late >= step8000

## Window-vs-late bootstrap

|   window_mean |   late_mean |       diff |      ci_low |    ci_high |   p_le_0 | metric                 |
|--------------:|------------:|-----------:|------------:|-----------:|---------:|:-----------------------|
|      0.656593 |    0.367107 |   0.289486 |    0.248684 |   0.330012 |        0 | retention_margin       |
|      1.40393  |    3.10652  |  -1.70259  |   -2.00799  |  -1.41901  |        1 | uptake_margin          |
|    -81.159    |    8.53088  | -89.6899   | -109.007    | -70.6671   |        1 | degradation_auc_margin |

## Segmented-vs-monotone AIC

| model             |   break_step |      aic |     rss | params                                                                                             |
|:------------------|-------------:|---------:|--------:|:---------------------------------------------------------------------------------------------------|
| segmented_logstep |         3000 | -67.6052 | 140.314 | [np.float64(-1.781624897574206), np.float64(0.6900081526007033), np.float64(-0.9174876306096531)]  |
| segmented_logstep |         2000 | -67.4725 | 140.407 | [np.float64(-1.8080132501169084), np.float64(0.7250829904365546), np.float64(-0.8984371348438913)] |
| segmented_logstep |         4000 | -67.0892 | 140.673 | [np.float64(-1.759899877580685), np.float64(0.6643875761027311), np.float64(-0.9300585479373499)]  |
| segmented_logstep |         1400 | -66.496  | 141.087 | [np.float64(-1.8265964073361471), np.float64(0.7544342628484312), np.float64(-0.880388975543542)]  |
| segmented_logstep |         1000 | -65.1161 | 142.054 | [np.float64(-1.83684901808442), np.float64(0.775244435755401), np.float64(-0.8493690818233219)]    |
| segmented_logstep |         8000 | -64.2983 | 142.63  | [np.float64(-1.7053920160272766), np.float64(0.6085307638493433), np.float64(-0.9872798569826691)] |
| segmented_logstep |          512 | -60.3014 | 145.481 | [np.float64(-1.838122636198284), np.float64(0.8029982665582416), np.float64(-0.7736400352950472)]  |
| monotone_logstep  |              | -40.146  | 162.345 | [np.float64(-1.4537358073805724), np.float64(0.4378614769287378)]                                  |
| constant          |              |  62.5293 | 272.576 | [np.float64(-0.1233082935704259)]                                                                  |

Outputs:

- `figures/durability_sweep.png`
- `figures/break_test.png`
- `tables/step1_stage_summary.csv`
- `tables/window_vs_late_bootstrap.csv`
- `tables/segmented_vs_monotone_aic.csv`
