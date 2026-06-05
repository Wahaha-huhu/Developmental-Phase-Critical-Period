# E3 statistical strengthening report

- Cells: 40
- Models: EleutherAI/pythia-160m-deduped
- Stages: 0, 128, 512, 1000, 2000, 3000, 8000, 143000
- Seeds: 0, 1, 2, 3, 4

- Positive uptake by margin: 40/40
- Mean uptake margin delta: 1.7850

## Window-vs-late bootstrap

- `normalized_retention_margin`: window mean=0.6659, late mean=0.3525, diff=0.3134, 95% CI=[0.2881, 0.3486], bootstrap p(diff<=0)=0.0000
- `normalized_retention_accuracy`: window mean=0.5931, late mean=0.3701, diff=0.2230, 95% CI=[0.1623, 0.2851], bootstrap p(diff<=0)=0.0000
- `normalized_degradation_auc_margin`: window mean=0.8790, late mean=0.6783, diff=0.2007, 95% CI=[0.1838, 0.2179], bootstrap p(diff<=0)=0.0000
- `normalized_degradation_auc_accuracy`: window mean=0.8286, late mean=0.7548, diff=0.0737, 95% CI=[0.0567, 0.0958], bootstrap p(diff<=0)=0.0000

## Best model comparisons by AIC


### normalized_degradation_auc_accuracy, include_uptake=False
- monotone_log_step break= AIC=-174.55 BIC=-171.74 LOOCV RMSE=0.0538 ΔAIC=0.00
- fixed_break_segmented break=1000 AIC=-173.21 BIC=-169.01 LOOCV RMSE=0.0553 ΔAIC=1.33
- free_break_candidate_segmented break=1000 AIC=-173.21 BIC=-169.01 LOOCV RMSE=0.0553 ΔAIC=1.33

### normalized_degradation_auc_accuracy, include_uptake=True
- fixed_break_segmented break=1000 AIC=-172.86 BIC=-167.26 LOOCV RMSE=0.0548 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-172.86 BIC=-167.26 LOOCV RMSE=0.0548 ΔAIC=0.00
- monotone_log_step break= AIC=-172.83 BIC=-168.63 LOOCV RMSE=0.0544 ΔAIC=0.03

### normalized_degradation_auc_margin, include_uptake=False
- fixed_break_segmented break=1000 AIC=-219.45 BIC=-215.24 LOOCV RMSE=0.0259 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-219.45 BIC=-215.24 LOOCV RMSE=0.0259 ΔAIC=0.00
- quadratic_log_step break= AIC=-207.67 BIC=-203.47 LOOCV RMSE=0.0315 ΔAIC=11.78

### normalized_degradation_auc_margin, include_uptake=True
- fixed_break_segmented break=1000 AIC=-223.37 BIC=-217.76 LOOCV RMSE=0.0236 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-223.37 BIC=-217.76 LOOCV RMSE=0.0236 ΔAIC=0.00
- quadratic_log_step break= AIC=-211.57 BIC=-205.97 LOOCV RMSE=0.0295 ΔAIC=11.80

### normalized_retention_accuracy, include_uptake=False
- fixed_break_segmented break=1000 AIC=-123.19 BIC=-118.99 LOOCV RMSE=0.1263 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-123.19 BIC=-118.99 LOOCV RMSE=0.1263 ΔAIC=0.00
- window_indicator break= AIC=-120.70 BIC=-116.49 LOOCV RMSE=0.1278 ΔAIC=2.50

### normalized_retention_accuracy, include_uptake=True
- fixed_break_segmented break=1000 AIC=-122.20 BIC=-116.60 LOOCV RMSE=0.1275 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-122.20 BIC=-116.60 LOOCV RMSE=0.1275 ΔAIC=0.00
- window_indicator break= AIC=-119.27 BIC=-113.67 LOOCV RMSE=0.1286 ΔAIC=2.93

### normalized_retention_margin, include_uptake=False
- fixed_break_segmented break=1000 AIC=-167.98 BIC=-163.78 LOOCV RMSE=0.0623 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-167.98 BIC=-163.78 LOOCV RMSE=0.0623 ΔAIC=0.00
- fixed_break_segmented break=2000 AIC=-153.70 BIC=-149.50 LOOCV RMSE=0.0775 ΔAIC=14.28

### normalized_retention_margin, include_uptake=True
- fixed_break_segmented break=1000 AIC=-168.37 BIC=-162.77 LOOCV RMSE=0.0607 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-168.37 BIC=-162.77 LOOCV RMSE=0.0607 ΔAIC=0.00
- window_indicator break= AIC=-165.40 BIC=-159.79 LOOCV RMSE=0.0599 ΔAIC=2.98

## Uptake-controlled regression

- `normalized_retention_margin`: window_coef=0.2492, late_coef=0.0970, uptake_z_coef=-0.1072, log_step_coef=0.0047, R²=0.897
- `normalized_retention_accuracy`: window_coef=0.2126, late_coef=-0.0042, uptake_z_coef=-0.0465, log_step_coef=0.0474, R²=0.449
- `normalized_degradation_auc_margin`: window_coef=0.3793, late_coef=0.4064, uptake_z_coef=-0.0549, log_step_coef=-0.1008, R²=0.951
- `normalized_degradation_auc_accuracy`: window_coef=0.3698, late_coef=0.4304, uptake_z_coef=0.0108, log_step_coef=-0.1075, R²=0.558

## Interpretation guide

- A positive window-vs-late bootstrap difference supports higher durability inside the E1/E2 window than at late stages.
- If a segmented or window model beats monotone log-step models, the evidence is stronger than a smooth 'earlier is better' account.
- Uptake-controlled regressions test whether the window effect remains after accounting for stage differences in post-injection uptake.
- Step0 and step128 are retained as diagnostics but excluded from model-comparison tests by default, because uptake at these stages is often weak or qualitatively different from trained checkpoints.