# E3 statistical strengthening report

- Cells: 64
- Models: EleutherAI/pythia-160m-deduped, EleutherAI/pythia-410m-deduped
- Stages: 0, 128, 512, 1000, 2000, 3000, 8000, 143000
- Seeds: 0, 1, 2, 3, 4

- Positive uptake by margin: 63/64
- Mean uptake margin delta: 1.8868

## Window-vs-late bootstrap

- `normalized_retention_margin`: window mean=0.6732, late mean=0.3943, diff=0.2789, 95% CI=[0.2366, 0.3179], bootstrap p(diff<=0)=0.0000
- `normalized_retention_accuracy`: window mean=0.6170, late mean=0.4567, diff=0.1603, 95% CI=[0.0926, 0.2323], bootstrap p(diff<=0)=0.0000
- `normalized_degradation_auc_margin`: window mean=0.8565, late mean=0.6532, diff=0.2033, 95% CI=[0.1910, 0.2146], bootstrap p(diff<=0)=0.0000
- `normalized_degradation_auc_accuracy`: window mean=0.8084, late mean=0.7384, diff=0.0700, 95% CI=[0.0502, 0.0892], bootstrap p(diff<=0)=0.0000

## Best model comparisons by AIC


### normalized_degradation_auc_accuracy, include_uptake=False
- monotone_log_step break= AIC=-261.50 BIC=-257.75 LOOCV RMSE=0.0650 ΔAIC=0.00
- fixed_break_segmented break=1000 AIC=-261.07 BIC=-255.45 LOOCV RMSE=0.0656 ΔAIC=0.43
- free_break_candidate_segmented break=1000 AIC=-261.07 BIC=-255.45 LOOCV RMSE=0.0656 ΔAIC=0.43

### normalized_degradation_auc_accuracy, include_uptake=True
- fixed_break_segmented break=3000 AIC=-259.81 BIC=-252.32 LOOCV RMSE=0.0663 ΔAIC=0.00
- free_break_candidate_segmented break=3000 AIC=-259.81 BIC=-252.32 LOOCV RMSE=0.0663 ΔAIC=0.00
- monotone_log_step break= AIC=-259.70 BIC=-254.09 LOOCV RMSE=0.0658 ΔAIC=0.11

### normalized_degradation_auc_margin, include_uptake=False
- fixed_break_segmented break=1000 AIC=-307.32 BIC=-301.71 LOOCV RMSE=0.0407 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-307.32 BIC=-301.71 LOOCV RMSE=0.0407 ΔAIC=0.00
- fixed_break_segmented break=2000 AIC=-296.23 BIC=-290.62 LOOCV RMSE=0.0454 ΔAIC=11.09

### normalized_degradation_auc_margin, include_uptake=True
- fixed_break_segmented break=1000 AIC=-314.33 BIC=-306.84 LOOCV RMSE=0.0379 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-314.33 BIC=-306.84 LOOCV RMSE=0.0379 ΔAIC=0.00
- fixed_break_segmented break=2000 AIC=-306.80 BIC=-299.31 LOOCV RMSE=0.0411 ΔAIC=7.53

### normalized_retention_accuracy, include_uptake=False
- fixed_break_segmented break=1000 AIC=-171.59 BIC=-165.98 LOOCV RMSE=0.1671 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-171.59 BIC=-165.98 LOOCV RMSE=0.1671 ΔAIC=0.00
- window_indicator break= AIC=-164.15 BIC=-158.54 LOOCV RMSE=0.1766 ΔAIC=7.44

### normalized_retention_accuracy, include_uptake=True
- fixed_break_segmented break=1000 AIC=-175.21 BIC=-167.73 LOOCV RMSE=0.1625 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-175.21 BIC=-167.73 LOOCV RMSE=0.1625 ΔAIC=0.00
- window_indicator break= AIC=-162.49 BIC=-155.01 LOOCV RMSE=0.1795 ΔAIC=12.72

### normalized_retention_margin, include_uptake=False
- fixed_break_segmented break=1000 AIC=-242.95 BIC=-237.33 LOOCV RMSE=0.0801 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-242.95 BIC=-237.33 LOOCV RMSE=0.0801 ΔAIC=0.00
- fixed_break_segmented break=2000 AIC=-229.06 BIC=-223.45 LOOCV RMSE=0.0924 ΔAIC=13.88

### normalized_retention_margin, include_uptake=True
- fixed_break_segmented break=1000 AIC=-241.17 BIC=-233.69 LOOCV RMSE=0.0836 ΔAIC=0.00
- free_break_candidate_segmented break=1000 AIC=-241.17 BIC=-233.69 LOOCV RMSE=0.0836 ΔAIC=0.00
- fixed_break_segmented break=2000 AIC=-227.21 BIC=-219.73 LOOCV RMSE=0.0949 ΔAIC=13.96

## Uptake-controlled regression

- `normalized_retention_margin`: window_coef=0.2502, late_coef=0.0792, uptake_z_coef=-0.0789, log_step_coef=0.0160, R²=0.718
- `normalized_retention_accuracy`: window_coef=0.2342, late_coef=0.0115, uptake_z_coef=0.0004, log_step_coef=0.0440, R²=0.176
- `normalized_degradation_auc_margin`: window_coef=0.3336, late_coef=0.3329, uptake_z_coef=-0.0725, log_step_coef=-0.0588, R²=0.902
- `normalized_degradation_auc_accuracy`: window_coef=0.3220, late_coef=0.3656, uptake_z_coef=-0.0120, log_step_coef=-0.0666, R²=0.376

## Interpretation guide

- A positive window-vs-late bootstrap difference supports higher durability inside the E1/E2 window than at late stages.
- If a segmented or window model beats monotone log-step models, the evidence is stronger than a smooth 'earlier is better' account.
- Uptake-controlled regressions test whether the window effect remains after accounting for stage differences in post-injection uptake.
- Step0 and step128 are retained as diagnostics but excluded from model-comparison tests by default, because uptake at these stages is often weak or qualitatively different from trained checkpoints.