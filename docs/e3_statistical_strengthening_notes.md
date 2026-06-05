# E3 statistical strengthening analysis

This patch adds `scripts/analyze_e3_statistical_tests.py`.

It performs the analyses needed before starting a broad new experiment:

1. **Window-vs-late paired bootstrap** over seeds/cells.
2. **Segmented-vs-monotone model comparison** using AIC/BIC/LOOCV RMSE.
3. **Uptake-controlled regression** to check whether stage effects remain after accounting for uptake differences.
4. **Normalized degradation curves/AUC** computed from the raw poison-budget CSV.

The main window is the E1/E2-supported region: steps 512--3000. Late stages are step8000 and step143000. Step0 and step128 are kept as diagnostic baselines but excluded from model-comparison tests by default.
