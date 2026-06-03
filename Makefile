.PHONY: e1 plot-e1

e1:
	python scripts/run_e1_collect_spectra.py --config configs/e1_phase_identification.yaml

plot-e1:
	python scripts/plot_e1_phase_panels.py --metrics results/e1_phase_identification/raw/e1_spectral_metrics.csv
