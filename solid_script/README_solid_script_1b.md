# Solid scripts for E1 dense 1B plots

All new thesis-clean scripts should live in `solid_script/`. All new thesis-clean outputs should live in `solid_results/`.

## 1. Prepare folders

```bash
python solid_script/prepare_solid_results_layout.py
```

## 2. If you already have a 1B E1 metrics CSV

```bash
python solid_script/plot_solid_e1_dense_indicator_panels.py \
  --e1-metrics /path/to/1b/e1_spectral_metrics.csv \
  --out solid_results/e1_dense_indicator_panels_1b \
  --model-tag 1b \
  --exclude-step0
```

## 3. Generate a 1B E1 collection config

```bash
python solid_script/generate_e1_1b_solid_config.py \
  --template configs/e1_phase_identification.yaml \
  --out solid_results/configs/e1_dense_1b_early_sparse_late.yaml \
  --output-root solid_results/e1_dense_1b_early_sparse_late \
  --mode early_sparse_late
```

Optional cheaper collection request, if your collector respects metric flags:

```bash
python solid_script/generate_e1_1b_solid_config.py \
  --template configs/e1_phase_identification.yaml \
  --out solid_results/configs/e1_dense_1b_early_sparse_late.yaml \
  --output-root solid_results/e1_dense_1b_early_sparse_late \
  --mode early_sparse_late \
  --cheap-metrics
```

Then collect with your existing E1 runner:

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_e1_collect_spectra.py \
  --config solid_results/configs/e1_dense_1b_early_sparse_late.yaml \
  2>&1 | tee solid_results/logs/e1_dense_1b_collect.log
```

Then plot:

```bash
python solid_script/plot_solid_e1_dense_indicator_panels.py \
  --e1-metrics solid_results/e1_dense_1b_early_sparse_late/raw/e1_spectral_metrics.csv \
  --out solid_results/e1_dense_indicator_panels_1b \
  --model-tag 1b \
  --exclude-step0
```

## 4. One-command wrapper

The wrapper generates a config, optionally runs collection, then plots.

If the CSV already exists at the default location:

```bash
bash solid_script/run_solid_e1_1b_workflow.sh
```

If the CSV is elsewhere:

```bash
CSV=/path/to/1b/e1_spectral_metrics.csv bash solid_script/run_solid_e1_1b_workflow.sh
```

To run collection too:

```bash
RUN_COLLECT=1 bash solid_script/run_solid_e1_1b_workflow.sh
```

## Produced figures

For each model, the plotting script writes:

- `e1_1b_indicator_overview_logx.png`
- `e1_1b_indicator_overview_linear.png`
- `e1_1b_attention_vs_mlp_rank_logx.png`
- `e1_1b_attention_vs_mlp_rank_linear.png`
- `e1_1b_module_role_<indicator>_logx.png`
- `e1_1b_module_role_<indicator>_linear.png`
- `e1_1b_layer_heatmap_<indicator>_attention.png`
- `e1_1b_layer_heatmap_<indicator>_mlp.png`

Tables and a short report are written under the same output root.
