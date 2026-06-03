# Critical Periods in LLM Pretraining — Implementation Scaffold

This repository is the implementation scaffold for the thesis project on developmental phases and critical/sensitive periods in LLM pretraining.

The first implemented milestone is **E1: phase identification**. It computes weight-spectral indicators across Pythia checkpoints and writes all useful outputs into experiment-specific folders so that figures and tables can be reused cleanly in the thesis.

## Repository layout

```text
configs/                         # YAML experiment configs
scripts/                         # runnable experiment and plotting entry points
src/critical_periods/            # reusable Python package
results/
  e1_phase_identification/
    raw/                         # raw metric rows, one row per matrix/checkpoint
    processed/                   # cleaned or aggregated CSVs
    figures/                     # thesis-ready plots
    tables/                      # thesis-ready summary tables
    reports/                     # short markdown reports
    manifests/                   # artifact registry CSV/JSONL
  t1_toy_calibration/            # reserved
  e2_functional_grounding/       # reserved
  e3_intervention/               # reserved
  e4_generality/                 # reserved
  e5_module_specificity/         # reserved
  e6_alignment_proxy/            # reserved
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For a single RTX 4090, start with Pythia-70M only, then add Pythia-160M and Pythia-410M after the pipeline is validated.

## Run E1 spectral collection

```bash
python scripts/run_e1_collect_spectra.py --config configs/e1_phase_identification.yaml
```

This writes:

```text
results/e1_phase_identification/raw/e1_spectral_metrics.csv
results/e1_phase_identification/manifests/artifact_manifest.csv
```

## Plot E1 figures

```bash
python scripts/plot_e1_phase_panels.py --metrics results/e1_phase_identification/raw/e1_spectral_metrics.csv
```

This writes figures and tables under:

```text
results/e1_phase_identification/figures/
results/e1_phase_identification/tables/
```

## Artifact convention

Every useful figure/table should be recorded in `artifact_manifest.csv` with:

- experiment id
- artifact type
- file path
- thesis target section
- caption draft
- source data path
- creation timestamp
- notes

This prevents losing track of which results support which thesis claim.
