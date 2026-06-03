# Results Artifact Protocol

This project treats figures and tables as first-class thesis evidence. Every useful artifact must be saved in the corresponding experiment folder and registered in the manifest.

## Experiment folders

```text
results/<experiment_id>/
  raw/          # unmodified outputs from data collection or training
  processed/    # cleaned/aggregated outputs used for plots and tests
  figures/      # thesis-ready PNG/PDF figures
  tables/       # thesis-ready CSV/LaTeX tables
  reports/      # short markdown interpretation reports
  manifests/    # artifact_manifest.csv
```

## Artifact manifest

The manifest file is append-only:

```text
results/<experiment_id>/manifests/artifact_manifest.csv
```

Each row records:

- `experiment_id`: e.g. `e1_phase_identification`
- `artifact_type`: `raw_csv`, `processed_csv`, `figure`, `table`, or `report`
- `path`: path to the artifact
- `thesis_section`: where the artifact is likely to be used
- `caption_draft`: draft thesis caption
- `source_data`: data used to create the artifact
- `code_entrypoint`: script that produced it
- `status`: `draft`, `validated`, `thesis_candidate`, or `deprecated`
- `notes`: details such as seed, checkpoint grid, metric version, caveats

## Status meanings

- `draft`: generated but not checked.
- `validated`: sanity checks passed.
- `thesis_candidate`: likely to appear in thesis.
- `deprecated`: superseded or invalidated; keep it for provenance but do not cite it.

## Naming convention

Use stable, descriptive names:

```text
e1_pythia-70m-deduped_stable_rank_linearstep.png
e1_pythia-70m-deduped_stable_rank_logstep.png
e1_module_checkpoint_summary.csv
```

Avoid names like `plot1.png` or `final_final.png`.
