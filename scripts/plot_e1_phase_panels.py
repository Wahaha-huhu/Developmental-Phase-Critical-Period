#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from critical_periods.io.artifacts import ArtifactRecord, ArtifactRegistry, ensure_experiment_dirs
from critical_periods.plots.e1 import plot_metric_by_step, write_summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot E1 phase-identification panels.")
    parser.add_argument("--metrics", required=True, help="Path to e1_spectral_metrics.csv")
    parser.add_argument("--output-root", default="results/e1_phase_identification")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    output_root = Path(args.output_root)
    dirs = ensure_experiment_dirs(output_root)
    manifest_path = dirs["manifests"] / "artifact_manifest.csv"
    registry = ArtifactRegistry(manifest_path)

    df = pd.read_csv(metrics_path)
    metrics_to_plot = [
        "stable_rank",
        "effective_rank",
        "subspace_stability_topk",
        "mp_outliers_x1",
        "spectral_norm",
        "frobenius_norm",
    ]
    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    records = []
    for model_name, model_df in df.groupby("model"):
        safe_model = model_name.split("/")[-1]
        for metric in metrics_to_plot:
            for log_x in (False, True):
                axis_name = "logstep" if log_x else "linearstep"
                fig_path = dirs["figures"] / f"e1_{safe_model}_{metric}_{axis_name}.png"
                plot_metric_by_step(
                    model_df,
                    metric=metric,
                    output_path=fig_path,
                    log_x=log_x,
                    title=f"{safe_model}: {metric} ({axis_name})",
                )
                records.append(
                    ArtifactRecord(
                        experiment_id="e1_phase_identification",
                        artifact_type="figure",
                        path=fig_path,
                        thesis_section="Chapter 4, E1: Phase identification and validation",
                        caption_draft=f"{safe_model} {metric.replace('_', ' ')} across Pythia checkpoints, averaged by module.",
                        source_data=metrics_path,
                        code_entrypoint="scripts/plot_e1_phase_panels.py",
                        status="draft",
                        notes=f"x-axis={axis_name}; grouped by module; mean over layers.",
                    )
                )

    table_path = dirs["tables"] / "e1_module_checkpoint_summary.csv"
    write_summary_table(df, table_path)
    records.append(
        ArtifactRecord(
            experiment_id="e1_phase_identification",
            artifact_type="table",
            path=table_path,
            thesis_section="Chapter 4, E1: Phase identification and validation",
            caption_draft="Module-level summary of E1 spectral indicators by model and checkpoint.",
            source_data=metrics_path,
            code_entrypoint="scripts/plot_e1_phase_panels.py",
            status="draft",
        )
    )

    registry.append_many(records)
    print(f"Wrote {len(records)} figure/table artifact records to {manifest_path}")


if __name__ == "__main__":
    main()
