#!/usr/bin/env python
"""Analyze E1 spectral metrics and produce thesis-ready artifacts.

This script is intentionally conservative: it does not declare a phase boundary.
It produces candidate boundary tables and validation figures that should be read
before writing the Results chapter.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "stable_rank",
    "effective_rank",
    "spectral_norm",
    "frobenius_norm",
    "subspace_stability_topk",
    "mp_outliers_x1",
    "mp_outliers_x1.1",
    "mp_outliers_x1.25",
    "alpha_tail_frac_0.3",
]

BOUNDARY_METRICS = [
    "stable_rank",
    "effective_rank",
    "subspace_stability_topk",
    "mp_outliers_x1",
    "alpha_tail_frac_0.3",
]

MANIFEST_COLUMNS = [
    "experiment_id",
    "artifact_type",
    "path",
    "thesis_section",
    "caption_draft",
    "source_data",
    "code_entrypoint",
    "status",
    "notes",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find metrics CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Metrics CSV is empty: {path}")
    required = {"model", "checkpoint", "step", "matrix_name", "module", "layer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {sorted(missing)}")
    return df


def _write_manifest_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: row.get(k, "") for k in MANIFEST_COLUMNS}
    write_header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False)


def _artifact(
    manifest_path: Path,
    artifact_type: str,
    path: Path,
    caption: str,
    notes: str = "",
    thesis_section: str = "Chapter 4, E1 phase identification",
) -> None:
    _write_manifest_row(
        manifest_path,
        {
            "experiment_id": "e1_phase_identification",
            "artifact_type": artifact_type,
            "path": str(path),
            "thesis_section": thesis_section,
            "caption_draft": caption,
            "source_data": "results/e1_phase_identification/raw/e1_spectral_metrics.csv",
            "code_entrypoint": "scripts/analyze_e1_results.py",
            "status": "draft",
            "notes": notes,
        },
    )


def available_metrics(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    out: list[str] = []
    for metric in candidates:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            if df[metric].notna().any():
                out.append(metric)
    return out


def write_completeness_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for (model, checkpoint, step), sub in df.groupby(["model", "checkpoint", "step"], dropna=False):
        rows.append(
            {
                "model": model,
                "checkpoint": checkpoint,
                "step": int(step),
                "n_rows": len(sub),
                "n_layers": sub["layer"].nunique(),
                "n_modules": sub["module"].nunique(),
                "n_matrices": sub["matrix_name"].nunique(),
                "modules": ";".join(sorted(map(str, sub["module"].unique()))),
            }
        )
    table = pd.DataFrame(rows).sort_values(["model", "step"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def write_metric_summary(df: pd.DataFrame, metrics: list[str], output_path: Path) -> pd.DataFrame:
    agg = (
        df.groupby(["model", "step", "checkpoint", "module"], as_index=False)[metrics]
        .agg(["mean", "std", "count"])
    )
    # Flatten MultiIndex columns created by aggregation.
    agg.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in agg.columns]
    sort_cols = [c for c in ["model", "step", "module"] if c in agg.columns]
    agg = agg.sort_values(sort_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)
    return agg


def _step_label(prev_step: float, step: float) -> str:
    return f"{int(prev_step)}→{int(step)}"


def write_change_tables(df: pd.DataFrame, metrics: list[str], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute adjacent-checkpoint changes and boundary candidates.

    For each metric/model/module we aggregate over layers first, then calculate
    adjacent changes. Candidate strength is a normalized absolute change. For
    subspace stability, low stability is itself treated as a transition signal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base = df.groupby(["model", "module", "step"], as_index=False)[metrics].mean(numeric_only=True)
    change_rows = []
    candidate_rows = []
    eps = 1e-12

    for (model, module), sub in base.groupby(["model", "module"]):
        sub = sub.sort_values("step")
        for metric in metrics:
            if metric not in sub.columns:
                continue
            vals = sub[["step", metric]].dropna()
            if len(vals) < 2:
                continue
            values = vals[metric].to_numpy(dtype=float)
            steps = vals["step"].to_numpy(dtype=float)
            for i in range(1, len(vals)):
                prev_v, v = values[i - 1], values[i]
                prev_step, step = steps[i - 1], steps[i]
                abs_delta = v - prev_v
                rel_delta = abs_delta / (abs(prev_v) + eps)
                log_step_delta = math.log1p(step) - math.log1p(prev_step)
                slope_log_step = abs_delta / (log_step_delta + eps)
                if metric == "subspace_stability_topk":
                    # Rows at current step summarize stability between previous and current checkpoint.
                    candidate_strength = 1.0 - v if np.isfinite(v) else np.nan
                    direction = "low_stability"
                else:
                    candidate_strength = abs(rel_delta)
                    direction = "increase" if abs_delta > 0 else "decrease"
                change_rows.append(
                    {
                        "model": model,
                        "module": module,
                        "metric": metric,
                        "transition": _step_label(prev_step, step),
                        "prev_step": int(prev_step),
                        "step": int(step),
                        "prev_value": prev_v,
                        "value": v,
                        "delta": abs_delta,
                        "relative_delta": rel_delta,
                        "slope_per_log_step": slope_log_step,
                        "candidate_strength": candidate_strength,
                        "direction": direction,
                    }
                )

            local = pd.DataFrame([r for r in change_rows if r["model"] == model and r["module"] == module and r["metric"] == metric])
            if not local.empty:
                best = local.sort_values("candidate_strength", ascending=False).iloc[0].to_dict()
                candidate_rows.append(best)

    changes = pd.DataFrame(change_rows).sort_values(["model", "module", "metric", "step"])
    candidates = pd.DataFrame(candidate_rows).sort_values(["model", "metric", "candidate_strength"], ascending=[True, True, False])
    changes.to_csv(output_dir / "e1_adjacent_checkpoint_changes.csv", index=False)
    candidates.to_csv(output_dir / "e1_boundary_candidates_by_metric.csv", index=False)

    # A compact consensus table: count how often each transition is the strongest candidate.
    if not candidates.empty:
        consensus = (
            candidates.groupby(["model", "transition", "step"], as_index=False)
            .agg(
                n_metric_module_votes=("metric", "count"),
                mean_candidate_strength=("candidate_strength", "mean"),
                modules=("module", lambda x: ";".join(sorted(set(map(str, x))))),
                metrics=("metric", lambda x: ";".join(sorted(set(map(str, x))))),
            )
            .sort_values(["model", "n_metric_module_votes", "mean_candidate_strength"], ascending=[True, False, False])
        )
        consensus.to_csv(output_dir / "e1_boundary_consensus_table.csv", index=False)
    return changes, candidates


def plot_metric_lines(df: pd.DataFrame, metrics: list[str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        plot_df = df.groupby(["model", "module", "step"], as_index=False)[metric].mean(numeric_only=True)
        if plot_df.empty:
            continue
        for log_x in [False, True]:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            for (model, module), sub in plot_df.groupby(["model", "module"]):
                sub = sub.sort_values("step")
                label = f"{model.split('/')[-1]} | {module}"
                ax.plot(sub["step"], sub[metric], marker="o", linewidth=1.5, markersize=3.5, label=label)
            if log_x:
                ax.set_xscale("symlog", linthresh=1)
                suffix = "logstep"
                title_suffix = "log-step axis"
            else:
                suffix = "linear"
                title_suffix = "linear-step axis"
            ax.set_xlabel("Training step")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"E1 {metric.replace('_', ' ')} ({title_suffix})")
            ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            out = output_dir / f"e1_{metric}_{suffix}.png"
            fig.savefig(out, dpi=220)
            plt.close(fig)
            written.append(out)
    return written


def plot_change_heatmap(changes: pd.DataFrame, output_path: Path) -> Path | None:
    if changes.empty:
        return None
    # Use mean candidate strength by transition/metric over modules/models as a compact diagnostic.
    h = (
        changes.groupby(["metric", "transition"], as_index=False)["candidate_strength"]
        .mean()
        .pivot(index="metric", columns="transition", values="candidate_strength")
    )
    if h.empty:
        return None
    h = h.reindex(sorted(h.columns, key=lambda s: int(s.split("→")[-1])), axis=1)
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(h.columns)), max(3.5, 0.45 * len(h.index))))
    im = ax.imshow(h.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(h.columns)))
    ax.set_xticklabels(h.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(h.index)))
    ax.set_yticklabels(h.index)
    ax.set_title("E1 candidate transition strength by metric")
    fig.colorbar(im, ax=ax, label="mean candidate strength")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def write_markdown_report(
    df: pd.DataFrame,
    completeness: pd.DataFrame,
    candidates: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = ", ".join(sorted(map(str, df["model"].unique())))
    steps = sorted(int(x) for x in df["step"].unique())
    n_rows = len(df)
    n_mats = df["matrix_name"].nunique()
    n_modules = df["module"].nunique()
    n_layers = df["layer"].nunique()

    top_lines = []
    if not candidates.empty:
        top = candidates.sort_values("candidate_strength", ascending=False).head(12)
        for _, r in top.iterrows():
            top_lines.append(
                f"- `{r['model']}` / `{r['module']}` / `{r['metric']}`: "
                f"strongest adjacent signal at `{r['transition']}` "
                f"(strength={r['candidate_strength']:.4g}, direction={r['direction']})."
            )
    else:
        top_lines.append("- No candidate table could be computed; check metric availability.")

    text = f"""# E1 phase-identification artifact report

This report is generated automatically from `e1_spectral_metrics.csv`. It is a diagnostic artifact, not a final interpretation. Candidate boundaries should be accepted only after visual inspection, threshold sensitivity checks, and replication across model sizes.

## Run coverage

- Models: {models}
- Steps: {steps}
- Metric rows: {n_rows:,}
- Unique matrices: {n_mats:,}
- Unique modules: {n_modules:,}
- Unique layers: {n_layers:,}
- Metrics summarized: {', '.join(metrics)}

## Completeness checks

The table `tables/e1_checkpoint_completeness.csv` records how many matrices, layers, and modules were measured at each checkpoint. Before interpreting phase boundaries, confirm that each checkpoint has the expected number of matrices.

## Strongest adjacent-change candidates

{chr(10).join(top_lines)}

## Interpretation protocol

1. Inspect both linear-step and log/symlog-step plots. A transition that appears only on a log axis should not be treated as a sharp phase boundary.
2. Check whether multiple independent indicators agree: stable/effective-rank change, subspace-stability drop, and MP-outlier emergence are stronger together than any single metric alone.
3. Check module consistency. A global phase claim requires several modules/layers to show the same broad transition; otherwise the result should be phrased as module-specific.
4. Check model-size replication. The initial 70M result is a pipeline validation; the thesis claim should rely on replication across 70M/160M and, if feasible, 410M.
5. Treat norm growth as corroborative only. Norms often move monotonically and are not by themselves evidence of a developmental phase.

## Thesis use

These artifacts support Chapter 4, E1: phase identification and validation. They should not yet be used to claim a critical period. The critical/sensitive-period claim requires E3: behavioural durability after checkpoint-specific intervention.
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E1 spectral metrics and generate thesis artifacts.")
    parser.add_argument(
        "--metrics",
        default="results/e1_phase_identification/raw/e1_spectral_metrics.csv",
        help="Path to raw E1 metrics CSV.",
    )
    parser.add_argument(
        "--output-root",
        default="results/e1_phase_identification",
        help="E1 output root directory.",
    )
    parser.add_argument(
        "--manifest",
        default="results/e1_phase_identification/manifests/artifact_manifest.csv",
        help="Artifact manifest CSV to append to.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)

    df = _safe_read_csv(metrics_path)
    metrics = available_metrics(df, DEFAULT_METRICS)
    boundary_metrics = available_metrics(df, BOUNDARY_METRICS)
    if not metrics:
        raise ValueError("No known numeric E1 metrics found in the CSV.")

    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    processed_dir = output_root / "processed"
    for d in [tables_dir, figures_dir, reports_dir, processed_dir, manifest_path.parent]:
        d.mkdir(parents=True, exist_ok=True)

    completeness_path = tables_dir / "e1_checkpoint_completeness.csv"
    summary_path = tables_dir / "e1_metric_summary_by_model_step_module.csv"
    completeness = write_completeness_table(df, completeness_path)
    summary = write_metric_summary(df, metrics, summary_path)

    changes, candidates = write_change_tables(df, boundary_metrics, processed_dir)
    figure_paths = plot_metric_lines(df, metrics, figures_dir)
    heatmap = plot_change_heatmap(changes, figures_dir / "e1_candidate_transition_strength_heatmap.png")
    if heatmap is not None:
        figure_paths.append(heatmap)

    report_path = reports_dir / "e1_validation_report.md"
    write_markdown_report(df, completeness, candidates, metrics, report_path)

    _artifact(manifest_path, "table", completeness_path, "Completeness check for E1 spectral measurements by model and checkpoint.")
    _artifact(manifest_path, "table", summary_path, "Model/module/checkpoint summary of E1 weight-spectral indicators.")
    _artifact(manifest_path, "processed_csv", processed_dir / "e1_adjacent_checkpoint_changes.csv", "Adjacent-checkpoint changes used to inspect candidate phase transitions.")
    _artifact(manifest_path, "processed_csv", processed_dir / "e1_boundary_candidates_by_metric.csv", "Strongest candidate transition per metric and module.")
    consensus_path = processed_dir / "e1_boundary_consensus_table.csv"
    if consensus_path.exists():
        _artifact(manifest_path, "processed_csv", consensus_path, "Compact consensus table counting candidate-boundary votes across metrics and modules.")
    for p in figure_paths:
        _artifact(manifest_path, "figure", p, f"E1 diagnostic figure: {p.stem.replace('_', ' ')}.")
    _artifact(manifest_path, "report", report_path, "Automatically generated E1 validation report for thesis planning.")

    print(f"Loaded {len(df):,} rows from {metrics_path}")
    print(f"Wrote completeness table: {completeness_path}")
    print(f"Wrote summary table: {summary_path}")
    print(f"Wrote processed change tables under: {processed_dir}")
    print(f"Wrote {len(figure_paths)} figures under: {figures_dir}")
    print(f"Wrote validation report: {report_path}")
    print(f"Updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
