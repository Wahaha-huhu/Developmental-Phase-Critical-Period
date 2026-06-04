#!/usr/bin/env python3
"""Analyze E1 spectral-metric outputs and create thesis-ready tables/reports.

This script is intentionally self-contained and tolerant of small schema changes in
`e1_spectral_metrics.csv`. It expects at least: model, checkpoint, step, module,
layer, matrix_name, plus any subset of spectral metric columns.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_METRICS = [
    "frobenius_norm",
    "spectral_norm",
    "stable_rank",
    "effective_rank",
    "nuclear_norm",
    "mp_outliers_x1",
    "mp_outliers_x1.1",
    "mp_outliers_x1.25",
    "alpha_tail_frac_0.2",
    "alpha_tail_frac_0.3",
    "alpha_tail_frac_0.5",
    "subspace_stability_topk",
]

BOUNDARY_METRICS = [
    "stable_rank",
    "effective_rank",
    "mp_outliers_x1",
    "mp_outliers_x1.1",
    "mp_outliers_x1.25",
    "alpha_tail_frac_0.3",
    "subspace_stability_topk",
]


def _safe_name(s: str) -> str:
    return str(s).replace("/", "__").replace(" ", "_").replace(".", "_")


def _metric_cols(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _mkdirs(root: Path) -> dict[str, Path]:
    paths = {
        "tables": root / "tables",
        "processed": root / "processed",
        "reports": root / "reports",
        "figures": root / "figures",
        "manifests": root / "manifests",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _infer_output_root(metrics_path: Path) -> Path:
    # Common layout: results/e1_phase_identification/raw/e1_spectral_metrics.csv
    if metrics_path.parent.name == "raw":
        return metrics_path.parent.parent
    return metrics_path.parent


def write_artifact_manifest(root: Path, artifacts: list[dict]) -> None:
    manifest = root / "manifests" / "artifact_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    existing = []
    if manifest.exists():
        with manifest.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
    existing_keys = {(row.get("path"), row.get("artifact_type")) for row in existing}
    new_rows = [a for a in artifacts if (a.get("path"), a.get("artifact_type")) not in existing_keys]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in existing + new_rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def checkpoint_completeness(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model", "checkpoint", "step"], dropna=False)
        .agg(
            n_rows=("matrix_name", "size"),
            n_matrices=("matrix_name", "nunique"),
            n_layers=("layer", "nunique"),
            n_modules=("module", "nunique"),
        )
        .reset_index()
        .sort_values(["model", "step"])
    )
    return agg


def module_checkpoint_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg_spec = {}
    for m in metrics:
        agg_spec[f"{m}_mean"] = (m, "mean")
        agg_spec[f"{m}_std"] = (m, "std")
    out = (
        df.groupby(["model", "checkpoint", "step", "module"], dropna=False)
        .agg(n_layers=("layer", "nunique"), n_matrices=("matrix_name", "nunique"), **agg_spec)
        .reset_index()
        .sort_values(["model", "module", "step"])
    )
    return out


def adjacent_changes(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (model, module), g in summary.groupby(["model", "module"], dropna=False):
        g = g.sort_values("step")
        for m in metrics:
            col = f"{m}_mean"
            if col not in g.columns:
                continue
            vals = g[["checkpoint", "step", col]].dropna().reset_index(drop=True)
            for i in range(1, len(vals)):
                prev = vals.loc[i - 1]
                cur = vals.loc[i]
                v0 = float(prev[col])
                v1 = float(cur[col])
                delta = v1 - v0
                rel_delta = np.nan if abs(v0) < 1e-12 else delta / v0
                # Symmetric change score for scale-sensitive metrics. For subspace stability,
                # the value itself is an adjacent-checkpoint similarity, so low value = high change.
                if m == "subspace_stability_topk":
                    score = max(0.0, 1.0 - v1) if np.isfinite(v1) else np.nan
                else:
                    score = abs(np.log((abs(v1) + 1e-12) / (abs(v0) + 1e-12)))
                rows.append(
                    {
                        "model": model,
                        "module": module,
                        "metric": m,
                        "from_checkpoint": prev["checkpoint"],
                        "to_checkpoint": cur["checkpoint"],
                        "from_step": int(prev["step"]),
                        "to_step": int(cur["step"]),
                        "value_before": v0,
                        "value_after": v1,
                        "delta": delta,
                        "relative_delta": rel_delta,
                        "transition_score": score,
                    }
                )
    return pd.DataFrame(rows).sort_values(["model", "module", "metric", "from_step"])


def candidate_boundaries(changes: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
    if changes.empty:
        return changes
    rows = []
    for (model, module, metric), g in changes.groupby(["model", "module", "metric"], dropna=False):
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["transition_score"])
        if g.empty:
            continue
        top = g.sort_values("transition_score", ascending=False).head(top_n).copy()
        top["rank_within_metric_module"] = range(1, len(top) + 1)
        rows.append(top)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["model", "rank_within_metric_module", "from_step", "module", "metric"])


def boundary_consensus(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = (
        candidates.groupby(["model", "from_step", "to_step"], dropna=False)
        .agg(
            n_votes=("metric", "size"),
            n_metrics=("metric", "nunique"),
            n_modules=("module", "nunique"),
            total_transition_score=("transition_score", "sum"),
            mean_transition_score=("transition_score", "mean"),
            metrics=("metric", lambda x: "; ".join(sorted(set(map(str, x))))),
            modules=("module", lambda x: "; ".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .sort_values(["model", "n_votes", "n_metrics", "n_modules", "total_transition_score"], ascending=[True, False, False, False, False])
    )
    return out


def write_report(root: Path, df: pd.DataFrame, completeness: pd.DataFrame, consensus: pd.DataFrame, candidates: pd.DataFrame) -> Path:
    report = root / "reports" / "e1_validation_report.md"
    lines = []
    lines.append("# E1 phase-identification validation report\n")
    lines.append("## Data coverage\n")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Models: {df['model'].nunique() if 'model' in df else 'unknown'}")
    lines.append(f"- Checkpoints: {df['checkpoint'].nunique() if 'checkpoint' in df else 'unknown'}")
    lines.append(f"- Layers: {df['layer'].nunique() if 'layer' in df else 'unknown'}")
    lines.append(f"- Modules: {df['module'].nunique() if 'module' in df else 'unknown'}\n")

    if not completeness.empty:
        lines.append("## Checkpoint completeness\n")
        for _, row in completeness.iterrows():
            lines.append(
                f"- `{row['model']}` `{row['checkpoint']}` (step {int(row['step'])}): "
                f"{int(row['n_rows'])} rows, {int(row['n_layers'])} layers, {int(row['n_modules'])} modules."
            )
        lines.append("")

    lines.append("## Candidate boundary consensus\n")
    if consensus.empty:
        lines.append("No candidate boundaries could be computed from the available metrics.\n")
    else:
        top = consensus.head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['model']}` {int(row['from_step'])}→{int(row['to_step'])}: "
                f"{int(row['n_votes'])} votes across {int(row['n_metrics'])} metrics and "
                f"{int(row['n_modules'])} modules. Metrics: {row['metrics']}. Modules: {row['modules']}."
            )
        lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- This report is observational. It supports phase identification only, not the causal sensitive-period claim.")
    lines.append("- A strong E1 result requires multiple independent indicators to concentrate around similar checkpoint intervals.")
    lines.append("- Module-staggered boundaries should be retained as an empirical finding, not smoothed into a single global transition.")
    lines.append("- The next validation steps are model-size replication and behavioural grounding.\n")

    if not candidates.empty:
        lines.append("## Files generated\n")
        lines.append("- `tables/e1_checkpoint_completeness.csv`")
        lines.append("- `tables/e1_metric_summary_by_checkpoint.csv`")
        lines.append("- `processed/e1_adjacent_checkpoint_changes.csv`")
        lines.append("- `processed/e1_candidate_boundaries.csv`")
        lines.append("- `processed/e1_boundary_consensus_table.csv`")
        lines.append("- `figures/e1_candidate_transition_strength_heatmap.png` if matplotlib is available")

    report.write_text("\n".join(lines))
    return report


def plot_heatmap(root: Path, candidates: pd.DataFrame) -> Path | None:
    if plt is None or candidates.empty:
        return None
    # Use all candidate scores, aggregated by module and transition interval.
    c = candidates.copy()
    c["transition"] = c["from_step"].astype(int).astype(str) + "→" + c["to_step"].astype(int).astype(str)
    pivot = c.pivot_table(index="module", columns="transition", values="transition_score", aggfunc="sum", fill_value=0.0)
    # Sort transitions by numeric from step.
    ordered = sorted(pivot.columns, key=lambda s: int(s.split("→")[0]))
    pivot = pivot[ordered]

    fig_w = max(8, 0.7 * len(pivot.columns) + 3)
    fig_h = max(3.5, 0.55 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("E1 candidate transition strength by module")
    ax.set_xlabel("Adjacent checkpoint interval")
    ax.set_ylabel("Module")
    fig.colorbar(im, ax=ax, label="summed transition score")
    fig.tight_layout()
    out = root / "figures" / "e1_candidate_transition_strength_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E1 spectral metrics and produce tables/reports.")
    parser.add_argument("--metrics", required=True, help="Path to e1_spectral_metrics.csv")
    parser.add_argument("--output-root", default=None, help="Output root; defaults to parent of raw/ folder")
    parser.add_argument("--top-n", type=int, default=2, help="Top candidate intervals per metric/module")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    root = Path(args.output_root) if args.output_root else _infer_output_root(metrics_path)
    _mkdirs(root)

    df = pd.read_csv(metrics_path)
    if "step" not in df.columns and "checkpoint" in df.columns:
        df["step"] = df["checkpoint"].astype(str).str.extract(r"(\d+)").astype(int)
    df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)

    metrics = _metric_cols(df, DEFAULT_METRICS)
    boundary_metrics = _metric_cols(df, BOUNDARY_METRICS)
    if not metrics:
        raise ValueError("No recognised numeric metric columns found in the CSV.")

    artifacts: list[dict] = []
    def add_artifact(path: Path, artifact_type: str, caption: str, notes: str = "") -> None:
        artifacts.append(
            {
                "experiment_id": "e1_phase_identification",
                "artifact_type": artifact_type,
                "path": str(path.relative_to(root.parent) if root.parent in path.parents else path),
                "thesis_section": "Chapter 4, E1 phase identification",
                "caption_draft": caption,
                "source_data": str(metrics_path),
                "code_entrypoint": "scripts/analyze_e1_results.py",
                "status": "generated",
                "notes": notes,
            }
        )

    comp = checkpoint_completeness(df)
    comp_path = root / "tables" / "e1_checkpoint_completeness.csv"
    comp.to_csv(comp_path, index=False)
    add_artifact(comp_path, "table", "Completeness of processed E1 checkpoints, layers, and modules.")

    summary = module_checkpoint_summary(df, metrics)
    summary_path = root / "tables" / "e1_metric_summary_by_checkpoint.csv"
    summary.to_csv(summary_path, index=False)
    add_artifact(summary_path, "table", "Module-level spectral metric summaries by checkpoint.")

    changes = adjacent_changes(summary, boundary_metrics)
    changes_path = root / "processed" / "e1_adjacent_checkpoint_changes.csv"
    changes.to_csv(changes_path, index=False)
    add_artifact(changes_path, "processed_data", "Adjacent-checkpoint changes in E1 spectral indicators.")

    candidates = candidate_boundaries(changes, top_n=args.top_n)
    cand_path = root / "processed" / "e1_candidate_boundaries.csv"
    candidates.to_csv(cand_path, index=False)
    add_artifact(cand_path, "processed_data", "Top candidate transition intervals per metric and module.")

    consensus = boundary_consensus(candidates)
    consensus_path = root / "processed" / "e1_boundary_consensus_table.csv"
    consensus.to_csv(consensus_path, index=False)
    add_artifact(consensus_path, "table", "Consensus table aggregating candidate transition intervals across indicators and modules.")

    fig_path = plot_heatmap(root, candidates)
    if fig_path is not None:
        add_artifact(fig_path, "figure", "Candidate transition strength across modules and adjacent checkpoint intervals.")

    report_path = write_report(root, df, comp, consensus, candidates)
    add_artifact(report_path, "report", "Automated validation report for E1 phase identification.")

    write_artifact_manifest(root, artifacts)

    print(f"Loaded {len(df):,} rows from {metrics_path}")
    print(f"Metrics analysed: {', '.join(metrics)}")
    print(f"Wrote outputs under: {root}")
    if not consensus.empty:
        print("\nTop consensus intervals:")
        print(consensus.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
