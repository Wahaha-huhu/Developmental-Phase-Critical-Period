#!/usr/bin/env python3
"""
Plot dense E1 weight-spectral indicators.

Purpose
-------
This is a standalone plotting utility for the Step-0/Step-2 artifact and
mechanism figures. It reads a dense E1 spectral-metrics CSV and produces:

1. Dense indicator overview curves over checkpoint step.
2. Stable/effective rank attention-vs-MLP comparisons.
3. Per-module-family curves for all available indicators.
4. Layerwise heatmaps for selected indicators and module families.
5. A compact machine-readable summary table used for thesis plotting.

The script is schema-tolerant: it accepts either long-format metrics
(metric,value columns) or wide-format metrics (stable_rank/effective_rank/etc.
as columns).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INDICATORS = [
    "stable_rank",
    "effective_rank",
    "spectral_norm",
    "frobenius_norm",
    "mp_outliers",
    "alpha",
    "sv_stability",
    "subspace_stability",
]

CANONICAL_ALIASES = {
    "stable_rank": ["stable_rank", "stable-rank", "srank"],
    "effective_rank": ["effective_rank", "effective-rank", "erank", "entropy_rank"],
    "spectral_norm": ["spectral_norm", "spectral-norm", "operator_norm", "top_singular", "sigma_max"],
    "frobenius_norm": ["frobenius_norm", "fro_norm", "frob_norm", "frobenius"],
    "mp_outliers": ["mp_outliers", "mp_outlier_count", "n_mp_outliers", "outlier_count"],
    "alpha": ["alpha", "powerlaw_alpha", "power_law_alpha"],
    "sv_stability": ["sv_stability", "singular_vector_stability", "subspace_similarity"],
    "subspace_stability": ["subspace_stability", "sv_subspace_stability", "topk_subspace_stability"],
}


def parse_step(value) -> int:
    """Parse step strings like step1000, global_step_1000, or integers."""
    if pd.isna(value):
        return -1
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    s = str(value)
    m = re.search(r"(\d+)", s)
    if not m:
        return -1
    return int(m.group(1))


def choose_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def canonical_metric_name(name: str) -> str:
    n = str(name).strip()
    nl = n.lower()
    for canon, aliases in CANONICAL_ALIASES.items():
        if nl == canon.lower() or nl in [a.lower() for a in aliases]:
            return canon
    return n


def infer_module_family(module: str) -> str:
    s = str(module).lower()
    if any(x in s for x in ["mlp", "dense_h_to_4h", "dense_4h_to_h"]):
        return "MLP"
    if any(x in s for x in ["attention", "attn", "query_key_value", "qkv", "query", "key", "value"]):
        return "Attention"
    if any(x in s for x in ["embed", "wte", "wpe"]):
        return "Embedding"
    return "Other"


def infer_module_role(module: str) -> str:
    s = str(module).lower()
    if "query_key_value" in s or "qkv" in s:
        return "QKV"
    if "attention.dense" in s or "attn.dense" in s or "out_proj" in s:
        return "Attention output"
    if "dense_h_to_4h" in s or "fc_in" in s or "up_proj" in s or "gate_proj" in s:
        return "MLP in"
    if "dense_4h_to_h" in s or "fc_out" in s or "down_proj" in s:
        return "MLP out"
    if "mlp" in s:
        return "MLP other"
    if "attention" in s or "attn" in s:
        return "Attention other"
    return str(module)


def normalize_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns: step, step_num, layer, module, module_family, module_role, metric, value."""
    df = df.copy()

    step_col = choose_col(df.columns, ["step", "checkpoint", "revision", "stage"])
    step_num_col = choose_col(df.columns, ["step_num", "step_number", "global_step"])
    layer_col = choose_col(df.columns, ["layer", "layer_idx", "layer_id", "block", "block_idx"])
    module_col = choose_col(df.columns, ["module", "module_name", "matrix", "name", "module_suffix", "param_name"])
    metric_col = choose_col(df.columns, ["metric", "indicator", "measure", "stat"])
    value_col = choose_col(df.columns, ["value", "metric_value", "score"])

    if step_num_col is None:
        if step_col is None:
            raise ValueError("Could not infer a step/checkpoint column. Expected one of step, checkpoint, revision, stage, or step_num.")
        df["step_num"] = df[step_col].map(parse_step)
        step_num_col = "step_num"
    else:
        df["step_num"] = df[step_num_col].map(parse_step)
        step_num_col = "step_num"

    if step_col is None:
        df["step"] = df["step_num"].map(lambda x: f"step{x}")
        step_col = "step"

    if layer_col is None:
        df["layer"] = 0
        layer_col = "layer"

    if module_col is None:
        df["module"] = "unknown"
        module_col = "module"

    id_cols = [step_col, step_num_col, layer_col, module_col]

    if metric_col is not None and value_col is not None:
        out = df[id_cols + [metric_col, value_col]].copy()
        out.columns = ["step", "step_num", "layer", "module", "metric", "value"]
    else:
        # Wide-format fallback. Melt any numeric column that looks metric-like and is not an id column.
        id_set = set(id_cols)
        numeric_cols = [c for c in df.columns if c not in id_set and pd.api.types.is_numeric_dtype(df[c])]
        metric_like = []
        for c in numeric_cols:
            cl = c.lower()
            if any(alias in cl for aliases in CANONICAL_ALIASES.values() for alias in aliases):
                metric_like.append(c)
        if not metric_like:
            # Conservative fallback: all numeric non-id columns except obvious metadata.
            banned = {"seed", "rank", "n", "count"}
            metric_like = [c for c in numeric_cols if c.lower() not in banned]
        if not metric_like:
            raise ValueError("Could not infer metric columns. Expected metric/value columns or wide numeric metric columns.")
        out = df[id_cols + metric_like].melt(
            id_vars=id_cols,
            value_vars=metric_like,
            var_name="metric",
            value_name="value",
        )
        out.columns = ["step", "step_num", "layer", "module", "metric", "value"]

    out["metric"] = out["metric"].map(canonical_metric_name)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["layer"] = out["layer"].map(parse_step)
    out["module"] = out["module"].astype(str)
    out["module_family"] = out["module"].map(infer_module_family)
    out["module_role"] = out["module"].map(infer_module_role)
    out = out[np.isfinite(out["step_num"]) & np.isfinite(out["value"])].copy()
    out = out[out["step_num"] >= 0].copy()
    return out


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "figures": root / "figures",
        "tables": root / "tables",
        "reports": root / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def ordered_steps(df: pd.DataFrame) -> List[int]:
    return sorted(int(x) for x in df["step_num"].dropna().unique())


def maybe_log_x(ax, use_log: bool) -> None:
    if use_log:
        # symlog keeps step0 visible if present while making early checkpoints readable.
        ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Checkpoint step")
    ax.grid(True, which="both", alpha=0.25)


def plot_indicator_overview(long: pd.DataFrame, out_dir: Path, indicators: Sequence[str], use_log: bool) -> Path:
    present = [m for m in indicators if m in set(long["metric"])]
    if not present:
        raise ValueError("No requested indicators found in metric table.")

    n = len(present)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    summary = (
        long[long["metric"].isin(present)]
        .groupby(["step_num", "metric"], as_index=False)["value"]
        .mean()
        .sort_values(["metric", "step_num"])
    )

    for ax, metric in zip(axes_flat, present):
        sub = summary[summary["metric"] == metric]
        ax.plot(sub["step_num"], sub["value"], marker="o", linewidth=1.5, markersize=3)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Mean over layers/modules")
        maybe_log_x(ax, use_log)

    for ax in axes_flat[len(present):]:
        ax.axis("off")

    fig.suptitle("Dense E1 spectral indicators over training", y=1.0, fontsize=14)
    fig.tight_layout()
    suffix = "logx" if use_log else "linear"
    path = out_dir / f"e1_dense_indicator_overview_{suffix}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_family_comparison(long: pd.DataFrame, out_dir: Path, metrics: Sequence[str], use_log: bool) -> Path:
    families = ["Attention", "MLP"]
    present = [m for m in metrics if m in set(long["metric"])]
    if not present:
        raise ValueError("No requested rank metrics found for family comparison.")

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.4), squeeze=False)
    axes_flat = axes.ravel()

    summary = (
        long[long["metric"].isin(present) & long["module_family"].isin(families)]
        .groupby(["step_num", "metric", "module_family"], as_index=False)["value"]
        .mean()
        .sort_values(["metric", "module_family", "step_num"])
    )

    for ax, metric in zip(axes_flat, present):
        for fam in families:
            sub = summary[(summary["metric"] == metric) & (summary["module_family"] == fam)]
            if len(sub):
                ax.plot(sub["step_num"], sub["value"], marker="o", linewidth=1.8, markersize=3, label=fam)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Mean over layers/modules")
        maybe_log_x(ax, use_log)
        ax.legend(frameon=False)

    fig.suptitle("Attention versus MLP rank dynamics", y=1.0, fontsize=14)
    fig.tight_layout()
    suffix = "logx" if use_log else "linear"
    path = out_dir / f"e1_attention_vs_mlp_rank_{suffix}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_module_role_curves(long: pd.DataFrame, out_dir: Path, indicators: Sequence[str], use_log: bool) -> List[Path]:
    paths: List[Path] = []
    present = [m for m in indicators if m in set(long["metric"])]
    roles = [r for r in ["QKV", "Attention output", "MLP in", "MLP out"] if r in set(long["module_role"])]

    for metric in present:
        summary = (
            long[(long["metric"] == metric) & long["module_role"].isin(roles)]
            .groupby(["step_num", "module_role"], as_index=False)["value"]
            .mean()
            .sort_values(["module_role", "step_num"])
        )
        if summary.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for role in roles:
            sub = summary[summary["module_role"] == role]
            if len(sub):
                ax.plot(sub["step_num"], sub["value"], marker="o", linewidth=1.6, markersize=3, label=role)
        ax.set_title(f"{metric.replace('_', ' ').title()} by module role")
        ax.set_ylabel("Mean over layers")
        maybe_log_x(ax, use_log)
        ax.legend(frameon=False)
        fig.tight_layout()
        suffix = "logx" if use_log else "linear"
        path = out_dir / f"e1_module_role_{metric}_{suffix}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_heatmap_matrix(sub: pd.DataFrame) -> pd.DataFrame:
    heat = (
        sub.groupby(["layer", "step_num"], as_index=False)["value"]
        .mean()
        .pivot(index="layer", columns="step_num", values="value")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return heat


def plot_layer_heatmaps(long: pd.DataFrame, out_dir: Path, metrics: Sequence[str], families: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    present = [m for m in metrics if m in set(long["metric"])]
    for metric in present:
        for fam in families:
            sub = long[(long["metric"] == metric) & (long["module_family"] == fam)]
            if sub.empty:
                continue
            heat = make_heatmap_matrix(sub)
            if heat.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, 4.8))
            im = ax.imshow(heat.values, aspect="auto", origin="lower", interpolation="nearest")
            ax.set_title(f"Layerwise {metric.replace('_', ' ')} — {fam}")
            ax.set_ylabel("Layer")
            ax.set_xlabel("Checkpoint step")

            y_labels = list(heat.index)
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels([str(int(y)) for y in y_labels])

            x_steps = list(heat.columns)
            # Show at most 12 x tick labels.
            if len(x_steps) <= 12:
                tick_idx = list(range(len(x_steps)))
            else:
                tick_idx = sorted(set(np.linspace(0, len(x_steps) - 1, 12).round().astype(int).tolist()))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(int(x_steps[i])) for i in tick_idx], rotation=45, ha="right")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            fig.tight_layout()
            safe_fam = fam.lower().replace(" ", "_")
            path = out_dir / f"e1_layer_heatmap_{metric}_{safe_fam}.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def write_summary_tables(long: pd.DataFrame, tables_dir: Path, indicators: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    present = [m for m in indicators if m in set(long["metric"])]

    overall = (
        long[long["metric"].isin(present)]
        .groupby(["step_num", "metric"], as_index=False)
        .agg(value_mean=("value", "mean"), value_sem=("value", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0), n=("value", "size"))
        .sort_values(["metric", "step_num"])
    )
    p = tables_dir / "e1_indicator_overall_summary.csv"
    overall.to_csv(p, index=False)
    paths.append(p)

    family = (
        long[long["metric"].isin(present)]
        .groupby(["step_num", "metric", "module_family"], as_index=False)
        .agg(value_mean=("value", "mean"), value_sem=("value", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0), n=("value", "size"))
        .sort_values(["metric", "module_family", "step_num"])
    )
    p = tables_dir / "e1_indicator_by_module_family.csv"
    family.to_csv(p, index=False)
    paths.append(p)

    layer = (
        long[long["metric"].isin(present)]
        .groupby(["step_num", "metric", "module_family", "layer"], as_index=False)
        .agg(value_mean=("value", "mean"), n=("value", "size"))
        .sort_values(["metric", "module_family", "layer", "step_num"])
    )
    p = tables_dir / "e1_indicator_by_layer_family.csv"
    layer.to_csv(p, index=False)
    paths.append(p)
    return paths


def write_report(root: Path, paths: Sequence[Path], long: pd.DataFrame, indicators: Sequence[str]) -> Path:
    report = root / "reports" / "e1_dense_indicator_panel_report.md"
    present = [m for m in indicators if m in set(long["metric"])]
    steps = ordered_steps(long)
    families = sorted(long["module_family"].dropna().unique().tolist())
    roles = sorted(long["module_role"].dropna().unique().tolist())
    lines = [
        "# E1 dense indicator panel report",
        "",
        f"- Rows after normalisation: {len(long):,}",
        f"- Checkpoints: {len(steps)} ({steps[0] if steps else 'NA'} → {steps[-1] if steps else 'NA'})",
        f"- Indicators plotted: {', '.join(present)}",
        f"- Module families: {', '.join(families)}",
        f"- Module roles: {', '.join(roles)}",
        "",
        "## Generated artifacts",
    ]
    for path in paths:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        lines.append(f"- `{rel}`")
    lines.extend([
        "",
        "## Intended thesis use",
        "",
        "- Main text: use the compact attention-vs-MLP rank comparison when discussing module-level differences.",
        "- Appendix: use the full dense overview, per-module-role curves, and layerwise heatmaps.",
        "- Avoid treating norm-only plots as primary evidence for phase boundaries; use them as corroborative diagnostics.",
    ])
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dense E1 indicator panels by module family and layer.")
    parser.add_argument("--e1-metrics", required=True, help="Path to dense E1 spectral metrics CSV.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--indicators", nargs="*", default=DEFAULT_INDICATORS, help="Indicators to plot if present.")
    parser.add_argument("--heatmap-indicators", nargs="*", default=["stable_rank", "effective_rank", "sv_stability", "subspace_stability"], help="Indicators for layer heatmaps if present.")
    parser.add_argument("--exclude-step0", action="store_true", help="Exclude step0 from plots/tables.")
    args = parser.parse_args()

    root = Path(args.out)
    dirs = ensure_dirs(root)

    raw = pd.read_csv(args.e1_metrics)
    long = normalize_to_long(raw)
    if args.exclude_step0:
        long = long[long["step_num"] != 0].copy()

    # Save normalised long table for debugging/reuse.
    norm_path = dirs["tables"] / "e1_metrics_long_normalized.csv"
    long.to_csv(norm_path, index=False)

    generated: List[Path] = [norm_path]
    generated += write_summary_tables(long, dirs["tables"], args.indicators)

    for use_log in [True, False]:
        generated.append(plot_indicator_overview(long, dirs["figures"], args.indicators, use_log=use_log))
        try:
            generated.append(plot_family_comparison(long, dirs["figures"], ["stable_rank", "effective_rank"], use_log=use_log))
        except ValueError:
            pass
        generated.extend(plot_module_role_curves(long, dirs["figures"], args.indicators, use_log=use_log))

    generated.extend(plot_layer_heatmaps(long, dirs["figures"], args.heatmap_indicators, ["Attention", "MLP"]))
    report = write_report(root, generated, long, args.indicators)
    generated.append(report)

    print(f"Wrote E1 dense indicator panels to {root}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
