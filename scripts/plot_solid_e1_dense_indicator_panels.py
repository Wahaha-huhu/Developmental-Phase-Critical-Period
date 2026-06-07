#!/usr/bin/env python3
"""Plot dense E1 indicator panels into solid_results/.

This script is intentionally schema-tolerant: it accepts either a wide E1 metrics CSV
(one column per indicator) or a long CSV with columns like metric/indicator and value.
It produces appendix-ready plots for overall indicators, attention-vs-MLP rank curves,
module-role curves, and layerwise heatmaps.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_step(x) -> int:
    if pd.isna(x):
        return -1
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x)
    m = re.search(r"step(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def infer_layer(row) -> int:
    for key in ["layer", "layer_idx", "layer_id"]:
        if key in row and not pd.isna(row[key]):
            try:
                return int(row[key])
            except Exception:
                pass
    text = " ".join(str(row.get(k, "")) for k in ["name", "module", "matrix", "param", "parameter"])
    for pat in [r"layers?\.(\d+)", r"gpt_neox\.layers\.(\d+)", r"h\.(\d+)"]:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return -1


def infer_module_role(text: str) -> str:
    t = str(text).lower()
    if "query_key_value" in t or "qkv" in t or "attn.q" in t or "attention.query" in t:
        return "attention_qkv"
    if "attention.dense" in t or "attn.dense" in t or "attention.out" in t or "o_proj" in t:
        return "attention_out"
    if "dense_h_to_4h" in t or "mlp.dense_h" in t or "up_proj" in t or "gate_proj" in t:
        return "mlp_in"
    if "dense_4h_to_h" in t or "mlp.dense_4h" in t or "down_proj" in t:
        return "mlp_out"
    if "attention" in t or "attn" in t:
        return "attention_other"
    if "mlp" in t or "feed" in t:
        return "mlp_other"
    return "other"


def family_from_role(role: str) -> str:
    if role.startswith("attention"):
        return "attention"
    if role.startswith("mlp"):
        return "mlp"
    return "other"


def canonical_indicator(c: str) -> Optional[str]:
    low = c.lower()
    aliases = {
        "stable_rank": ["stable_rank", "srank"],
        "effective_rank": ["effective_rank", "eff_rank", "erank", "entropy_rank"],
        "spectral_norm": ["spectral_norm", "spec_norm", "top_singular"],
        "frobenius_norm": ["frobenius_norm", "frob_norm", "fro_norm"],
        "mp_outliers": ["mp_outliers", "marchenko", "mp_count", "outlier_count"],
        "alpha": ["alpha", "powerlaw_alpha", "pl_alpha"],
        "sv_stability": ["sv_stability", "subspace_stability", "singular_vector", "sv_overlap", "cosine_stability"],
    }
    for name, pats in aliases.items():
        if any(p in low for p in pats):
            return name
    return None


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize common columns.
    if "step_num" not in df.columns:
        for cand in ["step", "checkpoint", "stage", "revision"]:
            if cand in df.columns:
                df["step_num"] = df[cand].map(parse_step)
                break
    if "step" not in df.columns:
        df["step"] = df["step_num"].map(lambda x: f"step{int(x)}" if int(x) >= 0 else "unknown")

    # Determine long/wide format.
    metric_col = None
    for c in ["indicator", "metric", "metric_name", "name"]:
        if c in df.columns and "value" in df.columns:
            metric_col = c
            break
    if metric_col:
        long = df.rename(columns={metric_col: "indicator"}).copy()
        long["indicator"] = long["indicator"].map(lambda x: canonical_indicator(str(x)) or str(x))
    else:
        id_cols = set(["model", "step", "step_num", "checkpoint", "stage", "revision", "layer", "layer_idx", "layer_id", "module", "module_name", "matrix", "param", "parameter", "name"])
        value_cols = []
        for c in df.columns:
            if c in id_cols:
                continue
            ind = canonical_indicator(c)
            if ind and pd.api.types.is_numeric_dtype(df[c]):
                value_cols.append(c)
        if not value_cols:
            # Fallback: numeric columns except obvious ids.
            for c in df.select_dtypes(include=[np.number]).columns:
                if c not in ["step_num", "layer", "layer_idx", "layer_id"]:
                    value_cols.append(c)
        long = df.melt(id_vars=[c for c in df.columns if c not in value_cols], value_vars=value_cols, var_name="indicator", value_name="value")
        long["indicator"] = long["indicator"].map(lambda x: canonical_indicator(str(x)) or str(x))

    if "value" not in long.columns:
        raise ValueError("Could not find/melt a value column in E1 metrics CSV")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value", "step_num"]).copy()

    # Infer module/layer columns.
    module_source = None
    for c in ["module", "module_name", "matrix", "param", "parameter", "name"]:
        if c in long.columns:
            module_source = c
            break
    if module_source is None:
        long["module_text"] = "unknown"
    else:
        long["module_text"] = long[module_source].astype(str)
    if "layer" not in long.columns:
        long["layer"] = long.apply(infer_layer, axis=1)
    long["layer"] = pd.to_numeric(long["layer"], errors="coerce").fillna(-1).astype(int)
    long["module_role"] = long["module_text"].map(infer_module_role)
    long["module_family"] = long["module_role"].map(family_from_role)
    return long


def mean_sem(g):
    return pd.Series({"mean": g.mean(), "sem": g.sem() if len(g) > 1 else 0.0, "n": len(g)})


def plot_lines(df: pd.DataFrame, x="step_num", y="mean", group="indicator", out: Path = None, title="", ylabel="Normalized value", logx=True):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, sub in df.groupby(group):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", linewidth=1.6, markersize=3, label=str(name))
        if "sem" in sub.columns:
            ax.fill_between(sub[x], sub[y]-sub["sem"], sub[y]+sub["sem"], alpha=0.12)
    if logx:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200)
        plt.close(fig)
    else:
        return fig


def plot_heatmap(pivot: pd.DataFrame, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    # Use at most 12 x ticks
    n = len(pivot.columns)
    idx = np.linspace(0, n-1, min(n, 12), dtype=int)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(pivot.columns[i])) for i in idx], rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-metrics", required=True)
    ap.add_argument("--out", default=None, help="Output directory. Defaults to solid_results/e1_dense_indicator_panels_<model-tag>")
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--exclude-step0", action="store_true")
    ap.add_argument("--warmup-end", type=int, default=1400)
    args = ap.parse_args()

    metrics_path = Path(args.e1_metrics)
    tag = args.model_tag
    if tag is None:
        m = re.search(r"(\d+p?\d*b|\d+m|160m|410m|1p4b|1\.4b)", str(metrics_path).lower())
        tag = (m.group(1) if m else metrics_path.stem).replace(".", "p")
    out = Path(args.out or f"solid_results/e1_dense_indicator_panels_{tag}")
    figdir = out / "figures"
    tabdir = out / "tables"
    repdir = out / "reports"
    for d in [figdir, tabdir, repdir]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_metrics(metrics_path)
    if args.exclude_step0:
        df = df[df["step_num"] != 0].copy()

    df.to_csv(tabdir / "e1_metrics_long_normalized.csv", index=False)
    summary = df.groupby(["indicator", "step_num"])["value"].apply(mean_sem).reset_index()
    # apply(mean_sem) yields columns maybe with level; simpler recompute:
    summary = df.groupby(["indicator", "step_num"]).agg(mean=("value", "mean"), sem=("value", "sem"), n=("value", "size")).reset_index()
    summary.to_csv(tabdir / "e1_indicator_overall_summary.csv", index=False)

    fam = df.groupby(["indicator", "module_family", "step_num"]).agg(mean=("value", "mean"), sem=("value", "sem"), n=("value", "size")).reset_index()
    fam.to_csv(tabdir / "e1_indicator_by_module_family.csv", index=False)

    role = df.groupby(["indicator", "module_role", "step_num"]).agg(mean=("value", "mean"), sem=("value", "sem"), n=("value", "size")).reset_index()
    role.to_csv(tabdir / "e1_indicator_by_module_role.csv", index=False)

    # Overview: normalize each indicator to first observed mean for readability.
    overview = summary.copy()
    overview["mean_raw"] = overview["mean"]
    overview["sem_raw"] = overview["sem"]
    overview["base"] = overview.groupby("indicator")["mean"].transform(lambda s: s.iloc[0] if len(s) else np.nan)
    overview["mean"] = overview["mean"] / overview["base"].replace(0, np.nan)
    overview["sem"] = overview["sem"] / overview["base"].replace(0, np.nan)
    plot_lines(overview, out=figdir/"e1_dense_indicator_overview_logx.png", title=f"Dense E1 indicator overview ({tag})", logx=True)
    plot_lines(overview, out=figdir/"e1_dense_indicator_overview_linear.png", title=f"Dense E1 indicator overview ({tag})", logx=False)

    # Attention vs MLP rank curves.
    rank_inds = [x for x in ["stable_rank", "effective_rank"] if x in set(fam["indicator"])]
    if rank_inds:
        rank = fam[(fam["indicator"].isin(rank_inds)) & (fam["module_family"].isin(["attention", "mlp"]))].copy()
        rank["series"] = rank["indicator"] + " / " + rank["module_family"]
        plot_lines(rank, group="series", out=figdir/"e1_attention_vs_mlp_rank_logx.png", title=f"Attention vs MLP rank dynamics ({tag})", ylabel="Rank value", logx=True)
        plot_lines(rank, group="series", out=figdir/"e1_attention_vs_mlp_rank_linear.png", title=f"Attention vs MLP rank dynamics ({tag})", ylabel="Rank value", logx=False)

    # Module role plots for each indicator.
    for ind in sorted(role["indicator"].unique()):
        sub = role[(role["indicator"] == ind) & (role["module_role"] != "other")].copy()
        if sub.empty:
            continue
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", ind)
        plot_lines(sub, group="module_role", out=figdir/f"e1_module_role_{safe}_logx.png", title=f"{ind}: module-role dynamics ({tag})", ylabel=ind, logx=True)
        plot_lines(sub, group="module_role", out=figdir/f"e1_module_role_{safe}_linear.png", title=f"{ind}: module-role dynamics ({tag})", ylabel=ind, logx=False)

    # Layer heatmaps for selected indicators and families.
    for ind in ["stable_rank", "effective_rank", "sv_stability", "mp_outliers", "alpha"]:
        if ind not in set(df["indicator"]):
            continue
        for fam_name in ["attention", "mlp"]:
            sub = df[(df["indicator"] == ind) & (df["module_family"] == fam_name) & (df["layer"] >= 0)]
            if sub.empty:
                continue
            heat = sub.groupby(["layer", "step_num"])["value"].mean().reset_index()
            pivot = heat.pivot(index="layer", columns="step_num", values="value").sort_index(axis=0).sort_index(axis=1)
            safe = re.sub(r"[^a-zA-Z0-9_]+", "_", ind)
            plot_heatmap(pivot, figdir/f"e1_layer_heatmap_{safe}_{fam_name}.png", f"{ind}: layerwise {fam_name} dynamics ({tag})")

    report = []
    report.append(f"# E1 dense indicator panels ({tag})\n")
    report.append(f"- Input CSV: `{metrics_path}`")
    report.append(f"- Output root: `{out}`")
    report.append(f"- Rows after filtering: {len(df):,}")
    report.append(f"- Step range: {int(df['step_num'].min())} to {int(df['step_num'].max())}")
    report.append(f"- Indicators: {', '.join(sorted(map(str, df['indicator'].unique())))}")
    report.append(f"- Module families: {', '.join(sorted(map(str, df['module_family'].unique())))}")
    report.append("\n## Main files\n")
    for f in sorted(figdir.glob("*.png")):
        report.append(f"- `{f.relative_to(out)}`")
    (repdir/"e1_dense_indicator_panel_report.md").write_text("\n".join(report)+"\n")
    print("Wrote", out)

if __name__ == "__main__":
    main()
