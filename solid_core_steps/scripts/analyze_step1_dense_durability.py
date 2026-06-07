#!/usr/bin/env python3
"""Analyze Step-1 dense factual durability sweep.

Produces: stage summary, window-vs-late bootstrap CIs, segmented-vs-monotone
AIC comparison, and figures for the thesis main body.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_step(x) -> int:
    if pd.isna(x):
        return -1
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x)
    if s.startswith("step"):
        s = s[4:]
    try:
        return int(float(s))
    except Exception:
        return -1


def infer_step_col(df):
    for c in ["step_num", "stage_step", "inject_step_num", "step", "checkpoint", "stage"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in ["step", "stage", "checkpoint"]):
            return c
    raise ValueError(f"No step column found: {list(df.columns)}")


def choose_col(df, candidates, contains_all=()):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in df.columns:
        cl = c.lower()
        if all(x in cl for x in contains_all):
            return c
    return None


def load_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            csvs = list(path.glob("**/*cell_summary*.csv")) + list(path.glob("**/*summary*.csv"))
            if not csvs:
                raise FileNotFoundError(f"No summary CSV found under {path}")
            for c in csvs:
                frames.append(pd.read_csv(c).assign(_source=str(c)))
        else:
            frames.append(pd.read_csv(path).assign(_source=str(path)))
    return pd.concat(frames, ignore_index=True, sort=False)


def fit_aic(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[mask], X[mask]
    if len(y) <= X.shape[1] + 1:
        return np.nan, np.full(X.shape[1], np.nan), np.nan
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    rss = max(rss, 1e-12)
    n, k = len(y), X.shape[1]
    aic = n * math.log(rss / n) + 2 * k
    return aic, beta, rss


def bootstrap_diff(df, metric, window_lo, window_hi, late_lo, n_boot, seed):
    rng = np.random.default_rng(seed)
    w = df[(df.step >= window_lo) & (df.step <= window_hi)][metric].dropna().values
    l = df[df.step >= late_lo][metric].dropna().values
    if len(w) == 0 or len(l) == 0:
        return dict(window_mean=np.nan, late_mean=np.nan, diff=np.nan, ci_low=np.nan, ci_high=np.nan, p_le_0=np.nan)
    boots = []
    for _ in range(n_boot):
        wb = rng.choice(w, size=len(w), replace=True)
        lb = rng.choice(l, size=len(l), replace=True)
        boots.append(float(np.mean(wb) - np.mean(lb)))
    boots = np.asarray(boots)
    return dict(
        window_mean=float(np.mean(w)), late_mean=float(np.mean(l)), diff=float(np.mean(w)-np.mean(l)),
        ci_low=float(np.quantile(boots, 0.025)), ci_high=float(np.quantile(boots, 0.975)),
        p_le_0=float(np.mean(boots <= 0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV(s) or result roots containing Step-1/E3 summaries")
    ap.add_argument("--out", default="results/step1_dense_durability_analysis")
    ap.add_argument("--window", default="512,3000")
    ap.add_argument("--late-min", type=int, default=8000)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    for d in ["tables", "figures", "reports"]:
        (out/d).mkdir(parents=True, exist_ok=True)

    df = load_inputs(args.inputs)
    step_col = infer_step_col(df)
    df["step"] = df[step_col].map(parse_step)
    df = df[df.step >= 0].copy()

    uptake_col = choose_col(df, ["uptake_margin", "margin_uptake_delta", "uptake_delta_margin"], ("uptake", "margin"))
    retention_col = choose_col(df, ["normalized_retention_margin", "norm_retention_margin", "retention_margin_norm", "clean_retention_norm_margin"], ("retention", "margin"))
    degrade_col = choose_col(df, ["normalized_degradation_auc_margin", "degradation_auc_margin", "poison_auc_margin"], ("auc", "margin"))

    if retention_col is None:
        raise ValueError(f"Could not infer retention-margin column. Columns: {list(df.columns)}")
    metrics = {"retention_margin": retention_col}
    if uptake_col: metrics["uptake_margin"] = uptake_col
    if degrade_col: metrics["degradation_auc_margin"] = degrade_col

    for new, old in metrics.items():
        df[new] = pd.to_numeric(df[old], errors="coerce")

    stage = df.groupby("step", as_index=False).agg({m: ["mean", "sem", "count"] for m in metrics})
    stage.columns = ["_".join([str(x) for x in col if x]).strip("_") for col in stage.columns.values]
    stage.to_csv(out/"tables"/"step1_stage_summary.csv", index=False)

    win_lo, win_hi = [int(x) for x in args.window.split(",")]
    boot_rows = []
    for metric in metrics:
        r = bootstrap_diff(df, metric, win_lo, win_hi, args.late_min, args.n_bootstrap, args.seed)
        r["metric"] = metric
        boot_rows.append(r)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out/"tables"/"window_vs_late_bootstrap.csv", index=False)

    # Model comparison on cell-level retention.
    comp_rows = []
    work = df[["step", "retention_margin"]].dropna().copy()
    work = work[work.step > 0]
    x = np.log10(work.step.values.astype(float))
    y = work.retention_margin.values.astype(float)
    X0 = np.ones((len(work), 1))
    aic, beta, rss = fit_aic(y, X0)
    comp_rows.append(dict(model="constant", break_step="", aic=aic, rss=rss, params=list(beta)))
    Xmono = np.column_stack([np.ones(len(x)), x])
    aic, beta, rss = fit_aic(y, Xmono)
    comp_rows.append(dict(model="monotone_logstep", break_step="", aic=aic, rss=rss, params=list(beta)))
    for br in [512, 1000, 1400, 2000, 3000, 4000, 8000]:
        xb = np.maximum(0, np.log10(work.step.values.astype(float)) - math.log10(br))
        X = np.column_stack([np.ones(len(x)), x, xb])
        aic, beta, rss = fit_aic(y, X)
        comp_rows.append(dict(model="segmented_logstep", break_step=br, aic=aic, rss=rss, params=list(beta)))
    comp = pd.DataFrame(comp_rows).sort_values("aic")
    comp.to_csv(out/"tables"/"segmented_vs_monotone_aic.csv", index=False)

    # Figures.
    fig, ax = plt.subplots(figsize=(9, 5))
    ss = stage.sort_values("step")
    for metric, label in [("uptake_margin", "uptake"), ("retention_margin", "retention"), ("degradation_auc_margin", "degradation AUC")]:
        mean_col, sem_col = f"{metric}_mean", f"{metric}_sem"
        if mean_col in ss.columns:
            ax.plot(ss.step, ss[mean_col], marker="o", markersize=3, label=label)
            if sem_col in ss.columns:
                ax.fill_between(ss.step, ss[mean_col]-ss[sem_col].fillna(0), ss[mean_col]+ss[sem_col].fillna(0), alpha=0.15)
    ax.axvspan(win_lo, win_hi, alpha=0.10, label="reorganisation window")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("injection checkpoint step")
    ax.set_ylabel("metric value")
    ax.set_title("Step 1 dense factual durability sweep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"figures"/"durability_sweep.png", dpi=200)
    fig.savefig(out/"figures"/"durability_sweep.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(work.step, work.retention_margin, s=14, alpha=0.55, label="cells")
    grid = np.unique(np.sort(work.step.values.astype(float)))
    # Best segmented/monotone overlay.
    mono = comp[comp.model == "monotone_logstep"].iloc[0]
    b = np.array(mono.params, dtype=float)
    ax.plot(grid, b[0] + b[1]*np.log10(grid), label=f"monotone AIC={mono.aic:.1f}")
    best_seg = comp[comp.model == "segmented_logstep"].iloc[0]
    br = float(best_seg.break_step)
    b = np.array(best_seg.params, dtype=float)
    ax.plot(grid, b[0] + b[1]*np.log10(grid) + b[2]*np.maximum(0, np.log10(grid)-math.log10(br)), label=f"segmented br={int(br)} AIC={best_seg.aic:.1f}")
    ax.axvline(br, linestyle="--", linewidth=1)
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("injection checkpoint step")
    ax.set_ylabel("retention margin")
    ax.set_title("Break-vs-monotone comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"figures"/"break_test.png", dpi=200)
    fig.savefig(out/"figures"/"break_test.pdf")
    plt.close(fig)

    report = [
        "# Step 1 dense factual durability analysis", "",
        f"Inputs: `{args.inputs}`", "",
        f"Retention column: `{retention_col}`", "",
        f"Window: step{win_lo}–step{win_hi}; late >= step{args.late_min}", "",
        "## Window-vs-late bootstrap", "", boot.to_markdown(index=False), "",
        "## Segmented-vs-monotone AIC", "", comp.head(10).to_markdown(index=False), "",
        "Outputs:", "", "- `figures/durability_sweep.png`", "- `figures/break_test.png`", "- `tables/step1_stage_summary.csv`", "- `tables/window_vs_late_bootstrap.csv`", "- `tables/segmented_vs_monotone_aic.csv`", "",
    ]
    (out/"reports"/"step1_dense_durability_report.md").write_text("\n".join(report))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
