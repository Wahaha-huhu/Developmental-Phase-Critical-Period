#!/usr/bin/env python3
"""Analyze Step 1 dense factual durability from clean cell-summary records.

This script is intentionally strict about the unit of analysis: one row per
(stage, seed) cell. It reads e3_factual_cell_summary.csv and maps the actual
runner columns:
  uptake_margin_delta          -> uptake
  normalized_retention_margin  -> retention
  degradation_auc_margin       -> degradation AUC

Outputs are written under --out with figures/tables/reports.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_step_value(x) -> int:
    if pd.isna(x):
        return -1
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    s = str(x).strip()
    if s.startswith("step"):
        s = s[4:]
    try:
        return int(float(s))
    except Exception:
        return -1


def find_cell_summary(inputs: Path) -> Path:
    if inputs.is_file():
        return inputs
    candidates = [
        inputs / "raw" / "e3_factual_cell_summary.csv",
        inputs / "e3_factual_cell_summary.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(inputs.rglob("e3_factual_cell_summary.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find e3_factual_cell_summary.csv under {inputs}. "
        "Pass either the CSV directly or a folder containing raw/e3_factual_cell_summary.csv."
    )


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def choose_column(df: pd.DataFrame, candidates: Iterable[str], required_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find column for {required_name}. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def load_clean_cells(inputs: Path, exclude_step0: bool = True) -> tuple[pd.DataFrame, dict]:
    csv_path = find_cell_summary(inputs)
    df = pd.read_csv(csv_path)

    step_col = choose_column(df, ["step", "step_num", "stage"], "step")
    stage_col = "stage" if "stage" in df.columns else step_col
    seed_col = choose_column(df, ["seed"], "seed")
    uptake_col = choose_column(
        df,
        ["uptake_margin_delta", "uptake_margin", "post_injection_mean_margin", "uptake_mean"],
        "uptake",
    )
    retention_col = choose_column(
        df,
        ["normalized_retention_margin", "retention_margin", "post_continuation_mean_margin", "retention_mean"],
        "retention",
    )
    auc_col = choose_column(
        df,
        ["degradation_auc_margin", "degradation_auc_margin_mean", "degradation_auc", "auc_margin"],
        "degradation AUC",
    )

    out = pd.DataFrame(
        {
            "stage": df[stage_col].astype(str),
            "step_num": df[step_col].map(parse_step_value),
            "seed": df[seed_col].astype(int),
            "uptake": pd.to_numeric(df[uptake_col], errors="coerce"),
            "retention": pd.to_numeric(df[retention_col], errors="coerce"),
            "degradation_auc": pd.to_numeric(df[auc_col], errors="coerce"),
        }
    )

    before = len(out)
    out = out.dropna(subset=["step_num", "seed", "uptake", "retention", "degradation_auc"]).copy()
    if exclude_step0:
        out = out[out["step_num"] > 0].copy()

    # Deduplicate only at the correct cell level. If duplicates exist, keep the latest row.
    dup_count = int(out.duplicated(["stage", "seed"]).sum())
    out = out.drop_duplicates(["stage", "seed"], keep="last").copy()
    out = out.sort_values(["step_num", "seed"]).reset_index(drop=True)

    meta = {
        "source_csv": str(csv_path),
        "raw_rows": int(before),
        "clean_rows": int(len(out)),
        "duplicates_dropped_stage_seed": dup_count,
        "step_col": step_col,
        "stage_col": stage_col,
        "seed_col": seed_col,
        "uptake_col": uptake_col,
        "retention_col": retention_col,
        "degradation_auc_col": auc_col,
        "exclude_step0": exclude_step0,
    }
    return out, meta


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step_num, g in cells.groupby("step_num", sort=True):
        stage = g["stage"].iloc[0]
        row = {"step_num": int(step_num), "stage": stage, "n": int(len(g))}
        for metric in ["uptake", "retention", "degradation_auc"]:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_sem"] = sem(g[metric])
            row[f"{metric}_median"] = float(g[metric].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step_num").reset_index(drop=True)


def plot_with_sem(summary: pd.DataFrame, metrics: list[str], out_png: Path, title: str, logx: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = summary["step_num"].to_numpy()
    for metric in metrics:
        y = summary[f"{metric}_mean"].to_numpy()
        e = summary[f"{metric}_sem"].to_numpy()
        label = metric.replace("_", " ").title()
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=label)
        if np.isfinite(e).any():
            ax.fill_between(x, y - e, y + e, alpha=0.18)
    ax.axvspan(512, 3000, alpha=0.12, label="Working window 512–3000")
    ax.axvline(1400, linestyle="--", linewidth=1.1, alpha=0.7, label="Warmup end ≈1400")
    ax.set_title(title)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Margin / normalized score")
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    n = len(y)
    k = X.shape[1]
    aic = n * math.log(max(rss / n, 1e-12)) + 2 * k
    return pred, rss, aic


def fit_segmented(x: np.ndarray, y: np.ndarray, break_step: int) -> tuple[np.ndarray, float, float]:
    # Continuous piecewise-linear hinge model: y = b0 + b1*x + b2*max(0, x-break)
    hinge = np.maximum(0, x - break_step)
    X = np.column_stack([np.ones_like(x), x, hinge])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    n = len(y)
    k = X.shape[1]
    aic = n * math.log(max(rss / n, 1e-12)) + 2 * k
    return pred, rss, aic


def break_vs_monotone(summary: pd.DataFrame, out_png: Path, out_csv: Path) -> None:
    # Positive x only; step0 is already excluded, but keep this guard.
    s = summary[summary["step_num"] > 0].copy()
    x_raw = s["step_num"].to_numpy(dtype=float)
    x = np.log10(x_raw)
    y = s["retention_mean"].to_numpy(dtype=float)

    mono_pred, mono_rss, mono_aic = fit_linear(x, y)

    candidate_steps = [st for st in s["step_num"].tolist() if 128 <= st <= 9000]
    records = []
    best = None
    for b in candidate_steps:
        b_log = math.log10(b)
        pred, rss, aic = fit_segmented(x, y, b_log)
        rec = {"break_step": int(b), "rss": rss, "aic": aic}
        records.append(rec)
        if best is None or aic < best["aic"]:
            best = {"break_step": int(b), "pred": pred, "rss": rss, "aic": aic}

    fit_df = pd.DataFrame(records).sort_values("aic")
    fit_df.insert(0, "monotone_aic", mono_aic)
    fit_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_raw, y, marker="o", linestyle="", label="Retention mean")
    order = np.argsort(x_raw)
    ax.plot(x_raw[order], mono_pred[order], linewidth=1.8, label=f"Monotone linear AIC={mono_aic:.1f}")
    if best is not None:
        ax.plot(x_raw[order], best["pred"][order], linewidth=1.8, label=f"Segmented break={best['break_step']} AIC={best['aic']:.1f}")
        ax.axvline(best["break_step"], linestyle="--", linewidth=1.1, alpha=0.7)
    ax.axvline(1400, linestyle=":", linewidth=1.2, alpha=0.8, label="Warmup end ≈1400")
    ax.set_xscale("log")
    ax.set_xlabel("Checkpoint step (positive only)")
    ax.set_ylabel("Retention mean")
    ax.set_title("Step 1 retention: segmented vs monotone fit")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_report(out_dir: Path, meta: dict, summary: pd.DataFrame) -> None:
    report = out_dir / "reports" / "step1_dense_durability_report.md"
    peak = summary.loc[summary["retention_mean"].idxmax()].to_dict() if len(summary) else {}
    lines = [
        "# Step 1 dense factual durability analysis",
        "",
        "## Input mapping",
        f"- Source CSV: `{meta['source_csv']}`",
        f"- Uptake column: `{meta['uptake_col']}`",
        f"- Retention column: `{meta['retention_col']}`",
        f"- Degradation-AUC column: `{meta['degradation_auc_col']}`",
        f"- Exclude step0: `{meta['exclude_step0']}`",
        f"- Clean cell rows: {meta['clean_rows']}",
        f"- Duplicate stage/seed rows dropped: {meta['duplicates_dropped_stage_seed']}",
        "",
        "## Summary",
        f"- Checkpoints in main analysis: {len(summary)}",
    ]
    if peak:
        lines += [
            f"- Peak retention checkpoint: step {int(peak['step_num'])}",
            f"- Peak retention mean: {peak['retention_mean']:.4f}",
            f"- Uptake at peak: {peak['uptake_mean']:.4f}",
        ]
    lines += [
        "",
        "## Generated figures",
        "- `figures/step1_uptake_retention_curve_logx.png`",
        "- `figures/step1_uptake_retention_curve_linear.png`",
        "- `figures/step1_degradation_auc_curve_logx.png`",
        "- `figures/step1_degradation_auc_curve_linear.png`",
        "- `figures/step1_break_vs_monotone_positive_x.png`",
        "",
        "AUC is plotted separately because its magnitude and interpretation differ from uptake/retention.",
    ]
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Path to e3_factual_cell_summary.csv or a result folder containing raw/e3_factual_cell_summary.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-step0", action="store_true", help="Include step0 in main analysis; default excludes it")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    cells, meta = load_clean_cells(inputs, exclude_step0=not args.include_step0)
    summary = summarize(cells)

    cells.to_csv(out_dir / "tables" / "step1_cell_records_standardized.csv", index=False)
    summary.to_csv(out_dir / "tables" / "step1_stage_summary.csv", index=False)
    (out_dir / "tables" / "input_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    plot_with_sem(summary, ["uptake", "retention"], out_dir / "figures" / "step1_uptake_retention_curve_logx.png", "Step 1 uptake and retention by checkpoint", logx=True)
    plot_with_sem(summary, ["uptake", "retention"], out_dir / "figures" / "step1_uptake_retention_curve_linear.png", "Step 1 uptake and retention by checkpoint", logx=False)
    plot_with_sem(summary, ["degradation_auc"], out_dir / "figures" / "step1_degradation_auc_curve_logx.png", "Step 1 degradation AUC by checkpoint", logx=True)
    plot_with_sem(summary, ["degradation_auc"], out_dir / "figures" / "step1_degradation_auc_curve_linear.png", "Step 1 degradation AUC by checkpoint", logx=False)
    break_vs_monotone(summary, out_dir / "figures" / "step1_break_vs_monotone_positive_x.png", out_dir / "tables" / "segmented_vs_monotone_aic.csv")
    write_report(out_dir, meta, summary)

    print(f"Wrote Step 1 analysis to {out_dir}")
    print("Columns:", list(summary.columns))
    print("n unique:", sorted(summary["n"].unique().tolist()))


if __name__ == "__main__":
    main()
