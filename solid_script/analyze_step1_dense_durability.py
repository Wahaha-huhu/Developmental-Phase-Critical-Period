#!/usr/bin/env python3
"""
Step 1 dense factual durability analysis for the solid thesis path.

Key plotting choices:
  - Exclude step0 from main analysis by default, because uptake-normalized quantities
    are unstable when uptake is near zero.
  - Plot uptake and retention together, but plot degradation/AUC separately because
    their scales are usually different.
  - Fit/plot break-vs-monotone only on positive step numbers.

The script is intentionally schema-tolerant: it accepts common E3/Step1 CSV names and
tries several likely column aliases for uptake, retention, and degradation AUC.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"(\d+)")


UPTAKE_ALIASES = [
    "uptake_margin_delta",
    "uptake_delta",
    "uptake_margin",
    "uptake",
    "U",
    "mean_uptake_margin_delta",
]
RETENTION_ALIASES = [
    "retention_margin_delta",
    "clean_retention_margin_delta",
    "retention_delta",
    "retention",
    "clean_retention",
    "retention_ratio",
    "normalized_retention",
    "R",
    "mean_retention_margin_delta",
]
AUC_ALIASES = [
    "degradation_auc",
    "degradation_resistance_auc",
    "poison_auc",
    "attack_auc",
    "auc",
    "normalized_auc",
    "degradation_auc_norm",
    "mean_degradation_auc",
]
MARGIN_AUC_ALIASES = [
    "degradation_margin_auc",
    "poison_margin_auc",
    "attack_margin_auc",
]

STAGE_ALIASES = ["stage", "checkpoint", "inject_checkpoint", "revision", "step", "stage_name"]
SEED_ALIASES = ["seed", "random_seed"]


CANDIDATE_FILES = [
    "tables/e3_v3_cell_summary.csv",
    "tables/e3_factual_cell_summary.csv",
    "raw/e3_factual_cell_summary.csv",
    "raw/e3_v3_cell_summary.csv",
    "raw/e3_factual_cell_records.csv",
    "e3_factual_cell_summary.csv",
    "e3_v3_cell_summary.csv",
    "tables/step1_cell_summary.csv",
    "step1_cell_summary.csv",
    "tables/step1_stage_summary.csv",
    "step1_stage_summary.csv",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_step(value) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    s = str(value)
    if s.startswith("step"):
        s = s[4:]
    m = STEP_RE.search(s)
    return int(m.group(1)) if m else None


def first_existing(columns: Iterable[str], aliases: list[str]) -> Optional[str]:
    cols = set(columns)
    for a in aliases:
        if a in cols:
            return a
    # Case-insensitive fallback.
    lower = {c.lower(): c for c in columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def find_input_csv(path: Path) -> Path:
    if path.is_file():
        return path
    for rel in CANDIDATE_FILES:
        p = path / rel
        if p.exists():
            return p
    csvs = sorted(path.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {path}")
    # Prefer files that look like cell/stage summaries.
    preferred = [p for p in csvs if any(k in p.name.lower() for k in ["summary", "cell", "stage"])]
    return preferred[0] if preferred else csvs[0]


def load_and_standardize(input_path: Path, exclude_step0: bool) -> tuple[pd.DataFrame, dict]:
    csv_path = find_input_csv(input_path)
    df = pd.read_csv(csv_path)

    stage_col = first_existing(df.columns, STAGE_ALIASES)
    if stage_col is None and "step_num" not in df.columns:
        raise KeyError(
            f"Could not infer stage column. Columns available: {list(df.columns)}"
        )

    if "stage" not in df.columns:
        if stage_col is not None:
            df["stage"] = df[stage_col].astype(str)
        else:
            df["stage"] = "step" + df["step_num"].astype(str)

    if "step_num" not in df.columns:
        df["step_num"] = df["stage"].map(parse_step)
    df["step_num"] = pd.to_numeric(df["step_num"], errors="coerce")

    seed_col = first_existing(df.columns, SEED_ALIASES)
    if seed_col is None:
        df["seed"] = 0
    elif seed_col != "seed":
        df["seed"] = df[seed_col]

    uptake_col = first_existing(df.columns, UPTAKE_ALIASES)
    retention_col = first_existing(df.columns, RETENTION_ALIASES)
    auc_col = first_existing(df.columns, AUC_ALIASES)
    margin_auc_col = first_existing(df.columns, MARGIN_AUC_ALIASES)

    rename = {}
    if uptake_col:
        rename[uptake_col] = "uptake"
    if retention_col:
        rename[retention_col] = "retention"
    if auc_col:
        rename[auc_col] = "degradation_auc"
    if margin_auc_col and margin_auc_col not in rename:
        rename[margin_auc_col] = "degradation_margin_auc"
    df = df.rename(columns=rename)

    if "uptake" not in df.columns:
        log("[WARN] No uptake column detected; uptake plots will be skipped.")
    if "retention" not in df.columns:
        log("[WARN] No retention column detected; retention and break-test will be skipped.")
    if "degradation_auc" not in df.columns:
        log("[WARN] No degradation/AUC column detected; AUC plots will be skipped.")

    # Force numeric metric columns.
    for col in ["uptake", "retention", "degradation_auc", "degradation_margin_auc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df[df["step_num"].notna()].copy()
    df["step_num"] = df["step_num"].astype(int)

    if exclude_step0:
        df = df[df["step_num"] > 0].copy()
    dropped = before - len(df)

    meta = {
        "input_csv": str(csv_path),
        "n_rows_loaded": int(before),
        "n_rows_after_filter": int(len(df)),
        "n_rows_dropped_or_excluded": int(dropped),
        "stage_col": stage_col,
        "uptake_col": uptake_col,
        "retention_col": retention_col,
        "degradation_auc_col": auc_col,
        "degradation_margin_auc_col": margin_auc_col,
        "exclude_step0": bool(exclude_step0),
    }
    return df, meta


def sem(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) <= 1:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def summarize_by_stage(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in ["uptake", "retention", "degradation_auc", "degradation_margin_auc"] if c in df.columns]
    rows = []
    for step, g in df.groupby("step_num", sort=True):
        row = {"step_num": int(step), "stage": f"step{int(step)}", "n": int(len(g))}
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean(skipna=True))
            row[f"{m}_sem"] = sem(g[m])
            row[f"{m}_median"] = float(g[m].median(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step_num")


def _plot_metric_pair(summary: pd.DataFrame, out: Path, xscale: str) -> None:
    cols = []
    for m in ["uptake", "retention"]:
        if f"{m}_mean" in summary.columns:
            cols.append(m)
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = summary["step_num"].astype(float).to_numpy()
    for m in cols:
        y = summary[f"{m}_mean"].astype(float).to_numpy()
        ax.plot(x, y, marker="o", label=m.replace("_", " ").title())
        sem_col = f"{m}_sem"
        if sem_col in summary.columns:
            e = summary[sem_col].astype(float).to_numpy()
            if np.isfinite(e).any():
                ax.fill_between(x, y - e, y + e, alpha=0.18)
    ax.set_xlabel("Injection step")
    ax.set_ylabel("Margin / normalized score")
    ax.set_title("Step 1 factual uptake and clean retention")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    if xscale == "log":
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_auc(summary: pd.DataFrame, out: Path, xscale: str) -> None:
    auc_cols = []
    for m in ["degradation_auc", "degradation_margin_auc"]:
        if f"{m}_mean" in summary.columns:
            auc_cols.append(m)
    if not auc_cols:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = summary["step_num"].astype(float).to_numpy()
    for m in auc_cols:
        label = "Degradation AUC" if m == "degradation_auc" else "Degradation margin AUC"
        y = summary[f"{m}_mean"].astype(float).to_numpy()
        ax.plot(x, y, marker="o", label=label)
        sem_col = f"{m}_sem"
        if sem_col in summary.columns:
            e = summary[sem_col].astype(float).to_numpy()
            if np.isfinite(e).any():
                ax.fill_between(x, y - e, y + e, alpha=0.18)
    ax.set_xlabel("Injection step")
    ax.set_ylabel("AUC")
    ax.set_title("Step 1 factual degradation-resistance AUC")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    if xscale == "log":
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    k = X.shape[1]
    n = len(y)
    aic = n * math.log(max(rss / max(n, 1), 1e-12)) + 2 * k
    return pred, rss, aic


def fit_segmented(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    # Piecewise linear hinge model: y = a + b*x + c*max(0, x-break).
    candidates = np.unique(x)[1:-1]
    best = None
    for bp in candidates:
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - bp)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        rss = float(np.sum((y - pred) ** 2))
        k = X.shape[1] + 1  # params + breakpoint selection
        n = len(y)
        aic = n * math.log(max(rss / max(n, 1), 1e-12)) + 2 * k
        if best is None or aic < best[2]:
            best = (pred, rss, aic, float(bp))
    if best is None:
        pred, rss, aic = fit_linear(x, y)
        return pred, rss, aic, float("nan")
    return best


def plot_break_vs_monotone(summary: pd.DataFrame, out: Path) -> pd.DataFrame:
    if "retention_mean" not in summary.columns:
        return pd.DataFrame()
    d = summary[(summary["step_num"] > 0) & summary["retention_mean"].notna()].copy()
    if len(d) < 5:
        return pd.DataFrame()
    x_step = d["step_num"].astype(float).to_numpy()
    x = np.log10(x_step)
    y = d["retention_mean"].astype(float).to_numpy()

    lin_pred, lin_rss, lin_aic = fit_linear(x, y)
    seg_pred, seg_rss, seg_aic, bp_log = fit_segmented(x, y)
    bp_step = 10 ** bp_log if np.isfinite(bp_log) else float("nan")

    order = np.argsort(x_step)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.scatter(x_step, y, label="Stage mean")
    ax.plot(x_step[order], lin_pred[order], label="Monotone linear fit")
    ax.plot(x_step[order], seg_pred[order], label="Segmented fit")
    if np.isfinite(bp_step):
        ax.axvline(bp_step, linestyle="--", alpha=0.7, label=f"Break ≈ step{bp_step:.0f}")
    ax.set_xscale("log")
    ax.set_xlim(left=max(float(np.min(x_step)), 1.0), right=float(np.max(x_step)))
    ax.set_xlabel("Injection step (positive only)")
    ax.set_ylabel("Retention")
    ax.set_title("Break vs monotone fit for Step 1 retention")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

    return pd.DataFrame([
        {"model": "monotone_linear", "rss": lin_rss, "aic": lin_aic, "break_step": np.nan},
        {"model": "segmented_hinge", "rss": seg_rss, "aic": seg_aic, "break_step": bp_step},
    ])


def write_report(out_dir: Path, meta: dict, summary: pd.DataFrame, fit_table: pd.DataFrame) -> None:
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Step 1 Dense Factual Durability Analysis\n")
    lines.append("\n## Input\n")
    lines.append(f"- CSV: `{meta['input_csv']}`\n")
    lines.append(f"- Rows loaded: {meta['n_rows_loaded']}\n")
    lines.append(f"- Rows after filtering: {meta['n_rows_after_filter']}\n")
    lines.append(f"- Exclude step0: {meta['exclude_step0']}\n")
    lines.append("\n## Plotting choices\n")
    lines.append("- Uptake and retention are plotted together.\n")
    lines.append("- Degradation/AUC is plotted separately because its magnitude is not comparable to uptake/retention.\n")
    lines.append("- Break-vs-monotone uses positive injection steps only.\n")
    lines.append("\n## Metric columns detected\n")
    for k in ["uptake_col", "retention_col", "degradation_auc_col", "degradation_margin_auc_col"]:
        lines.append(f"- {k}: `{meta.get(k)}`\n")
    if not fit_table.empty:
        lines.append("\n## Break-vs-monotone\n")
        best = fit_table.sort_values("aic").iloc[0]
        lines.append(f"- Best AIC model: `{best['model']}`\n")
        if np.isfinite(best.get("break_step", np.nan)):
            lines.append(f"- Estimated break step: {best['break_step']:.1f}\n")
    lines.append("\n## Output figures\n")
    for f in sorted((out_dir / "figures").glob("*.png")):
        lines.append(f"- `{f}`\n")
    (report_dir / "step1_dense_durability_report.md").write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Step1 results directory or CSV file")
    ap.add_argument("--out", required=True, help="Output directory under solid_results")
    ap.add_argument("--include-step0", action="store_true", help="Include step0 in main analysis (not recommended)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    report_dir = out_dir / "reports"
    for d in [fig_dir, table_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df, meta = load_and_standardize(Path(args.inputs), exclude_step0=not args.include_step0)
    df.to_csv(table_dir / "step1_cell_records_standardized.csv", index=False)
    summary = summarize_by_stage(df)
    summary.to_csv(table_dir / "step1_stage_summary.csv", index=False)
    (table_dir / "input_metadata.json").write_text(json.dumps(meta, indent=2))

    _plot_metric_pair(summary, fig_dir / "step1_uptake_retention_curve_logx.png", xscale="log")
    _plot_metric_pair(summary, fig_dir / "step1_uptake_retention_curve_linear.png", xscale="linear")
    _plot_auc(summary, fig_dir / "step1_degradation_auc_curve_logx.png", xscale="log")
    _plot_auc(summary, fig_dir / "step1_degradation_auc_curve_linear.png", xscale="linear")
    fit_table = plot_break_vs_monotone(summary, fig_dir / "step1_break_vs_monotone_positive_x.png")
    if not fit_table.empty:
        fit_table.to_csv(table_dir / "segmented_vs_monotone_aic.csv", index=False)

    write_report(out_dir, meta, summary, fit_table)
    log(f"Wrote Step 1 analysis to {out_dir}")


if __name__ == "__main__":
    main()
