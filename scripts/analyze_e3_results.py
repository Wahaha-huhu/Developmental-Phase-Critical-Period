#!/usr/bin/env python3
"""Analyze E3 factual-intervention outputs.

Produces summary tables, retention/degradation curves, and a short markdown report.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dirs(root: Path) -> dict[str, Path]:
    out = {k: root / k for k in ["raw", "processed", "figures", "tables", "reports", "manifests"]}
    out["root"] = root
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _read_e3_metrics_flexible(path: Path) -> pd.DataFrame:
    """Read E3 metrics even if an older run wrote degradation rows with extra columns.

    Early versions of the E3 writer appended base rows with a 10-column header and
    degradation rows with two extra values (poison_budget, degradation_steps).
    pandas.read_csv then fails with: expected 10 fields, saw 12. This reader maps
    those two trailing values back to their intended columns.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            cols = list(header)
            if len(row) > len(cols):
                extras_needed = len(row) - len(cols)
                candidate_extras = ["poison_budget", "degradation_steps"]
                for name in candidate_extras:
                    if extras_needed <= 0:
                        break
                    if name not in cols:
                        cols.append(name)
                        extras_needed -= 1
                for i in range(extras_needed):
                    cols.append(f"extra_{i}")
            if len(row) < len(cols):
                row = row + [""] * (len(cols) - len(row))
            rows.append(dict(zip(cols, row)))
    return pd.DataFrame(rows)


def load_metrics(root: Path, metrics_name: str) -> pd.DataFrame:
    path = root / metrics_name
    if not path.exists():
        raise FileNotFoundError(f"Missing E3 metrics CSV: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = _read_e3_metrics_flexible(path)
    if df.empty:
        return df
    # Remove marker rows.
    df = df[df["event"] != "cell_complete"].copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(-1).astype(int)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(-1).astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "poison_budget" in df.columns:
        df["poison_budget"] = pd.to_numeric(df["poison_budget"], errors="coerce")
    if "degradation_steps" in df.columns:
        df["degradation_steps"] = pd.to_numeric(df["degradation_steps"], errors="coerce")
    return df


def pivot_cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    base_events = ["base", "post_injection", "post_continuation"]
    rows: list[dict[str, Any]] = []
    keys = ["experiment_id", "model", "stage", "step", "seed", "signal_type"]
    for key_vals, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key_vals))
        for event in base_events:
            eg = g[g["event"] == event]
            for _, r in eg.iterrows():
                row[f"{event}_{r['metric']}"] = r["value"]
        # Degradation events by budget.
        deg = g[g["event"].str.startswith("post_degradation_k", na=False)]
        for _, r in deg.iterrows():
            budget = int(r.get("poison_budget", np.nan)) if not pd.isna(r.get("poison_budget", np.nan)) else str(r["event"]).replace("post_degradation_k", "")
            row[f"degradation_k{budget}_{r['metric']}"] = r["value"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["model", "step", "seed"])

    # Derived metrics.
    denom = out.get("post_injection_mean_margin", np.nan) - out.get("base_mean_margin", np.nan)
    out["uptake_margin_delta"] = out.get("post_injection_mean_margin", np.nan) - out.get("base_mean_margin", np.nan)
    out["uptake_accuracy_delta"] = out.get("post_injection_accuracy", np.nan) - out.get("base_accuracy", np.nan)
    out["clean_retention_margin_delta"] = out.get("post_continuation_mean_margin", np.nan) - out.get("base_mean_margin", np.nan)
    out["normalized_clean_retention_margin"] = out["clean_retention_margin_delta"] / denom.replace(0, np.nan)
    out["clean_retention_accuracy_delta"] = out.get("post_continuation_accuracy", np.nan) - out.get("base_accuracy", np.nan)

    # Degradation AUC over normalized margin/accuracy when available.
    budgets = sorted({int(c.split("_", 2)[1][1:]) for c in out.columns if c.startswith("degradation_k") and c.endswith("_mean_margin")})
    for budget in budgets:
        out[f"normalized_degradation_k{budget}_margin"] = (
            out.get(f"degradation_k{budget}_mean_margin", np.nan) - out.get("base_mean_margin", np.nan)
        ) / denom.replace(0, np.nan)
    if budgets:
        norm_cols = [f"normalized_degradation_k{b}_margin" for b in budgets]
        acc_cols = [f"degradation_k{b}_accuracy" for b in budgets]
        x = np.log10(np.asarray(budgets, dtype=float))
        auc_norm = []
        auc_acc = []
        k_star_acc = []
        for _, r in out.iterrows():
            y = r[norm_cols].astype(float).to_numpy()
            ya = r[acc_cols].astype(float).to_numpy() if all(c in out.columns for c in acc_cols) else np.full_like(x, np.nan)
            auc_norm.append(float(np.trapezoid(y, x)) if np.all(np.isfinite(y)) else np.nan)
            auc_acc.append(float(np.trapezoid(ya, x)) if np.all(np.isfinite(ya)) else np.nan)
            below = [b for b, val in zip(budgets, ya) if np.isfinite(val) and val < 0.5]
            k_star_acc.append(min(below) if below else np.nan)
        out["normalized_degradation_margin_auc_logk"] = auc_norm
        out["degradation_accuracy_auc_logk"] = auc_acc
        out["k_star_accuracy_below_0p5"] = k_star_acc
    return out


def summarize_by_stage(cell: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "base_accuracy", "post_injection_accuracy", "post_continuation_accuracy",
        "base_mean_margin", "post_injection_mean_margin", "post_continuation_mean_margin",
        "uptake_margin_delta", "normalized_clean_retention_margin",
        "normalized_degradation_margin_auc_logk", "degradation_accuracy_auc_logk", "k_star_accuracy_below_0p5",
    ]
    metric_cols = [c for c in metric_cols if c in cell.columns]
    rows = []
    for (model, step, stage), g in cell.groupby(["model", "step", "stage"]):
        row = {"model": model, "step": step, "stage": stage, "n_seeds": g["seed"].nunique()}
        for c in metric_cols:
            row[f"{c}_mean"] = g[c].mean(skipna=True)
            row[f"{c}_std"] = g[c].std(skipna=True)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "step"])


def fit_simple_models(stage_summary: pd.DataFrame, metric_col: str, boundary_step: int) -> pd.DataFrame:
    """Compare simple monotone-ish log-step and fixed-boundary segmented models.

    This is only a diagnostic summary for the MVP, not the final pre-registered test.
    """
    rows = []
    for model, g in stage_summary.groupby("model"):
        if metric_col not in g.columns:
            continue
        y = g[metric_col].astype(float).to_numpy()
        steps = g["step"].astype(float).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 4:
            continue
        y = y[mask]
        steps = steps[mask]
        x = np.log10(steps + 1.0)
        # Linear-in-log-step baseline.
        X1 = np.column_stack([np.ones_like(x), x])
        beta1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid1 = y - X1 @ beta1
        sse1 = float(np.sum(resid1**2))
        # Fixed boundary model: pre/post indicator plus log step.
        post = (steps > boundary_step).astype(float)
        X2 = np.column_stack([np.ones_like(x), x, post])
        beta2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        resid2 = y - X2 @ beta2
        sse2 = float(np.sum(resid2**2))
        n = len(y)
        aic1 = n * math.log(max(sse1 / n, 1e-12)) + 2 * X1.shape[1]
        aic2 = n * math.log(max(sse2 / n, 1e-12)) + 2 * X2.shape[1]
        rows.append({
            "model": model,
            "metric": metric_col,
            "boundary_step": boundary_step,
            "n_points": n,
            "log_step_sse": sse1,
            "fixed_boundary_sse": sse2,
            "log_step_aic": aic1,
            "fixed_boundary_aic": aic2,
            "delta_aic_boundary_minus_logstep": aic2 - aic1,
            "boundary_model_better": aic2 < aic1,
        })
    return pd.DataFrame(rows)


def plot_stage_metric(stage_summary: pd.DataFrame, metric_mean_col: str, ylabel: str, outpath: Path) -> None:
    if metric_mean_col not in stage_summary.columns:
        return
    plt.figure(figsize=(8, 5))
    for model, g in stage_summary.groupby("model"):
        g = g.sort_values("step")
        plt.plot(g["step"], g[metric_mean_col], marker="o", label=model.split("/")[-1])
    plt.axvspan(128, 2000, alpha=0.12, label="E1/E2 candidate window")
    plt.xscale("symlog", linthresh=128)
    plt.xlabel("Checkpoint step")
    plt.ylabel(ylabel)
    plt.title(ylabel + " by injection stage")
    plt.legend(fontsize=8)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_report(root: Path, cell: pd.DataFrame, stage: pd.DataFrame, model_tests: pd.DataFrame, outpath: Path) -> None:
    lines: list[str] = []
    lines.append("# E3 factual-intervention MVP report")
    lines.append("")
    if cell.empty:
        lines.append("No completed E3 cells were found.")
    else:
        lines.append("## Coverage")
        lines.append("")
        coverage = cell.groupby(["model", "stage", "step"])["seed"].nunique().reset_index(name="n_seeds")
        lines.append(coverage.to_markdown(index=False))
        lines.append("")
        lines.append("## Headline stage summary")
        lines.append("")
        keep = [c for c in [
            "model", "stage", "step", "n_seeds",
            "post_injection_accuracy_mean", "uptake_margin_delta_mean",
            "normalized_clean_retention_margin_mean", "normalized_degradation_margin_auc_logk_mean",
            "k_star_accuracy_below_0p5_mean",
        ] if c in stage.columns]
        lines.append(stage[keep].to_markdown(index=False))
        lines.append("")
        lines.append("## Diagnostic model comparison")
        lines.append("")
        if model_tests.empty:
            lines.append("Not enough points for the fixed-boundary-vs-log-step diagnostic.")
        else:
            lines.append(model_tests.to_markdown(index=False))
        lines.append("")
        lines.append("## Interpretation guide")
        lines.append("")
        lines.append("This MVP should first be judged by uptake. Cells with weak post-injection uptake are not interpretable as durability tests. The main durability quantities are uptake-normalized clean retention and degradation-resistance AUC. A sensitive-period result would require a boundary-like change near the independently measured E1 window, not merely a smooth monotone early-to-late decline.")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E3 factual intervention results.")
    parser.add_argument("--root", default="results/e3_critical_period_intervention")
    parser.add_argument("--metrics", default="raw/e3_factual_metrics.csv")
    parser.add_argument("--boundary-step", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root)
    dirs = ensure_dirs(root)
    df = load_metrics(root, args.metrics)
    cell = pivot_cell_metrics(df)
    stage = summarize_by_stage(cell)
    test_metric = "normalized_clean_retention_margin_mean"
    model_tests = fit_simple_models(stage, test_metric, args.boundary_step) if test_metric in stage.columns else pd.DataFrame()

    cell_path = dirs["processed"] / "e3_factual_cell_summary.csv"
    stage_path = dirs["tables"] / "e3_factual_stage_summary.csv"
    test_path = dirs["processed"] / "e3_fixed_boundary_vs_logstep_diagnostic.csv"
    cell.to_csv(cell_path, index=False)
    stage.to_csv(stage_path, index=False)
    model_tests.to_csv(test_path, index=False)

    plot_stage_metric(stage, "post_injection_accuracy_mean", "Post-injection choice accuracy", dirs["figures"] / "e3_post_injection_accuracy_by_stage.png")
    plot_stage_metric(stage, "uptake_margin_delta_mean", "Uptake margin delta", dirs["figures"] / "e3_uptake_margin_delta_by_stage.png")
    plot_stage_metric(stage, "normalized_clean_retention_margin_mean", "Normalized clean retention margin", dirs["figures"] / "e3_normalized_clean_retention_by_stage.png")
    plot_stage_metric(stage, "normalized_degradation_margin_auc_logk_mean", "Normalized degradation margin AUC", dirs["figures"] / "e3_normalized_degradation_auc_by_stage.png")

    report_path = dirs["reports"] / "e3_factual_intervention_report.md"
    write_report(root, cell, stage, model_tests, report_path)
    print(f"Wrote {cell_path}")
    print(f"Wrote {stage_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
