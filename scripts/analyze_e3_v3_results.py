#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trapz(xs, ys):
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(np.asarray(ys, dtype=float), np.asarray(xs, dtype=float)))


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    raw = root / "raw"
    tables = root / "tables"
    figs = root / "figures"
    reports = root / "reports"
    for p in [tables, figs, reports]:
        p.mkdir(parents=True, exist_ok=True)

    summary = safe_read(raw / "e3_factual_cell_summary.csv")
    deg = safe_read(raw / "e3_factual_degradation_curve.csv")
    if summary.empty:
        raise SystemExit(f"No summary file found at {raw / 'e3_factual_cell_summary.csv'}")

    summary.to_csv(tables / "e3_v3_cell_summary.csv", index=False)
    group_cols = ["model", "stage", "step"]
    metrics = [
        "base_probe_accuracy", "post_injection_accuracy", "uptake_accuracy_delta",
        "base_probe_mean_margin", "post_injection_mean_margin", "uptake_margin_delta",
        "normalized_retention_margin", "normalized_retention_accuracy",
        "k_star_accuracy_threshold", "degradation_auc_accuracy", "degradation_auc_margin",
    ]
    use_metrics = [m for m in metrics if m in summary.columns]
    stage_summary = summary.groupby(group_cols, dropna=False)[use_metrics].agg(["mean", "std", "count"]).reset_index()
    stage_summary.to_csv(tables / "e3_v3_stage_summary.csv", index=False)

    # Long-form for easy plotting/review.
    long_rows = []
    for _, r in summary.iterrows():
        for m in use_metrics:
            long_rows.append({"model": r["model"], "stage": r["stage"], "step": r["step"], "seed": r["seed"], "metric": m, "value": r[m]})
    pd.DataFrame(long_rows).to_csv(tables / "e3_v3_summary_long.csv", index=False)

    # Plots.
    plot_metrics = ["uptake_margin_delta", "normalized_retention_margin", "k_star_accuracy_threshold", "degradation_auc_accuracy"]
    for m in plot_metrics:
        if m not in summary.columns:
            continue
        df = summary.copy()
        df["step_num"] = pd.to_numeric(df["step"], errors="coerce")
        agg = df.groupby("step_num")[m].agg(["mean", "std", "count"]).reset_index().sort_values("step_num")
        plt.figure(figsize=(8, 4.5))
        plt.errorbar(agg["step_num"], agg["mean"], yerr=agg["std"].fillna(0), marker="o", capsize=3)
        plt.xscale("symlog", linthresh=128)
        plt.xlabel("Injection checkpoint step")
        plt.ylabel(m)
        plt.title(f"E3 factual v3: {m}")
        plt.tight_layout()
        plt.savefig(figs / f"e3_v3_{m}.png", dpi=180)
        plt.close()

    # Degradation curve plot.
    if not deg.empty:
        deg["step_num"] = pd.to_numeric(deg["step"], errors="coerce")
        for metric in ["probe_accuracy", "probe_mean_margin"]:
            plt.figure(figsize=(8, 4.5))
            for step, g in deg.groupby("step_num"):
                gg = g.groupby("poison_budget")[metric].mean().reset_index().sort_values("poison_budget")
                plt.plot(gg["poison_budget"], gg[metric], marker="o", label=f"step{int(step)}")
            plt.xscale("symlog", linthresh=4)
            plt.xlabel("Poison budget k")
            plt.ylabel(metric)
            plt.title(f"E3 factual v3 degradation: {metric}")
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(figs / f"e3_v3_degradation_{metric}.png", dpi=180)
            plt.close()

    n_cells = len(summary)
    n_positive_uptake = int((summary.get("uptake_margin_delta", pd.Series(dtype=float)) > 0).sum())
    mean_uptake = float(summary.get("uptake_margin_delta", pd.Series(dtype=float)).mean()) if "uptake_margin_delta" in summary else math.nan
    report = []
    report.append("# E3 factual v3 analysis report\n")
    report.append(f"- Root: `{root}`")
    report.append(f"- Cells: {n_cells}")
    report.append(f"- Positive uptake cells by margin: {n_positive_uptake}/{n_cells}")
    report.append(f"- Mean uptake margin delta: {mean_uptake:.4f}" if not math.isnan(mean_uptake) else "- Mean uptake margin delta: NA")
    report.append("\n## Interpretation guardrails\n")
    report.append("A cell is interpretable for durability only if uptake is positive and preferably comparable across stages. If uptake is weak or highly variable, normalize retention/degradation by uptake and treat absolute durability cautiously.")
    report.append("\n## Output files\n")
    report.append("- `tables/e3_v3_cell_summary.csv`")
    report.append("- `tables/e3_v3_stage_summary.csv`")
    report.append("- `figures/e3_v3_uptake_margin_delta.png`")
    report.append("- `figures/e3_v3_normalized_retention_margin.png`")
    report.append("- `figures/e3_v3_k_star_accuracy_threshold.png`")
    report.append("- `figures/e3_v3_degradation_auc_accuracy.png`")
    (reports / "e3_v3_analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    print(reports / "e3_v3_analysis_report.md")


if __name__ == "__main__":
    main()
