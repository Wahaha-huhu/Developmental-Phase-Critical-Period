#!/usr/bin/env python3
"""Analyse and plot T1 toy calibration outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/t1_toy_calibration")
    args = ap.parse_args()
    root = Path(args.root)
    raw = root / "raw"
    figs = root / "figures"
    tables = root / "tables"
    reports = root / "reports"
    for d in [figs, tables, reports]:
        d.mkdir(parents=True, exist_ok=True)

    curve_path = raw / "t1_training_curve.csv"
    spec_path = raw / "t1_spectral_metrics.csv"
    ret_path = raw / "t1_intervention_retention.csv"

    summary_lines = ["# T1 toy calibration analysis", ""]

    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(curve["step"], curve["train_accuracy"], marker="o", label="train")
        plt.plot(curve["step"], curve["test_accuracy"], marker="o", label="held-out")
        plt.xlabel("Training step")
        plt.ylabel("Accuracy")
        plt.title("T1 behavioural transition")
        plt.legend()
        savefig(figs / "t1_train_vs_heldout_accuracy.png")
        summary_lines.append(f"- Final train accuracy: `{curve['train_accuracy'].iloc[-1]:.4f}`")
        summary_lines.append(f"- Final held-out accuracy: `{curve['test_accuracy'].iloc[-1]:.4f}`")
        # Transition proxy: first step where held-out accuracy reaches 80% of final heldout improvement.
        start = float(curve["test_accuracy"].iloc[0])
        final = float(curve["test_accuracy"].iloc[-1])
        target = start + 0.8 * (final - start)
        hit = curve[curve["test_accuracy"] >= target]
        if not hit.empty:
            transition_step = int(hit["step"].iloc[0])
            summary_lines.append(f"- Behavioural transition proxy, 80% of final held-out improvement: step `{transition_step}`")
    else:
        summary_lines.append("- Missing training curve CSV.")

    if spec_path.exists():
        spec = pd.read_csv(spec_path)
        for metric in ["stable_rank", "effective_rank", "subspace_stability_topk", "mp_outlier_proxy", "alpha_tail_proxy"]:
            if metric not in spec.columns:
                continue
            plt.figure(figsize=(7.5, 4.5))
            for matrix, sdf in spec.groupby("matrix"):
                sdf = sdf.sort_values("step")
                plt.plot(sdf["step"], sdf[metric], marker="o", label=matrix)
            plt.xlabel("Training step")
            plt.ylabel(metric.replace("_", " "))
            plt.title(f"T1 spectral indicator: {metric}")
            plt.legend(fontsize=8)
            savefig(figs / f"t1_{metric}.png")
        compact = spec.groupby("step", as_index=False)[[c for c in ["stable_rank", "effective_rank", "mp_outlier_proxy", "alpha_tail_proxy"] if c in spec.columns]].mean()
        compact.to_csv(tables / "t1_spectral_summary_by_step.csv", index=False)
        summary_lines.append(f"- Spectral rows: `{len(spec)}`")
    else:
        summary_lines.append("- Missing spectral metrics CSV.")

    if ret_path.exists():
        ret = pd.read_csv(ret_path)
        ret.to_csv(tables / "t1_intervention_retention_summary.csv", index=False)
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(ret["checkpoint_step"], ret["normalized_retention"], marker="o")
        plt.xlabel("Injection checkpoint step")
        plt.ylabel("Normalised retention after washout")
        plt.title("T1 injection retention by training stage")
        savefig(figs / "t1_intervention_retention_by_checkpoint.png")
        summary_lines.append("- Intervention retention table generated.")
        if "normalized_retention" in ret.columns:
            best = ret.sort_values("normalized_retention", ascending=False).head(1).iloc[0]
            summary_lines.append(f"- Highest normalised retention checkpoint: step `{int(best['checkpoint_step'])}` with retention `{best['normalized_retention']:.4f}`")
    else:
        summary_lines.append("- No intervention-retention CSV found.")

    summary_lines.append("")
    summary_lines.append("Interpretation note: T1 is a calibration/sanity-check experiment. It should be used to validate indicator behaviour under controlled conditions, not to claim that Pythia has the same mechanism.")
    (reports / "t1_analysis_report.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote T1 analysis artifacts to {root}")


if __name__ == "__main__":
    main()
