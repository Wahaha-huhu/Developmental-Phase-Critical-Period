#!/usr/bin/env python3
"""Analyze E2-lite functional grounding outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e2_functional_grounding")
    args = ap.parse_args()

    root = Path(args.root)
    raw_path = root / "raw" / "e2_functional_metrics.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing raw metrics file: {raw_path}")

    tables = root / "tables"
    processed = root / "processed"
    figures = root / "figures"
    reports = root / "reports"
    manifests = root / "manifests"
    for d in [tables, processed, figures, reports, manifests]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df["step"] = df["step"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    coverage = (
        df.groupby(["model", "checkpoint", "step", "eval_type", "task", "metric"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["model", "step", "eval_type", "task", "metric"])
    )
    coverage.to_csv(tables / "e2_coverage.csv", index=False)

    summary = (
        df.groupby(["model", "step", "eval_type", "task", "metric"], dropna=False)
        .agg(mean_value=("value", "mean"), std_value=("value", "std"), n=("value", "count"))
        .reset_index()
        .sort_values(["model", "eval_type", "task", "metric", "step"])
    )
    summary.to_csv(tables / "e2_metric_summary_by_checkpoint.csv", index=False)

    # Adjacent changes by model/task/metric.
    rows = []
    for keys, g in summary.groupby(["model", "eval_type", "task", "metric"]):
        g = g.sort_values("step")
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                rows.append({
                    "model": keys[0],
                    "eval_type": keys[1],
                    "task": keys[2],
                    "metric": keys[3],
                    "from_step": int(prev["step"]),
                    "to_step": int(r["step"]),
                    "delta": float(r["mean_value"] - prev["mean_value"]),
                    "abs_delta": abs(float(r["mean_value"] - prev["mean_value"])),
                })
            prev = r
    changes = pd.DataFrame(rows)
    changes.to_csv(processed / "e2_adjacent_checkpoint_changes.csv", index=False)

    if not changes.empty:
        top_changes = changes.sort_values("abs_delta", ascending=False).head(80)
        top_changes.to_csv(tables / "e2_top_adjacent_changes.csv", index=False)

    # Plot all summary trajectories.
    for (model, eval_type, task, metric), g in summary.groupby(["model", "eval_type", "task", "metric"]):
        g = g.sort_values("step")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(g["step"], g["mean_value"], marker="o")
        ax.axvspan(128, 2000, alpha=0.12, label="E1 candidate window")
        ax.axvline(512, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(2000, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"{model}\n{task}: {metric}")
        ax.set_xlabel("training step")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        fig.tight_layout()
        out = figures / f"e2_{safe_model_name(model)}_{task}_{metric}_linear.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        x = g["step"].replace(0, 1)
        ax.plot(x, g["mean_value"], marker="o")
        ax.axvspan(128, 2000, alpha=0.12, label="E1 candidate window")
        ax.axvline(512, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(2000, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("log")
        ax.set_title(f"{model}\n{task}: {metric} (log step)")
        ax.set_xlabel("training step, step0 shown as 1")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        fig.tight_layout()
        out = figures / f"e2_{safe_model_name(model)}_{task}_{metric}_logstep.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)

    # Make a compact report.
    report_lines = []
    report_lines.append("# E2-lite functional grounding report\n")
    report_lines.append("## Coverage\n")
    for model, g in df.groupby("model"):
        report_lines.append(f"- `{model}`: {len(g)} metric rows, {g['step'].nunique()} checkpoints.\n")
    report_lines.append("\n## Interpretation guide\n")
    report_lines.append(
        "E2-lite is observational. It asks whether lightweight behavioural/log-likelihood measures change near the E1 candidate window (approximately 128--2000 steps). It does not establish a sensitive or critical period.\n"
    )
    report_lines.append("\n## Largest adjacent changes\n")
    if not changes.empty:
        for _, r in changes.sort_values("abs_delta", ascending=False).head(12).iterrows():
            report_lines.append(
                f"- `{r['model']}` `{r['task']}` `{r['metric']}`: {int(r['from_step'])}→{int(r['to_step'])}, delta={r['delta']:.4g}.\n"
            )
    else:
        report_lines.append("No adjacent-change table could be computed.\n")
    report_lines.append("\n## Thesis status\n")
    report_lines.append(
        "Use this report to decide whether the E1 geometric phase has a functional correlate. Strong support would require systematic loss/probe improvement near the E1 window rather than only at the final checkpoint. Smooth or noisy curves should be reported as weak/partial functional grounding.\n"
    )
    (reports / "e2_functional_grounding_report.md").write_text("".join(report_lines))

    manifest = pd.DataFrame([
        {
            "experiment_id": "E2",
            "artifact_type": "table",
            "path": str(tables / "e2_metric_summary_by_checkpoint.csv"),
            "thesis_section": "Results / E2 functional grounding",
            "caption_draft": "Checkpoint-wise summary of E2-lite behavioural and log-likelihood metrics.",
            "source_data": str(raw_path),
            "code_entrypoint": "scripts/analyze_e2_results.py",
            "status": "candidate",
            "notes": "Observational functional grounding only; not a causal test.",
        },
        {
            "experiment_id": "E2",
            "artifact_type": "report",
            "path": str(reports / "e2_functional_grounding_report.md"),
            "thesis_section": "Results / E2 functional grounding",
            "caption_draft": "Automatic report summarising E2-lite checkpoint behaviour.",
            "source_data": str(raw_path),
            "code_entrypoint": "scripts/analyze_e2_results.py",
            "status": "candidate",
            "notes": "Use after manual review.",
        },
    ])
    manifest.to_csv(manifests / "artifact_manifest.csv", index=False)

    print(f"Wrote E2 analysis outputs under {root}")


if __name__ == "__main__":
    main()
