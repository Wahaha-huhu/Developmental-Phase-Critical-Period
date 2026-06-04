#!/usr/bin/env python3
"""Create thesis-oriented multi-model summaries for E2 functional grounding.

This script reads results/e2_functional_grounding/raw/e2_functional_metrics.csv and
produces compact cross-model tables/figures focused on the two most interpretable
signals from E2-lite: fixed-text NLL and syntax-regularity gold-logprob margins.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = [
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
]


def short_model_name(name: str) -> str:
    if "pythia-" in name:
        return name.split("pythia-")[-1].replace("-deduped", "")
    return name.replace("EleutherAI/", "")


def safe_num(x):
    return pd.to_numeric(x, errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e2_functional_grounding")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    raw_path = Path(args.raw) if args.raw else root / "raw" / "e2_functional_metrics.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing raw E2 metrics: {raw_path}")

    tables = root / "tables"
    processed = root / "processed"
    figures = root / "figures"
    reports = root / "reports"
    manifests = root / "manifests"
    for d in [tables, processed, figures, reports, manifests]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df["step"] = safe_num(df["step"]).astype(int)
    df["value"] = safe_num(df["value"])
    df["model_short"] = df["model"].map(short_model_name)

    coverage = (
        df.groupby(["model", "eval_type", "task", "metric"])
        .agg(n_rows=("value", "count"), n_checkpoints=("step", "nunique"))
        .reset_index()
        .sort_values(["model", "eval_type", "task", "metric"])
    )
    coverage.to_csv(tables / "e2_multimodel_coverage.csv", index=False)

    summary = (
        df.groupby(["model", "model_short", "step", "eval_type", "task", "metric"], dropna=False)
        .agg(mean_value=("value", "mean"), std_value=("value", "std"), n=("value", "count"))
        .reset_index()
        .sort_values(["model", "eval_type", "task", "metric", "step"])
    )
    summary.to_csv(tables / "e2_multimodel_metric_summary_by_checkpoint.csv", index=False)

    # Fixed-text NLL progress: how much of total step0->final improvement is achieved by step2000.
    nll = summary[(summary["eval_type"] == "lm_loss") & (summary["task"] == "fixed_text") & (summary["metric"] == "nll")].copy()
    nll_progress_rows = []
    for model, g in nll.groupby("model"):
        g = g.sort_values("step")
        vals = dict(zip(g["step"], g["mean_value"]))
        if 0 not in vals:
            continue
        final_step = int(max(vals))
        final_val = vals[final_step]
        denom = vals[0] - final_val
        for step in sorted(vals):
            frac = float("nan") if abs(denom) < 1e-12 else (vals[0] - vals[step]) / denom
            nll_progress_rows.append({
                "model": model,
                "model_short": short_model_name(model),
                "step": step,
                "nll": vals[step],
                "final_step": final_step,
                "fraction_of_total_nll_reduction": frac,
            })
    nll_progress = pd.DataFrame(nll_progress_rows)
    nll_progress.to_csv(processed / "e2_fixed_text_nll_progress.csv", index=False)

    # Syntax margin progress.
    syntax = summary[(summary["task"] == "syntax_regularities") & (summary["metric"] == "gold_logprob_margin")].copy()
    syntax.to_csv(processed / "e2_syntax_margin_summary.csv", index=False)

    # Adjacent changes for headline metrics.
    headline = pd.concat([nll, syntax], ignore_index=True)
    rows = []
    for keys, g in headline.groupby(["model", "eval_type", "task", "metric"]):
        g = g.sort_values("step")
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                rows.append({
                    "model": keys[0],
                    "model_short": short_model_name(keys[0]),
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
    changes.to_csv(processed / "e2_headline_adjacent_changes.csv", index=False)

    # Overlay plots.
    if not nll.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for model in MODEL_ORDER + [m for m in sorted(nll["model"].unique()) if m not in MODEL_ORDER]:
            g = nll[nll["model"] == model].sort_values("step")
            if g.empty:
                continue
            ax.plot(g["step"], g["mean_value"], marker="o", label=short_model_name(model))
        ax.axvspan(128, 2000, alpha=0.12, label="E1 window")
        ax.axvline(512, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(2000, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("symlog", linthresh=128)
        ax.set_xlabel("training step")
        ax.set_ylabel("fixed-text NLL")
        ax.set_title("E2 fixed-text NLL across Pythia checkpoints")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "e2_multimodel_fixed_text_nll.png", dpi=200)
        plt.close(fig)

    if not syntax.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for model in MODEL_ORDER + [m for m in sorted(syntax["model"].unique()) if m not in MODEL_ORDER]:
            g = syntax[syntax["model"] == model].sort_values("step")
            if g.empty:
                continue
            ax.plot(g["step"], g["mean_value"], marker="o", label=short_model_name(model))
        ax.axhline(0, linewidth=1, alpha=0.5)
        ax.axvspan(128, 2000, alpha=0.12, label="E1 window")
        ax.axvline(512, linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(2000, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("symlog", linthresh=128)
        ax.set_xlabel("training step")
        ax.set_ylabel("gold-vs-best-wrong logprob margin")
        ax.set_title("E2 syntax-regularity margin across Pythia checkpoints")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "e2_multimodel_syntax_margin.png", dpi=200)
        plt.close(fig)

    # Compact report.
    lines = []
    lines.append("# E2 multi-model functional grounding summary\n\n")
    lines.append("## Coverage\n\n")
    for model, g in df.groupby("model"):
        lines.append(f"- `{model}`: {len(g)} rows, {g['step'].nunique()} checkpoints.\n")

    lines.append("\n## Fixed-text NLL progress\n\n")
    if not nll_progress.empty:
        for model, g in nll_progress.groupby("model"):
            vals = dict(zip(g["step"], g["fraction_of_total_nll_reduction"]))
            nll_vals = dict(zip(g["step"], g["nll"]))
            msg = f"- `{model}`: step0 NLL={nll_vals.get(0, float('nan')):.3f}"
            if 2000 in nll_vals:
                msg += f", step2000 NLL={nll_vals[2000]:.3f}, fraction of total reduction by step2000={vals.get(2000, float('nan')):.3f}"
            msg += ".\n"
            lines.append(msg)
    else:
        lines.append("No fixed-text NLL rows found.\n")

    lines.append("\n## Syntax margin\n\n")
    if not syntax.empty:
        for model, g in syntax.groupby("model"):
            vals = dict(zip(g["step"], g["mean_value"]))
            msg = f"- `{model}`:"
            for step in [0, 128, 512, 1000, 2000, 3000, 8000, 143000]:
                if step in vals:
                    msg += f" step{step}={vals[step]:.3f};"
            lines.append(msg.rstrip(";") + ".\n")
    else:
        lines.append("No syntax margin rows found.\n")

    lines.append("\n## Interpretation\n\n")
    lines.append(
        "Use this summary to judge whether the lightweight functional measures move in the same early interval as E1. "
        "A consistent result across 410M/1B would strengthen the claim that E1's weight-space reorganisation has functional correlates. "
        "This remains observational evidence, not a causal sensitive-period test.\n"
    )

    (reports / "e2_multimodel_summary.md").write_text("".join(lines))

    manifest = pd.DataFrame([
        {
            "experiment_id": "E2",
            "artifact_type": "figure",
            "path": str(figures / "e2_multimodel_fixed_text_nll.png"),
            "thesis_section": "Results / E2 functional grounding",
            "caption_draft": "Fixed-text next-token negative log-likelihood over Pythia checkpoints, with the E1 candidate window shaded.",
            "source_data": str(raw_path),
            "code_entrypoint": "scripts/summarize_e2_multimodel.py",
            "status": "candidate",
            "notes": "Observational functional grounding only.",
        },
        {
            "experiment_id": "E2",
            "artifact_type": "figure",
            "path": str(figures / "e2_multimodel_syntax_margin.png"),
            "thesis_section": "Results / E2 functional grounding",
            "caption_draft": "Syntax-regularity gold-vs-best-wrong log-probability margins over Pythia checkpoints, with the E1 candidate window shaded.",
            "source_data": str(raw_path),
            "code_entrypoint": "scripts/summarize_e2_multimodel.py",
            "status": "candidate",
            "notes": "Probe set is small; interpret as lightweight functional grounding.",
        },
    ])
    manifest.to_csv(manifests / "e2_multimodel_artifact_manifest.csv", index=False)

    print(f"Wrote E2 multi-model summary outputs under {root}")


if __name__ == "__main__":
    main()
