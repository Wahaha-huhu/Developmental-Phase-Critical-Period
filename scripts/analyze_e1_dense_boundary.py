#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


METRIC_CANDIDATES = [
    "stable_rank",
    "effective_rank",
    "spectral_norm",
    "frobenius_norm",
    "mp_outliers",
    "alpha",
    "alpha_tail_frac_0.3",
    "subspace_stability_topk",
    "subspace_stability",
]

CHANGE_METRICS_SIGN = {
    # For these, drops indicate consolidation; use -delta as transition strength.
    "stable_rank": -1.0,
    "effective_rank": -1.0,
    "alpha": -1.0,
    "alpha_tail_frac_0.3": -1.0,
    # For these, increases are transition strength by default.
    "spectral_norm": 1.0,
    "frobenius_norm": 1.0,
    "mp_outliers": 1.0,
    # Subspace stability can mean similarity, so drops are movement. If the script's metric
    # is named as stability, use -delta. If users store instability, rename it upstream.
    "subspace_stability_topk": -1.0,
    "subspace_stability": -1.0,
}


def parse_step_num(x) -> int:
    if pd.isna(x):
        return -1
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    s = str(x)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def infer_model(metrics_path: Path, requested: str | None, df: pd.DataFrame) -> str:
    if requested:
        return requested
    # Try existing metadata-ish columns first.
    for c in ["model_name", "model_id", "repo", "repo_id"]:
        if c in df.columns and df[c].notna().any():
            return str(df[c].dropna().iloc[0])
    text = str(metrics_path)
    low = text.lower()
    if "160m" in low:
        return "EleutherAI/pythia-160m-deduped"
    if "410m" in low:
        return "EleutherAI/pythia-410m-deduped"
    if "1b" in low:
        return "EleutherAI/pythia-1b-deduped"
    if "70m" in low:
        return "EleutherAI/pythia-70m-deduped"
    return "single_model"


def ensure_schema(df: pd.DataFrame, metrics_path: Path, model: str | None) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicated column names if any were created by previous patch attempts.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "model" not in df.columns:
        df["model"] = infer_model(metrics_path, model, df)
    if "step_num" not in df.columns:
        if "step" in df.columns:
            df["step_num"] = df["step"].apply(parse_step_num)
        elif "checkpoint" in df.columns:
            df["step"] = df["checkpoint"]
            df["step_num"] = df["step"].apply(parse_step_num)
        else:
            raise KeyError("CSV needs either a 'step' or 'step_num' column.")
    if "step" not in df.columns:
        df["step"] = "step" + df["step_num"].astype(str)
    if "module" not in df.columns:
        # Try common alternatives.
        for alt in ["module_name", "matrix", "matrix_name", "parameter", "param_name"]:
            if alt in df.columns:
                df["module"] = df[alt].astype(str)
                break
        else:
            df["module"] = "all_modules"
    if "layer" not in df.columns:
        df["layer"] = 0
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    # Already long format.
    if {"metric", "value"}.issubset(df.columns):
        out = df.copy()
        out["metric"] = out["metric"].astype(str)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.dropna(subset=["value"])

    metric_cols = [c for c in METRIC_CANDIDATES if c in df.columns]
    # Add any numeric columns that look like metrics and are not identifiers.
    id_cols = {
        "model", "model_name", "model_id", "repo", "repo_id", "step", "step_num",
        "checkpoint", "layer", "module", "module_name", "matrix", "matrix_name",
        "parameter", "param_name"
    }
    for c in df.columns:
        if c in id_cols or c in metric_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and any(k in c.lower() for k in ["rank", "norm", "outlier", "alpha", "stability"]):
            metric_cols.append(c)
    if not metric_cols:
        raise ValueError(
            "Could not find metric columns. Expected either long format with columns "
            "['metric','value'] or wide columns such as stable_rank/effective_rank/spectral_norm."
        )

    id_vars = [c for c in ["model", "step", "step_num", "layer", "module"] if c in df.columns]
    out = df.melt(id_vars=id_vars, value_vars=metric_cols, var_name="metric", value_name="value")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])


def adjacent_changes(long_df: pd.DataFrame) -> pd.DataFrame:
    # Average over layers first, preserving module-level trajectories.
    traj = (
        long_df.groupby(["model", "module", "metric", "step_num"], as_index=False)["value"]
        .mean()
        .sort_values(["model", "module", "metric", "step_num"])
    )
    rows = []
    for (model, module, metric), g in traj.groupby(["model", "module", "metric"], sort=False):
        g = g.sort_values("step_num")
        steps = g["step_num"].to_numpy()
        vals = g["value"].to_numpy(dtype=float)
        if len(steps) < 2:
            continue
        for i in range(1, len(steps)):
            prev_step, next_step = int(steps[i-1]), int(steps[i])
            prev_val, next_val = vals[i-1], vals[i]
            delta = next_val - prev_val
            rel_delta = delta / (abs(prev_val) + 1e-12)
            sign = CHANGE_METRICS_SIGN.get(str(metric), 1.0)
            transition_strength = sign * rel_delta
            rows.append({
                "model": model,
                "module": module,
                "metric": metric,
                "prev_step": prev_step,
                "next_step": next_step,
                "interval": f"{prev_step}->{next_step}",
                "prev_value": prev_val,
                "next_value": next_val,
                "delta": delta,
                "relative_delta": rel_delta,
                "transition_strength": transition_strength,
                "abs_relative_delta": abs(rel_delta),
            })
    return pd.DataFrame(rows)


def boundary_votes(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return changes
    # For each model/module/metric choose the interval with highest signed transition strength.
    idx = changes.groupby(["model", "module", "metric"])["transition_strength"].idxmax()
    winners = changes.loc[idx].copy()
    votes = (
        winners.groupby(["model", "interval", "prev_step", "next_step"], as_index=False)
        .agg(
            votes=("metric", "size"),
            mean_transition_strength=("transition_strength", "mean"),
            metrics=("metric", lambda x: ";".join(sorted(set(map(str, x))))),
            modules=("module", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .sort_values(["model", "votes", "mean_transition_strength"], ascending=[True, False, False])
    )
    return votes


def aggregate_interval_strength(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return changes
    return (
        changes.groupby(["model", "interval", "prev_step", "next_step"], as_index=False)
        .agg(
            mean_transition_strength=("transition_strength", "mean"),
            median_transition_strength=("transition_strength", "median"),
            mean_abs_relative_delta=("abs_relative_delta", "mean"),
            n=("transition_strength", "size"),
        )
        .sort_values(["model", "mean_transition_strength"], ascending=[True, False])
    )


def plot_interval_strength(df: pd.DataFrame, out: Path):
    if plt is None or df.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    for model, g in df.groupby("model"):
        g = g.sort_values("prev_step")
        labels = g["interval"].astype(str).tolist()
        vals = g["mean_transition_strength"].to_numpy()
        fig = plt.figure(figsize=(max(8, len(labels) * 0.22), 4.5))
        ax = fig.add_subplot(111)
        ax.bar(range(len(labels)), vals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Mean signed transition strength")
        ax.set_title(f"Dense E1 interval strength — {model}")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model))
        fig.savefig(out / f"e1_dense_interval_strength_{safe}.png", dpi=180)
        plt.close(fig)


def write_report(out: Path, metrics_path: Path, df: pd.DataFrame, votes: pd.DataFrame, interval_strength: pd.DataFrame):
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# E1 dense boundary analysis report\n")
    lines.append(f"- Metrics: `{metrics_path}`")
    lines.append(f"- Rows: {len(df)}")
    lines.append(f"- Models: {', '.join(map(str, sorted(df['model'].unique())))}")
    lines.append(f"- Checkpoints: {df['step_num'].nunique()} ({int(df['step_num'].min())} to {int(df['step_num'].max())})")
    lines.append(f"- Modules: {df['module'].nunique()}")
    lines.append(f"- Metrics: {df['metric'].nunique()}\n")

    if not votes.empty:
        lines.append("## Top boundary-vote intervals\n")
        for model, g in votes.groupby("model"):
            top = g.sort_values(["votes", "mean_transition_strength"], ascending=[False, False]).head(5)
            lines.append(f"### {model}\n")
            for _, r in top.iterrows():
                lines.append(
                    f"- `{r['interval']}`: {int(r['votes'])} votes, "
                    f"mean strength {r['mean_transition_strength']:.4g}"
                )
            lines.append("")
    if not interval_strength.empty:
        lines.append("## Top aggregate-strength intervals\n")
        for model, g in interval_strength.groupby("model"):
            top = g.sort_values("mean_transition_strength", ascending=False).head(5)
            lines.append(f"### {model}\n")
            for _, r in top.iterrows():
                lines.append(
                    f"- `{r['interval']}`: mean signed strength {r['mean_transition_strength']:.4g}, "
                    f"mean abs relative delta {r['mean_abs_relative_delta']:.4g}, n={int(r['n'])}"
                )
            lines.append("")

    (out / "reports").mkdir(exist_ok=True)
    (out / "reports" / "e1_dense_boundary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Robust dense E1 boundary analysis; tolerant of missing model column.")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="Optional model name if the CSV lacks a model column.")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "processed").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    raw = pd.read_csv(metrics_path)
    raw = ensure_schema(raw, metrics_path, args.model)
    long_df = to_long(raw)
    long_df = ensure_schema(long_df, metrics_path, args.model)
    long_df.to_csv(out / "processed" / "e1_dense_metrics_long.csv", index=False)

    coverage = (
        long_df.groupby(["model", "step_num"], as_index=False)
        .agg(rows=("value", "size"), modules=("module", "nunique"), metrics=("metric", "nunique"), layers=("layer", "nunique"))
        .sort_values(["model", "step_num"])
    )
    coverage.to_csv(out / "tables" / "e1_dense_checkpoint_coverage.csv", index=False)

    changes = adjacent_changes(long_df)
    changes.to_csv(out / "processed" / "e1_dense_adjacent_changes.csv", index=False)

    votes = boundary_votes(changes)
    votes.to_csv(out / "tables" / "e1_dense_boundary_vote_table.csv", index=False)

    interval_strength = aggregate_interval_strength(changes)
    interval_strength.to_csv(out / "tables" / "e1_dense_interval_strength.csv", index=False)

    plot_interval_strength(interval_strength, out / "figures")
    write_report(out, metrics_path, long_df, votes, interval_strength)

    print("Wrote:", out)
    print("Report:", out / "reports" / "e1_dense_boundary_report.md")


if __name__ == "__main__":
    main()
