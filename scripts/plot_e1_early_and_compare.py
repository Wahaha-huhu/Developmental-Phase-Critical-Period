#!/usr/bin/env python3
"""Create thesis-oriented E1 early-window plots and cross-model summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    fig = root / "figures"
    tab = root / "tables"
    proc = root / "processed"
    for d in [fig, tab, proc]:
        d.mkdir(parents=True, exist_ok=True)
    return fig, tab, proc


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def metric_col(df: pd.DataFrame, wanted: str) -> str | None:
    if wanted in df.columns:
        return wanted
    # Handle either raw or older metric naming.
    aliases = {
        "mp_outliers": ["mp_outliers_x1", "mp_outlier_count", "mp_outlier_proxy"],
        "alpha": ["alpha_tail_frac_0.3", "alpha_tail_proxy"],
    }
    for c in aliases.get(wanted, []):
        if c in df.columns:
            return c
    return None


def add_step_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "step_num" not in df.columns:
        if "checkpoint" in df.columns:
            src = df["checkpoint"].astype(str)
        elif "step" in df.columns:
            src = df["step"].astype(str)
        else:
            raise ValueError("Need a checkpoint or step column")
        df["step_num"] = src.str.replace("step", "", regex=False).astype(int)
    return df


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = metric_col(df, metric)
    if col is None:
        raise ValueError(f"Could not find metric column for {metric}")
    group_cols = ["model", "step_num", "module"] if "module" in df.columns else ["model", "step_num", "matrix"]
    label_col = "module" if "module" in df.columns else "matrix"
    out = df.groupby(group_cols, as_index=False)[col].mean()
    out = out.rename(columns={col: metric, label_col: "module"})
    return out


def plot_metric_early(df: pd.DataFrame, metric: str, fig_dir: Path, max_step: int) -> None:
    agg = aggregate(df, metric)
    agg = agg[agg["step_num"] <= max_step]
    if agg.empty:
        return
    for model, mdf in agg.groupby("model"):
        plt.figure(figsize=(8, 4.8))
        for module, sdf in mdf.groupby("module"):
            sdf = sdf.sort_values("step_num")
            plt.plot(sdf["step_num"], sdf[metric], marker="o", label=module)
        plt.xlabel("Training step")
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"E1 early-window {metric}: {model}")
        plt.legend(fontsize=8)
        safe_model = str(model).replace("/", "__")
        savefig(fig_dir / f"e1_{safe_model}_{metric}_early_{max_step}.png")


def plot_normalised_stable_rank(df: pd.DataFrame, fig_dir: Path, max_step: int) -> None:
    col = metric_col(df, "stable_rank")
    if col is None:
        return
    label_col = "module" if "module" in df.columns else "matrix"
    agg = df.groupby(["model", "step_num", label_col], as_index=False)[col].mean().rename(columns={label_col: "module", col: "stable_rank"})
    agg = agg[agg["step_num"] <= max_step]
    rows = []
    for (model, module), sdf in agg.groupby(["model", "module"]):
        sdf = sdf.sort_values("step_num").copy()
        base = sdf.loc[sdf["step_num"] == sdf["step_num"].min(), "stable_rank"].iloc[0]
        sdf["stable_rank_relative_to_first_checkpoint"] = sdf["stable_rank"] / base if base else np.nan
        rows.append(sdf)
    if not rows:
        return
    rel = pd.concat(rows, ignore_index=True)
    for model, mdf in rel.groupby("model"):
        plt.figure(figsize=(8, 4.8))
        for module, sdf in mdf.groupby("module"):
            sdf = sdf.sort_values("step_num")
            plt.plot(sdf["step_num"], sdf["stable_rank_relative_to_first_checkpoint"], marker="o", label=module)
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("Training step")
        plt.ylabel("Stable rank / first checkpoint stable rank")
        plt.title(f"E1 early-window relative stable-rank change: {model}")
        plt.legend(fontsize=8)
        safe_model = str(model).replace("/", "__")
        savefig(fig_dir / f"e1_{safe_model}_stable_rank_relative_early_{max_step}.png")


def adjacent_changes(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = metric_col(df, metric)
    if col is None:
        return pd.DataFrame()
    label_col = "module" if "module" in df.columns else "matrix"
    agg = df.groupby(["model", "step_num", label_col], as_index=False)[col].mean().rename(columns={label_col: "module", col: metric})
    rows = []
    for (model, module), sdf in agg.groupby(["model", "module"]):
        sdf = sdf.sort_values("step_num")
        vals = sdf[metric].to_numpy()
        steps = sdf["step_num"].to_numpy()
        for i in range(1, len(sdf)):
            prev, cur = vals[i - 1], vals[i]
            abs_delta = cur - prev
            rel_delta = abs_delta / (abs(prev) + 1e-12)
            rows.append({
                "model": model,
                "module": module,
                "metric": metric,
                "from_step": int(steps[i - 1]),
                "to_step": int(steps[i]),
                "abs_delta": float(abs_delta),
                "rel_delta": float(rel_delta),
                "abs_rel_delta": float(abs(rel_delta)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/e1_phase_identification/raw/e1_spectral_metrics.csv")
    ap.add_argument("--output-root", default="results/e1_phase_identification")
    ap.add_argument("--max-step", type=int, default=8000)
    args = ap.parse_args()
    out_root = Path(args.output_root)
    fig_dir, tab_dir, proc_dir = ensure_dirs(out_root)
    df = pd.read_csv(args.metrics)
    df = add_step_numeric(df)
    if "model" not in df.columns:
        df["model"] = "unknown_model"

    for metric in ["stable_rank", "effective_rank", "mp_outliers", "alpha", "subspace_stability_topk"]:
        try:
            plot_metric_early(df, metric, fig_dir, args.max_step)
        except Exception as e:
            print(f"Skipping {metric}: {e}")
    plot_normalised_stable_rank(df, fig_dir, args.max_step)

    change_tables = []
    for metric in ["stable_rank", "effective_rank", "subspace_stability_topk", "mp_outliers", "alpha"]:
        c = adjacent_changes(df, metric)
        if not c.empty:
            change_tables.append(c)
    if change_tables:
        changes = pd.concat(change_tables, ignore_index=True)
        changes.to_csv(proc_dir / "e1_adjacent_changes_for_replication.csv", index=False)
        summary = (
            changes.groupby(["model", "from_step", "to_step"], as_index=False)["abs_rel_delta"]
            .mean()
            .sort_values(["model", "abs_rel_delta"], ascending=[True, False])
        )
        summary.to_csv(proc_dir / "e1_transition_strength_by_model_interval.csv", index=False)

    # A compact table for supervisor/thesis notes: stable rank by model/module/checkpoint.
    sr_col = metric_col(df, "stable_rank")
    if sr_col:
        label_col = "module" if "module" in df.columns else "matrix"
        sr = df.groupby(["model", "step_num", label_col], as_index=False)[sr_col].mean()
        sr = sr.rename(columns={label_col: "module", sr_col: "stable_rank"})
        sr.to_csv(tab_dir / "e1_stable_rank_by_model_module_checkpoint.csv", index=False)

    print(f"Wrote early-window E1 artifacts to {out_root}")


if __name__ == "__main__":
    main()
