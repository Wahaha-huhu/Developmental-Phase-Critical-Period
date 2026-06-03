from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_step(
    df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    log_x: bool = False,
    title: str | None = None,
    group_cols: Iterable[str] = ("module",),
) -> Path:
    """Plot model-averaged metric by step, grouped by module by default."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    group_cols = list(group_cols)
    agg_cols = ["step", *group_cols]
    plot_df = df.groupby(agg_cols, as_index=False)[metric].mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, sub in plot_df.groupby(group_cols):
        label = key if isinstance(key, str) else "/".join(map(str, key))
        sub = sub.sort_values("step")
        ax.plot(sub["step"], sub[metric], marker="o", label=label)
    if log_x:
        ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"E1 {metric} across checkpoints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_summary_table(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a compact checkpoint/module summary table for thesis inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        "stable_rank",
        "effective_rank",
        "spectral_norm",
        "frobenius_norm",
        "subspace_stability_topk",
        "mp_outliers_x1",
        "alpha_tail_frac_0.3",
    ]
    keep = [m for m in metrics if m in df.columns]
    table = (
        df.groupby(["model", "step", "module"], as_index=False)[keep]
        .mean(numeric_only=True)
        .sort_values(["model", "step", "module"])
    )
    table.to_csv(output_path, index=False)
    return output_path
