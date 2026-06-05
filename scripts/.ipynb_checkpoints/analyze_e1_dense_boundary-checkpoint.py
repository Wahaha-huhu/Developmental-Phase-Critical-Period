#!/usr/bin/env python3
"""Dense E1 boundary analysis.

This is intentionally lightweight and schema-tolerant. It reads an E1 spectral metrics CSV,
aggregates each metric by model/module/checkpoint, computes adjacent-checkpoint change strength,
and reports which intervals receive the largest votes across metrics and modules.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_STEP_RE = re.compile(r"step(\d+)")
ID_COLS = {"model", "checkpoint", "revision", "step", "layer", "module", "matrix_name", "name", "path"}




def _safe_reset_index(df):
    """Reset index without failing when an index level name already exists as a column."""
    import pandas as _pd
    if not hasattr(df, "index"):
        return df
    names = [n for n in getattr(df.index, "names", []) if n is not None]
    collisions = [n for n in names if n in getattr(df, "columns", [])]
    if collisions:
        return df.reset_index(drop=True)
    return df.reset_index(drop=True)

def step_number(x) -> int:
    if isinstance(x, str):
        m = _STEP_RE.search(x)
        if m:
            return int(m.group(1))
    return int(x)


def infer_cols(df: pd.DataFrame) -> tuple[str, str, str | None]:
    if "model" not in df.columns:
        raise ValueError("Expected a 'model' column")
    step_col = "step" if "step" in df.columns else "checkpoint"
    if step_col not in df.columns:
        raise ValueError("Expected a 'step' or 'checkpoint' column")
    module_col = "module" if "module" in df.columns else None
    return "model", step_col, module_col


def numeric_metric_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().sum() > 0:
                cols.append(c)
    # Prefer the main spectral metrics if present, but keep unknown numeric metric columns too.
    preferred_order = [
        "stable_rank", "effective_rank", "subspace_stability_topk", "mp_outliers_x1",
        "alpha_tail_frac_0.3", "spectral_norm", "frobenius_norm", "nuclear_norm",
    ]
    ordered = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
    return ordered


def compute_interval_table(df: pd.DataFrame) -> pd.DataFrame:
    model_col, step_col, module_col = infer_cols(df)
    df = df.copy()
    df["step_num"] = df[step_col].map(step_number)
    group_cols = [model_col, "step_num"]
    if module_col:
        group_cols.append(module_col)
    metrics = numeric_metric_cols(df)
    if not metrics:
        raise ValueError("No numeric metric columns found")
    agg = _safe_reset_index(df.groupby(group_cols, dropna=False)[metrics].mean())
    rows = []
    by_cols = [model_col]
    if module_col:
        by_cols.append(module_col)
    for keys, g in agg.groupby(by_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(by_cols, keys))
        g = g.sort_values("step_num")
        steps = g["step_num"].to_numpy()
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            if np.isfinite(vals).sum() < 3:
                continue
            scale = np.nanmedian(np.abs(vals))
            if not np.isfinite(scale) or scale == 0:
                scale = np.nanstd(vals)
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            for i in range(len(steps)-1):
                a, b = int(steps[i]), int(steps[i+1])
                delta = vals[i+1] - vals[i]
                rows.append({
                    **key_map,
                    "metric": metric,
                    "step_a": a,
                    "step_b": b,
                    "interval": f"{a}->{b}",
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "norm_abs_delta": abs(delta) / scale,
                })
    return pd.DataFrame(rows)


def boundary_votes(intervals: pd.DataFrame) -> pd.DataFrame:
    # Each model/module/metric votes for its strongest adjacent interval.
    group_cols = ["model", "metric"]
    if "module" in intervals.columns:
        group_cols.insert(1, "module")
    idx = intervals.groupby(group_cols)["norm_abs_delta"].idxmax()
    top = intervals.loc[idx].copy()
    votes = (
        top.groupby(["model", "interval", "step_a", "step_b"], dropna=False)
        .agg(votes=("metric", "size"), mean_strength=("norm_abs_delta", "mean"), max_strength=("norm_abs_delta", "max"))
        .reset_index(drop=True)
        .sort_values(["model", "votes", "mean_strength"], ascending=[True, False, False])
    )
    return votes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to e1_spectral_metrics.csv")
    ap.add_argument("--out", required=True, help="Output root")
    args = ap.parse_args()
    out = Path(args.out)
    (out/"tables").mkdir(parents=True, exist_ok=True)
    (out/"reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics)
    # Guard against duplicate columns from previous processing.
    if hasattr(df, 'columns'):
        df = df.loc[:, ~df.columns.duplicated()].copy()

    intervals = compute_interval_table(df)
    votes = boundary_votes(intervals)
    intervals.to_csv(out/"tables"/"e1_dense_interval_changes.csv", index=False)
    votes.to_csv(out/"tables"/"e1_dense_boundary_votes.csv", index=False)

    lines = ["# E1 dense boundary validation report", "", f"Input: `{args.metrics}`", f"Rows: {len(df)}", ""]
    for model, g in votes.groupby("model"):
        top = g.iloc[0]
        lines.append(f"## {model}")
        lines.append(f"Top interval: **{top['interval']}** ({int(top['votes'])} votes, mean strength {top['mean_strength']:.3f})")
        lines.append("")
        lines.append(g.head(8).to_markdown(index=False))
        lines.append("")
    (out/"reports"/"e1_dense_boundary_report.md").write_text("\n".join(lines))
    print(out/"reports"/"e1_dense_boundary_report.md")


if __name__ == "__main__":
    main()
