#!/usr/bin/env python3
"""
Standalone Step 0 artifact overlay.

Purpose:
  Build a credibility figure before the dense Step-1 factual sweep is finished.
  It overlays:
    (1) Pythia LR schedule proxy / warmup end,
    (2) dense E1 geometry indicators (stable-rank and/or SV-stability if present),
    (3) optional behavioural durability curve from any existing E3/Step-1 summary.

This script is intentionally schema-tolerant because earlier E1/E3 artifacts in the repo
may use different column names.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_step_value(x) -> Optional[int]:
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    s = str(x).strip()
    if s.startswith("step"):
        s = s[4:]
    s = s.replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return None


def infer_step_column(df: pd.DataFrame) -> str:
    for c in ["step_num", "step", "stage", "checkpoint", "revision", "inject_step", "stage_step"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer step column. Columns: {list(df.columns)}")


def add_step_num(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "step_num" not in df.columns:
        c = infer_step_column(df)
        df["step_num"] = df[c].map(parse_step_value)
    else:
        df["step_num"] = df["step_num"].map(parse_step_value)
    df = df[df["step_num"].notna()].copy()
    df["step_num"] = df["step_num"].astype(int)
    return df


def robust_zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        std = np.nanstd(x)
        if not np.isfinite(std) or std == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - np.nanmean(x)) / std
    return 0.6745 * (x - med) / mad


def minmax01(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)


def pick_value_column(df: pd.DataFrame, preferred: Iterable[str]) -> Optional[str]:
    for c in preferred:
        if c in df.columns:
            return c
    return None


def load_geometry(e1_path: Path) -> pd.DataFrame:
    df = pd.read_csv(e1_path)
    df = add_step_num(df)

    # Remove step0 from main overlay by default; it is often diagnostic only.
    df = df[df["step_num"] > 0].copy()

    # Two common schemas:
    # 1) long: columns include metric,value
    # 2) wide: columns include stable_rank, sv_stability, effective_rank, ...
    if "metric" in df.columns and "value" in df.columns:
        long = df.copy()
        long["metric"] = long["metric"].astype(str)
        pivot = (
            long.groupby(["step_num", "metric"], as_index=False)["value"]
            .mean()
            .pivot(index="step_num", columns="metric", values="value")
            .reset_index()
        )
        g = pivot
    else:
        numeric_cols = [c for c in df.columns if c != "step_num" and pd.api.types.is_numeric_dtype(df[c])]
        g = df.groupby("step_num", as_index=False)[numeric_cols].mean()

    # Find geometry series flexibly.
    stable_candidates = [
        "stable_rank", "stable-rank", "stable_rank_mean", "rank_stable", "srank",
        "effective_rank", "effective_rank_mean",
    ]
    sv_candidates = [
        "sv_stability", "subspace_stability", "topk_sv_stability", "singular_vector_stability",
        "sv_cosine", "cosine", "stability",
    ]

    stable_col = pick_value_column(g, stable_candidates)
    sv_col = pick_value_column(g, sv_candidates)

    # More permissive fuzzy search.
    if stable_col is None:
        for c in g.columns:
            lc = str(c).lower()
            if "stable" in lc and "rank" in lc:
                stable_col = c
                break
    if sv_col is None:
        for c in g.columns:
            lc = str(c).lower()
            if ("stability" in lc or "cos" in lc) and ("sv" in lc or "subspace" in lc or "singular" in lc):
                sv_col = c
                break

    out = g[["step_num"]].copy()
    if stable_col is not None:
        out["stable_rank_like"] = pd.to_numeric(g[stable_col], errors="coerce")
    if sv_col is not None:
        out["sv_stability_like"] = pd.to_numeric(g[sv_col], errors="coerce")

    # If no recognized columns, use the first numeric column as a fallback, clearly named.
    if len(out.columns) == 1:
        numeric_cols = [c for c in g.columns if c != "step_num" and pd.api.types.is_numeric_dtype(g[c])]
        if not numeric_cols:
            raise ValueError(f"No numeric geometry columns found in {e1_path}. Columns: {list(g.columns)}")
        out["geometry_fallback"] = pd.to_numeric(g[numeric_cols[0]], errors="coerce")

    return out.sort_values("step_num")


def load_durability(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = add_step_num(df)
    df = df[df["step_num"] > 0].copy()

    candidates = [
        "normalized_retention_margin", "norm_retention_margin", "retention_margin_normalized",
        "clean_retention_margin_norm", "retention_margin", "clean_retention_margin",
        "retention", "normalized_retention", "retention_rate",
    ]
    ycol = pick_value_column(df, candidates)
    if ycol is None:
        # Fuzzy fallback.
        for c in df.columns:
            lc = str(c).lower()
            if "retention" in lc and ("margin" in lc or "norm" in lc or "rate" in lc):
                ycol = c
                break
    if ycol is None:
        numeric_cols = [c for c in df.columns if c != "step_num" and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError(f"Could not infer durability column. Columns: {list(df.columns)}")
        ycol = numeric_cols[0]

    out = df.groupby("step_num", as_index=False)[ycol].mean().rename(columns={ycol: "durability"})
    return out.sort_values("step_num")


def pythia_lr_schedule(steps: np.ndarray, max_steps: int, warmup_end: int, max_lr: float, min_lr: float) -> np.ndarray:
    steps = np.asarray(steps, dtype=float)
    lr = np.empty_like(steps, dtype=float)
    warm = steps <= warmup_end
    lr[warm] = max_lr * np.maximum(steps[warm], 0) / max(warmup_end, 1)
    t = np.clip((steps[~warm] - warmup_end) / max(max_steps - warmup_end, 1), 0, 1)
    lr[~warm] = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * t))
    return lr


def make_overlay(geometry: pd.DataFrame, durability: Optional[pd.DataFrame], out_dir: Path, warmup_end: int, max_step: int, max_lr: float, min_lr: float, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    rep_dir = out_dir / "reports"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)
    rep_dir.mkdir(exist_ok=True)

    # Merge for output table but plot series independently.
    overlay = geometry.copy()
    if durability is not None:
        overlay = overlay.merge(durability, on="step_num", how="left")
    overlay.to_csv(tab_dir / "step0_artifact_overlay_table.csv", index=False)

    x_all = np.array(sorted(set(geometry["step_num"].tolist() + ([] if durability is None else durability["step_num"].tolist()))), dtype=int)
    if len(x_all) == 0:
        raise ValueError("No steps to plot")
    dense_x = np.linspace(max(1, int(x_all.min())), max(max_step, int(x_all.max())), 500)
    lr = pythia_lr_schedule(dense_x, max(max_step, int(x_all.max())), warmup_end, max_lr, min_lr)
    lr01 = (lr - lr.min()) / (lr.max() - lr.min() if lr.max() > lr.min() else 1)

    plt.figure(figsize=(11.5, 6.5))
    plt.plot(dense_x, lr01, linestyle="--", linewidth=2, label="LR schedule (normalised)")
    plt.axvline(warmup_end, linestyle=":", linewidth=2, label=f"warmup end ≈ step {warmup_end}")

    for col in geometry.columns:
        if col == "step_num":
            continue
        y = minmax01(geometry[col])
        plt.plot(geometry["step_num"], y, marker="o", linewidth=1.8, label=f"geometry: {col} (0–1)")

    if durability is not None:
        plt.plot(durability["step_num"], minmax01(durability["durability"]), marker="s", linewidth=2.4, label="durability/retention (0–1)")

    plt.xscale("log")
    plt.xlabel("Pre-training checkpoint step (log scale; step0 excluded)")
    plt.ylabel("Normalised value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "artifact_overlay_logx.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11.5, 6.5))
    plt.plot(dense_x, lr01, linestyle="--", linewidth=2, label="LR schedule (normalised)")
    plt.axvline(warmup_end, linestyle=":", linewidth=2, label=f"warmup end ≈ step {warmup_end}")
    for col in geometry.columns:
        if col == "step_num":
            continue
        plt.plot(geometry["step_num"], minmax01(geometry[col]), marker="o", linewidth=1.8, label=f"geometry: {col} (0–1)")
    if durability is not None:
        plt.plot(durability["step_num"], minmax01(durability["durability"]), marker="s", linewidth=2.4, label="durability/retention (0–1)")
    plt.xlabel("Pre-training checkpoint step (linear scale; step0 excluded)")
    plt.ylabel("Normalised value")
    plt.title(title + " — linear axis check")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "artifact_overlay_linear.png", dpi=220)
    plt.close()

    report = []
    report.append("# Step 0 artifact overlay report\n")
    report.append(f"- Warmup-end marker: step {warmup_end}\n")
    report.append(f"- Step0 excluded from main overlay: yes\n")
    report.append(f"- Geometry series plotted: {', '.join([c for c in geometry.columns if c != 'step_num'])}\n")
    report.append(f"- Behavioural durability included: {'yes' if durability is not None else 'no'}\n")
    report.append("\n## Interpretation rule\n")
    report.append("This figure is a credibility/artifact audit. The key thesis-safe reading is whether the durability decline and SV/stable-rank consolidation occur offset from the warmup-end landmark rather than sitting exactly on top of it. Norm-like indicators should be treated as corroborative only; directional/stability indicators carry more mechanistic weight.\n")
    (rep_dir / "step0_artifact_overlay_report.md").write_text("".join(report))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-metrics", required=True, type=Path)
    ap.add_argument("--durability", type=Path, default=None, help="Optional existing E3/Step-1 stage summary CSV. Can be omitted for geometry+LR provisional overlay.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--warmup-end", type=int, default=1400)
    ap.add_argument("--max-step", type=int, default=143000)
    ap.add_argument("--max-lr", type=float, default=6.0e-4)
    ap.add_argument("--min-lr", type=float, default=6.0e-5)
    ap.add_argument("--title", default="Step 0 artifact overlay: LR schedule, geometry, and durability")
    args = ap.parse_args()

    geometry = load_geometry(args.e1_metrics)
    durability = load_durability(args.durability) if args.durability else None
    make_overlay(geometry, durability, args.out, args.warmup_end, args.max_step, args.max_lr, args.min_lr, args.title)
    print(f"Wrote Step 0 overlay to {args.out}")


if __name__ == "__main__":
    main()
