#!/usr/bin/env python3
"""Step 0 artifact overlay: LR schedule + E1 geometry + Step-1 durability.

Inputs are intentionally schema-tolerant. The E1 dense table may be long-form
(metric,value) or wide-form. The durability table may come from E3/Step-1 raw
cell summaries or post-analysis stage summaries.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_step(x) -> int:
    if pd.isna(x):
        return -1
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x)
    if s.startswith("step"):
        s = s[4:]
    try:
        return int(float(s))
    except Exception:
        return -1


def infer_step_col(df: pd.DataFrame) -> str:
    for c in ["step_num", "step", "stage_step", "checkpoint_step", "inject_step_num", "revision_step"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "step" in c.lower() or "checkpoint" in c.lower() or "stage" in c.lower():
            return c
    raise ValueError(f"Could not infer step column from columns: {list(df.columns)}")


def aggregate_metric(df: pd.DataFrame, metric_keywords: Iterable[str], prefer_directional: bool = True) -> pd.DataFrame:
    step_col = infer_step_col(df)
    work = df.copy()
    work["_step"] = work[step_col].map(parse_step)
    work = work[work["_step"] >= 0]

    metric_keywords = [m.lower() for m in metric_keywords]

    if "metric" in work.columns and "value" in work.columns:
        m = work["metric"].astype(str).str.lower()
        mask = np.logical_or.reduce([m.str.contains(k, regex=False) for k in metric_keywords])
        sub = work[mask].copy()
        if sub.empty:
            raise ValueError(f"No long-form metric matched {metric_keywords}")
        sub["_value"] = pd.to_numeric(sub["value"], errors="coerce")
        return sub.groupby("_step", as_index=False)["_value"].mean().rename(columns={"_step": "step", "_value": "value"})

    candidates = []
    for c in work.columns:
        cl = c.lower()
        if any(k in cl for k in metric_keywords):
            candidates.append(c)
    if not candidates:
        raise ValueError(f"No wide metric column matched {metric_keywords}. Columns: {list(work.columns)}")
    vals = []
    for c in candidates:
        vals.append(pd.to_numeric(work[c], errors="coerce"))
    work["_value"] = pd.concat(vals, axis=1).mean(axis=1)
    return work.groupby("_step", as_index=False)["_value"].mean().rename(columns={"_step": "step", "_value": "value"})


def aggregate_durability(df: pd.DataFrame) -> pd.DataFrame:
    step_col = infer_step_col(df)
    work = df.copy()
    work["_step"] = work[step_col].map(parse_step)
    work = work[work["_step"] >= 0]

    # Prefer uptake-normalised clean retention margin.
    priority = [
        "normalized_retention_margin", "norm_retention_margin", "clean_retention_norm_margin",
        "retention_margin_norm", "retention_norm_margin", "normalized_clean_retention_margin",
        "retention_margin", "clean_retention_margin",
    ]
    col = None
    lower = {c.lower(): c for c in work.columns}
    for p in priority:
        if p in lower:
            col = lower[p]
            break
    if col is None:
        candidates = []
        for c in work.columns:
            cl = c.lower()
            if "retention" in cl and "margin" in cl and "base" not in cl:
                candidates.append(c)
        if not candidates:
            raise ValueError(f"Could not infer durability/retention column. Columns: {list(work.columns)}")
        col = candidates[0]
    work["_value"] = pd.to_numeric(work[col], errors="coerce")
    return work.groupby("_step", as_index=False)["_value"].mean().rename(columns={"_step": "step", "_value": "durability"})


def pythia_lr(steps: np.ndarray, max_lr: float, min_lr: float, warmup_end: int, final_step: int) -> np.ndarray:
    steps = np.asarray(steps, dtype=float)
    lr = np.zeros_like(steps, dtype=float)
    warm = steps <= warmup_end
    lr[warm] = max_lr * np.clip(steps[warm] / max(1, warmup_end), 0, 1)
    post = ~warm
    t = np.clip((steps[post] - warmup_end) / max(1, final_step - warmup_end), 0, 1)
    lr[post] = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * t))
    return lr


def normalise(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if np.all(~np.isfinite(y)):
        return y
    lo, hi = np.nanpercentile(y, [5, 95])
    if not np.isfinite(hi - lo) or abs(hi - lo) < 1e-12:
        lo, hi = np.nanmin(y), np.nanmax(y)
    return (y - lo) / (hi - lo + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-metrics", required=True, help="Dense E1 metrics CSV")
    ap.add_argument("--durability", required=True, help="Step-1/E3 durability summary CSV")
    ap.add_argument("--out", default="results/step0_artifact_overlay")
    ap.add_argument("--warmup-end", type=int, default=1400)
    ap.add_argument("--final-step", type=int, default=143000)
    ap.add_argument("--max-lr", type=float, default=6.0e-4, help="Approx Pythia-160M max LR; override if needed")
    ap.add_argument("--min-lr", type=float, default=6.0e-5, help="Approx min LR; override if needed")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    e1 = pd.read_csv(args.e1_metrics)
    dur = pd.read_csv(args.durability)

    # Directional/stability indicator first, stable rank second.
    try:
        sv = aggregate_metric(e1, ["sv_stability", "subspace", "singular_vector", "stability"])
        sv_name = "SV/subspace stability"
    except Exception as exc:
        print(f"WARN: could not aggregate SV stability ({exc}); falling back to stable rank")
        sv = aggregate_metric(e1, ["stable_rank", "effective_rank", "rank"])
        sv_name = "rank indicator"
    try:
        sr = aggregate_metric(e1, ["stable_rank"])
    except Exception:
        sr = None
    d = aggregate_durability(dur)

    max_step = max(int(max(sv.step.max(), d.step.max())), args.final_step)
    xs = np.unique(np.concatenate([sv.step.values, d.step.values, np.array([0, args.warmup_end, args.final_step])]))
    lr = pythia_lr(xs, args.max_lr, args.min_lr, args.warmup_end, args.final_step)
    lr_df = pd.DataFrame({"step": xs, "lr": lr})
    lr_df.to_csv(out / "tables" / "pythia_lr_schedule_overlay.csv", index=False)

    overlay = pd.merge(sv.rename(columns={"value": "geometry_value"}), d, on="step", how="outer").sort_values("step")
    overlay.to_csv(out / "tables" / "artifact_overlay_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, normalise(lr), label="LR schedule (normalised)")
    ax.plot(sv.step, normalise(sv.value), marker="o", markersize=3, label=f"{sv_name} (normalised)")
    ax.plot(d.step, normalise(d.durability), marker="s", markersize=3, label="durability/retention (normalised)")
    ax.axvline(args.warmup_end, linestyle="--", linewidth=1.5, label=f"warmup end ≈ step{args.warmup_end}")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Pythia checkpoint step")
    ax.set_ylabel("normalised value")
    ax.set_title("Step 0 artifact overlay: LR, geometry, and durability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "artifact_overlay.png", dpi=200)
    fig.savefig(out / "figures" / "artifact_overlay.pdf")
    plt.close(fig)

    report = f"""# Step 0 artifact overlay report\n\n- E1 metrics: `{args.e1_metrics}`\n- Durability file: `{args.durability}`\n- Warmup-end marker: step {args.warmup_end}\n- Geometry trace used: {sv_name}\n\nInterpretation checklist:\n\n1. Does the durability decline occur at/after the geometry re-stabilisation region rather than exactly at warmup end?\n2. Are directional indicators (SV/subspace stability or stable rank) offset from the warmup landmark?\n3. If geometry and durability sit directly on warmup-end, hedge mechanism as schedule-entangled.\n\nOutputs:\n\n- `figures/artifact_overlay.png`\n- `tables/artifact_overlay_values.csv`\n- `tables/pythia_lr_schedule_overlay.csv`\n"""
    (out / "reports" / "artifact_overlay_report.md").write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
