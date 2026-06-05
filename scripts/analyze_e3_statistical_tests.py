#!/usr/bin/env python3
"""E3 statistical strengthening analysis.

Adds the post-hoc-but-predefined statistical analyses needed to distinguish a
stage/window effect from a smooth monotone early-to-late trend and from uptake
mismatch.

Inputs: one or more E3 result roots produced by the v3 runner. Each root should
contain tables/e3_v3_cell_summary.csv and optionally raw/e3_factual_degradation_curve.csv.

Outputs: tables, figures, and a markdown report under --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-8


def safe_name(s: str) -> str:
    return str(s).replace("/", "__").replace(" ", "_").replace(":", "_")


def read_cell_summary(root: Path) -> pd.DataFrame:
    candidates = [
        root / "tables" / "e3_v3_cell_summary.csv",
        root / "raw" / "e3_factual_cell_summary.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["source_root"] = str(root)
            return df
    raise FileNotFoundError(f"Could not find E3 cell summary under {root}")


def read_degradation(root: Path) -> Optional[pd.DataFrame]:
    p = root / "raw" / "e3_factual_degradation_curve.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["source_root"] = str(root)
        return df
    return None


def add_stage_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "step" not in df.columns:
        if "stage" in df.columns:
            df["step"] = df["stage"].astype(str).str.replace("step", "", regex=False).astype(int)
        else:
            raise ValueError("Need step or stage column")
    df["step"] = df["step"].astype(int)
    df["log_step"] = np.log10(df["step"].clip(lower=1) + 1)
    # Main E1/E2 window supported by current results.
    df["is_window"] = df["step"].between(512, 3000).astype(int)
    df["is_late"] = df["step"].isin([8000, 143000]).astype(int)
    df["is_diagnostic_early"] = df["step"].isin([0, 128]).astype(int)
    df["stage_order"] = df["step"].rank(method="dense").astype(int)
    return df


def normalise_degradation_curve(cell: pd.DataFrame, deg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute normalized poison retention and cell-level AUC from raw degradation curve."""
    keys = ["source_root", "model", "stage", "step", "seed"]
    needed = keys + ["base_probe_mean_margin", "post_injection_mean_margin", "base_probe_accuracy", "post_injection_accuracy"]
    cell_small = cell[needed].copy()
    merged = deg.merge(cell_small, on=keys, how="left")
    denom_m = merged["post_injection_mean_margin"] - merged["base_probe_mean_margin"]
    denom_a = merged["post_injection_accuracy"] - merged["base_probe_accuracy"]
    merged["normalized_poison_retention_margin"] = (merged["probe_mean_margin"] - merged["base_probe_mean_margin"]) / denom_m.replace(0, np.nan)
    merged["normalized_poison_retention_accuracy"] = (merged["probe_accuracy"] - merged["base_probe_accuracy"]) / denom_a.replace(0, np.nan)
    merged["log_poison_budget"] = np.log1p(merged["poison_budget"].astype(float))

    rows = []
    for gkey, g in merged.groupby(keys, dropna=False):
        g = g.sort_values("poison_budget")
        x = g["log_poison_budget"].to_numpy(float)
        xr = x.max() - x.min()
        if xr < EPS:
            auc_m = np.nan
            auc_a = np.nan
        else:
            auc_m = np.trapezoid(g["normalized_poison_retention_margin"].to_numpy(float), x) / xr
            auc_a = np.trapezoid(g["normalized_poison_retention_accuracy"].to_numpy(float), x) / xr
        row = dict(zip(keys, gkey if isinstance(gkey, tuple) else (gkey,)))
        row.update({
            "normalized_degradation_auc_margin": auc_m,
            "normalized_degradation_auc_accuracy": auc_a,
            "final_normalized_poison_retention_margin": g["normalized_poison_retention_margin"].iloc[-1],
            "final_normalized_poison_retention_accuracy": g["normalized_poison_retention_accuracy"].iloc[-1],
        })
        rows.append(row)
    return merged, pd.DataFrame(rows)


def ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    n, p = X.shape
    if n <= p:
        return {"n": n, "p": p, "rss": np.nan, "aic": np.nan, "bic": np.nan, "rmse": np.nan, "loocv_rmse": np.nan}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    sigma2 = rss / n if n else np.nan
    aic = n * math.log(max(sigma2, EPS)) + 2 * p
    bic = n * math.log(max(sigma2, EPS)) + p * math.log(n)
    rmse = float(np.sqrt(np.mean(resid**2)))
    # LOOCV via hat diagonal; stable enough for small OLS.
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        h = np.sum((X @ XtX_inv) * X, axis=1)
        loo_resid = resid / np.clip(1 - h, 1e-6, None)
        loocv_rmse = float(np.sqrt(np.mean(loo_resid**2)))
    except Exception:
        loocv_rmse = np.nan
    return {"n": n, "p": p, "rss": rss, "aic": aic, "bic": bic, "rmse": rmse, "loocv_rmse": loocv_rmse}


def design_matrix(df: pd.DataFrame, model_kind: str, break_step: Optional[int] = None, include_uptake: bool = False) -> np.ndarray:
    cols = [np.ones(len(df))]
    log_step = df["log_step"].to_numpy(float)
    if model_kind == "monotone_log_step":
        cols.append(log_step)
    elif model_kind == "quadratic_log_step":
        cols.append(log_step)
        cols.append(log_step**2)
    elif model_kind == "window_indicator":
        cols.append(df["is_window"].to_numpy(float))
        cols.append(df["is_late"].to_numpy(float))
    elif model_kind == "fixed_break_segmented":
        if break_step is None:
            raise ValueError("break_step needed")
        b = math.log10(max(break_step, 1) + 1)
        cols.append(np.minimum(log_step, b))
        cols.append(np.maximum(0, log_step - b))
    else:
        raise ValueError(model_kind)
    if include_uptake:
        u = df["uptake_margin_delta"].to_numpy(float)
        u = (u - np.nanmean(u)) / (np.nanstd(u) + EPS)
        cols.append(u)
    return np.vstack(cols).T


def compare_models(df: pd.DataFrame, outcome: str, include_uptake_options=(False, True)) -> pd.DataFrame:
    rows = []
    # Exclude step0/128 by default: they are diagnostic and often weak uptake.
    d = df[df["step"].isin([512, 1000, 2000, 3000, 8000, 143000])].copy()
    y = d[outcome].to_numpy(float)
    for include_uptake in include_uptake_options:
        specs = [
            ("monotone_log_step", None),
            ("quadratic_log_step", None),
            ("window_indicator", None),
            ("fixed_break_segmented", 1000),
            ("fixed_break_segmented", 2000),
            ("fixed_break_segmented", 3000),
        ]
        # Free-break segmented over candidate break locations.
        for kind, b in specs:
            X = design_matrix(d, kind, b, include_uptake=include_uptake)
            fit = ols_fit(X, y)
            rows.append({
                "outcome": outcome,
                "model_kind": kind,
                "break_step": b if b is not None else "",
                "include_uptake": include_uptake,
                **fit,
            })
        for b in [512, 1000, 2000, 3000, 8000]:
            X = design_matrix(d, "fixed_break_segmented", b, include_uptake=include_uptake)
            fit = ols_fit(X, y)
            rows.append({
                "outcome": outcome,
                "model_kind": "free_break_candidate_segmented",
                "break_step": b,
                "include_uptake": include_uptake,
                **fit,
            })
    res = pd.DataFrame(rows)
    # Rank within outcome/uptake group.
    if not res.empty:
        res["delta_aic"] = res.groupby(["outcome", "include_uptake"])["aic"].transform(lambda x: x - np.nanmin(x))
        res["delta_bic"] = res.groupby(["outcome", "include_uptake"])["bic"].transform(lambda x: x - np.nanmin(x))
    return res.sort_values(["outcome", "include_uptake", "aic"])


def uptake_control_regression(df: pd.DataFrame, outcome: str) -> Dict[str, float]:
    d = df[df["step"].isin([512, 1000, 2000, 3000, 8000, 143000])].copy()
    y = d[outcome].to_numpy(float)
    # y ~ intercept + window + late + standardized uptake + log step
    u = d["uptake_margin_delta"].to_numpy(float)
    u = (u - np.nanmean(u)) / (np.nanstd(u) + EPS)
    X = np.vstack([
        np.ones(len(d)),
        d["is_window"].to_numpy(float),
        d["is_late"].to_numpy(float),
        u,
        d["log_step"].to_numpy(float),
    ]).T
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    names = ["intercept", "window_coef", "late_coef", "uptake_z_coef", "log_step_coef"]
    if len(y) <= X.shape[1]:
        return {k: np.nan for k in names} | {"outcome": outcome, "n": len(y), "r2": np.nan}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > EPS else np.nan
    out = {name: float(val) for name, val in zip(names, beta)}
    out.update({"outcome": outcome, "n": int(len(y)), "r2": float(r2)})
    return out


def bootstrap_window_vs_late(df: pd.DataFrame, outcome: str, n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    # Pair by seed: mean window stages vs mean late stages within the same seed.
    rows = []
    for (root, model, seed_val), g in df.groupby(["source_root", "model", "seed"], dropna=False):
        w = g[g["is_window"] == 1][outcome].dropna()
        l = g[g["is_late"] == 1][outcome].dropna()
        if len(w) and len(l):
            rows.append({"source_root": root, "model": model, "seed": seed_val, "window_mean": w.mean(), "late_mean": l.mean(), "diff": w.mean() - l.mean()})
    paired = pd.DataFrame(rows)
    if paired.empty:
        return {"outcome": outcome, "n_pairs": 0, "mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_diff_le_0": np.nan}
    vals = paired["diff"].to_numpy(float)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(np.mean(sample))
    boots = np.array(boots)
    return {
        "outcome": outcome,
        "n_pairs": int(len(vals)),
        "mean_window": float(paired["window_mean"].mean()),
        "mean_late": float(paired["late_mean"].mean()),
        "mean_diff": float(np.mean(vals)),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "p_diff_le_0": float(np.mean(boots <= 0)),
    }


def make_plots(df: pd.DataFrame, outcomes: List[str], out_fig: Path):
    out_fig.mkdir(parents=True, exist_ok=True)
    stages = sorted(df["step"].unique())
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        agg = df.groupby("step")[outcome].agg(["mean", "sem", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.errorbar(agg["step"], agg["mean"], yerr=agg["sem"].fillna(0), marker="o", capsize=3)
        ax.axvspan(512, 3000, alpha=0.12, label="E1/E2 window")
        ax.set_xscale("symlog", linthresh=128)
        ax.set_xlabel("Injection stage (Pythia step)")
        ax.set_ylabel(outcome.replace("_", " "))
        ax.set_title(outcome.replace("_", " "))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_fig / f"e3_stat_{safe_name(outcome)}_by_stage.png", dpi=200)
        plt.close(fig)


def write_report(out: Path, cell: pd.DataFrame, model_cmp: pd.DataFrame, boot: pd.DataFrame, reg: pd.DataFrame):
    report = []
    report.append("# E3 statistical strengthening report\n")
    report.append(f"- Cells: {len(cell)}")
    report.append(f"- Models: {', '.join(sorted(map(str, cell['model'].unique())))}")
    report.append(f"- Stages: {', '.join(map(str, sorted(cell['step'].unique())))}")
    report.append(f"- Seeds: {', '.join(map(str, sorted(cell['seed'].unique())))}\n")
    if "uptake_margin_delta" in cell:
        pos = int((cell["uptake_margin_delta"] > 0).sum())
        report.append(f"- Positive uptake by margin: {pos}/{len(cell)}")
        report.append(f"- Mean uptake margin delta: {cell['uptake_margin_delta'].mean():.4f}\n")
    report.append("## Window-vs-late bootstrap\n")
    for _, r in boot.iterrows():
        report.append(f"- `{r['outcome']}`: window mean={r.get('mean_window', np.nan):.4f}, late mean={r.get('mean_late', np.nan):.4f}, diff={r.get('mean_diff', np.nan):.4f}, 95% CI=[{r.get('ci_low', np.nan):.4f}, {r.get('ci_high', np.nan):.4f}], bootstrap p(diff<=0)={r.get('p_diff_le_0', np.nan):.4f}")
    report.append("\n## Best model comparisons by AIC\n")
    for (outcome, uptake), g in model_cmp.groupby(["outcome", "include_uptake"]):
        best = g.sort_values("aic").head(3)
        report.append(f"\n### {outcome}, include_uptake={uptake}")
        for _, r in best.iterrows():
            report.append(f"- {r['model_kind']} break={r['break_step']} AIC={r['aic']:.2f} BIC={r['bic']:.2f} LOOCV RMSE={r['loocv_rmse']:.4f} ΔAIC={r['delta_aic']:.2f}")
    report.append("\n## Uptake-controlled regression\n")
    for _, r in reg.iterrows():
        report.append(f"- `{r['outcome']}`: window_coef={r.get('window_coef', np.nan):.4f}, late_coef={r.get('late_coef', np.nan):.4f}, uptake_z_coef={r.get('uptake_z_coef', np.nan):.4f}, log_step_coef={r.get('log_step_coef', np.nan):.4f}, R²={r.get('r2', np.nan):.3f}")
    report.append("\n## Interpretation guide\n")
    report.append("- A positive window-vs-late bootstrap difference supports higher durability inside the E1/E2 window than at late stages.")
    report.append("- If a segmented or window model beats monotone log-step models, the evidence is stronger than a smooth 'earlier is better' account.")
    report.append("- Uptake-controlled regressions test whether the window effect remains after accounting for stage differences in post-injection uptake.")
    report.append("- Step0 and step128 are retained as diagnostics but excluded from model-comparison tests by default, because uptake at these stages is often weak or qualitatively different from trained checkpoints.")
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "e3_statistical_strengthening_report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="E3 result roots to analyse")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "processed").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cells = []
    degs = []
    for r in args.roots:
        root = Path(r)
        c = read_cell_summary(root)
        cells.append(c)
        d = read_degradation(root)
        if d is not None:
            degs.append(d)
    cell = pd.concat(cells, ignore_index=True)
    cell = add_stage_features(cell)

    # Compute normalized degradation AUC from raw curves when available.
    if degs:
        deg = pd.concat(degs, ignore_index=True)
        deg = add_stage_features(deg)
        norm_deg, auc = normalise_degradation_curve(cell, deg)
        norm_deg.to_csv(out / "processed" / "e3_normalized_degradation_curve.csv", index=False)
        auc.to_csv(out / "processed" / "e3_normalized_degradation_auc_by_cell.csv", index=False)
        join_keys = ["source_root", "model", "stage", "step", "seed"]
        cell = cell.merge(auc, on=join_keys, how="left")
    else:
        norm_deg = pd.DataFrame()

    # Save combined cell file.
    cell.to_csv(out / "processed" / "e3_combined_cell_summary_with_features.csv", index=False)

    outcomes = [
        "normalized_retention_margin",
        "normalized_retention_accuracy",
        "normalized_degradation_auc_margin",
        "normalized_degradation_auc_accuracy",
    ]
    outcomes = [o for o in outcomes if o in cell.columns]

    stage_summary = cell.groupby("step").agg(
        n=("seed", "count"),
        uptake_margin_delta_mean=("uptake_margin_delta", "mean"),
        uptake_margin_delta_sem=("uptake_margin_delta", "sem"),
        normalized_retention_margin_mean=("normalized_retention_margin", "mean"),
        normalized_retention_margin_sem=("normalized_retention_margin", "sem"),
        normalized_degradation_auc_margin_mean=("normalized_degradation_auc_margin", "mean") if "normalized_degradation_auc_margin" in cell else ("uptake_margin_delta", "mean"),
        normalized_degradation_auc_margin_sem=("normalized_degradation_auc_margin", "sem") if "normalized_degradation_auc_margin" in cell else ("uptake_margin_delta", "sem"),
    ).reset_index()
    stage_summary.to_csv(out / "tables" / "e3_stat_stage_summary.csv", index=False)

    boot_rows = [bootstrap_window_vs_late(cell, o, n_boot=args.n_bootstrap, seed=args.seed) for o in outcomes]
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out / "tables" / "e3_window_vs_late_bootstrap.csv", index=False)

    cmp_frames = [compare_models(cell, o) for o in outcomes]
    model_cmp = pd.concat(cmp_frames, ignore_index=True) if cmp_frames else pd.DataFrame()
    model_cmp.to_csv(out / "tables" / "e3_segmented_vs_monotone_model_comparison.csv", index=False)

    reg_rows = [uptake_control_regression(cell, o) for o in outcomes]
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(out / "tables" / "e3_uptake_controlled_regression.csv", index=False)

    make_plots(cell, ["uptake_margin_delta"] + outcomes, out / "figures")
    if not norm_deg.empty:
        # Degradation curves by stage.
        agg = norm_deg.groupby(["step", "poison_budget"])["normalized_poison_retention_margin"].agg(["mean", "sem"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for step, g in agg.groupby("step"):
            g = g.sort_values("poison_budget")
            ax.errorbar(g["poison_budget"], g["mean"], yerr=g["sem"].fillna(0), marker="o", capsize=2, label=f"step{step}")
        ax.set_xscale("symlog", linthresh=4)
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Poison budget k (contradicted facts)")
        ax.set_ylabel("Normalized poison retention (margin)")
        ax.set_title("E3 normalized degradation curves")
        ax.legend(ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "figures" / "e3_stat_normalized_degradation_curves.png", dpi=200)
        plt.close(fig)

    write_report(out, cell, model_cmp, boot, reg)

    # Thesis snippet.
    ts = []
    ts.append("% E3 statistical strengthening snippet")
    ts.append("\\paragraph{Statistical tests of the window interpretation.}")
    ts.append("To distinguish a stage-localised sensitive-window effect from a smooth early-to-late trend, I reanalysed E3 with three post hoc robustness tests: paired bootstrap comparisons between the E1/E2 window (steps 512--3000) and late checkpoints (steps 8000 and 143000), segmented-versus-monotone regression models, and uptake-controlled regressions. Step 0 and step 128 are retained as diagnostic baselines but excluded from the main model-comparison tests because uptake at these near-initial checkpoints is qualitatively different from trained checkpoints.")
    if len(boot):
        for _, r in boot.iterrows():
            if r["outcome"] == "normalized_retention_margin":
                ts.append(f"For normalised clean-retention margin, the window--late difference was {r['mean_diff']:.3f} with a bootstrap 95\\% interval [{r['ci_low']:.3f}, {r['ci_high']:.3f}].")
            if r["outcome"] == "normalized_degradation_auc_margin":
                ts.append(f"For normalised degradation-resistance AUC, the window--late difference was {r['mean_diff']:.3f} with a bootstrap 95\\% interval [{r['ci_low']:.3f}, {r['ci_high']:.3f}].")
    ts.append("These analyses use uptake-normalised outcomes, so the reported durability effect is not simply a comparison of raw post-injection scores. The central test is whether injection stage predicts the fraction of the injected signal that survives after matched continuation or adversarial overwrite.")
    (out / "thesis_snippets").mkdir(parents=True, exist_ok=True)
    (out / "thesis_snippets" / "e3_statistical_strengthening_snippet.tex").write_text("\n".join(ts), encoding="utf-8")

    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
