#!/usr/bin/env python3
"""Plot dense E1 spectral/rank indicator panels for one model.

Works with long CSVs of the form (step/checkpoint, layer, module, metric, value)
and with wide CSVs where metric columns are numeric. Designed for thesis-clean
outputs under solid_results/.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

STEP_RE = re.compile(r"step(\d+)")
LAYER_RE = re.compile(r"(?:layers?|h)\.?([0-9]+)")

INDICATOR_PREFERRED = [
    "stable_rank",
    "effective_rank",
    "spectral_norm",
    "frobenius_norm",
    "sv_stability",
    "subspace_stability",
    "mp_outliers",
    "alpha",
]

ID_COLS_CANDIDATES = {
    "model", "checkpoint", "stage", "step", "step_name", "step_num", "revision",
    "layer", "layer_idx", "layer_id", "module", "module_name", "matrix", "name",
    "module_suffix", "module_family", "module_role", "metric", "value",
}


def parse_step_num(x: object) -> int | None:
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and np.isfinite(x):
        return int(x)
    s = str(x)
    m = STEP_RE.search(s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except Exception:
        return None


def infer_layer(row: pd.Series) -> int | None:
    for c in ["layer", "layer_idx", "layer_id"]:
        if c in row and pd.notna(row[c]):
            try:
                return int(row[c])
            except Exception:
                pass
    for c in ["module", "module_name", "matrix", "name"]:
        if c in row and pd.notna(row[c]):
            m = LAYER_RE.search(str(row[c]))
            if m:
                return int(m.group(1))
    return None


def module_text(row: pd.Series) -> str:
    for c in ["module", "module_name", "module_suffix", "matrix", "name"]:
        if c in row and pd.notna(row[c]):
            return str(row[c])
    return "unknown"


def module_family_from_text(s: str) -> str:
    low = s.lower()
    if any(k in low for k in ["attn", "attention", "query_key_value", "qkv", "q_proj", "k_proj", "v_proj"]):
        return "attention"
    if any(k in low for k in ["mlp", "dense_h_to_4h", "dense_4h_to_h", "ffn", "feed_forward"]):
        return "mlp"
    return "other"


def module_role_from_text(s: str) -> str:
    low = s.lower()
    if "query_key_value" in low or "qkv" in low:
        return "attention_qkv"
    if "attention.dense" in low or "attn.out" in low or "o_proj" in low:
        return "attention_out"
    if "dense_h_to_4h" in low or "fc_in" in low or "up_proj" in low or "gate_proj" in low:
        return "mlp_in"
    if "dense_4h_to_h" in low or "fc_out" in low or "down_proj" in low:
        return "mlp_out"
    fam = module_family_from_text(s)
    return fam if fam != "other" else "other"


def normalize_indicator_name(s: str) -> str:
    x = str(s).strip()
    x = x.replace(" ", "_").replace("-", "_").lower()
    aliases = {
        "erank": "effective_rank",
        "effective_rank_value": "effective_rank",
        "stable_rank_value": "stable_rank",
        "spec_norm": "spectral_norm",
        "spectral": "spectral_norm",
        "fro_norm": "frobenius_norm",
        "frob_norm": "frobenius_norm",
        "frobenius": "frobenius_norm",
        "sv_stab": "sv_stability",
        "singular_vector_stability": "sv_stability",
        "subspace": "subspace_stability",
        "mp_outlier_count": "mp_outliers",
        "outliers": "mp_outliers",
    }
    return aliases.get(x, x)


def read_and_longify(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")

    # Determine step number.
    if "step_num" not in df.columns:
        source = None
        for c in ["checkpoint", "stage", "step", "step_name", "revision"]:
            if c in df.columns:
                source = c
                break
        if source is None:
            raise ValueError("Could not infer step column. Expected one of checkpoint/stage/step/revision/step_num.")
        df["step_num"] = df[source].map(parse_step_num)
    else:
        df["step_num"] = df["step_num"].map(parse_step_num)
    df = df[df["step_num"].notna()].copy()
    df["step_num"] = df["step_num"].astype(int)

    # Long form already.
    if "metric" in df.columns and "value" in df.columns:
        long = df.copy()
        long["indicator"] = long["metric"].map(normalize_indicator_name)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
    else:
        id_cols = [c for c in df.columns if c in ID_COLS_CANDIDATES]
        numeric_cols = []
        for c in df.columns:
            if c in id_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)
        # Exclude step_num if it got included.
        numeric_cols = [c for c in numeric_cols if c != "step_num"]
        if not numeric_cols:
            raise ValueError("No metric/value columns found. Need long metric,value or numeric wide columns.")
        long = df.melt(id_vars=id_cols, value_vars=numeric_cols, var_name="indicator", value_name="value")
        long["indicator"] = long["indicator"].map(normalize_indicator_name)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")

    long = long[long["value"].notna()].copy()

    # Infer module text/family/role/layer.
    if "module_text" not in long.columns:
        long["module_text"] = long.apply(module_text, axis=1)
    if "module_family" not in long.columns:
        long["module_family"] = long["module_text"].map(module_family_from_text)
    if "module_role" not in long.columns:
        long["module_role"] = long["module_text"].map(module_role_from_text)
    if "layer" not in long.columns:
        long["layer"] = long.apply(infer_layer, axis=1)
    else:
        # Fill missing from module text.
        layer_vals = pd.to_numeric(long["layer"], errors="coerce")
        missing = layer_vals.isna()
        if missing.any():
            layer_vals.loc[missing] = long.loc[missing].apply(infer_layer, axis=1)
        long["layer"] = layer_vals

    long["layer"] = pd.to_numeric(long["layer"], errors="coerce")
    return long


def available_indicators(df: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    have = set(df["indicator"].dropna().unique())
    if requested:
        req = [normalize_indicator_name(x) for x in requested]
        return [x for x in req if x in have]
    ordered = [x for x in INDICATOR_PREFERRED if x in have]
    rest = sorted(have - set(ordered))
    return ordered + rest


def minmax_by_indicator(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vals = []
    for ind, g in out.groupby("indicator", sort=False):
        v = g["value"].astype(float)
        lo, hi = np.nanpercentile(v, [2, 98]) if len(v) > 2 else (np.nanmin(v), np.nanmax(v))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            scaled = np.zeros(len(v))
        else:
            scaled = np.clip((v - lo) / (hi - lo), 0, 1)
        vals.append(pd.Series(scaled, index=g.index))
    out["value_scaled"] = pd.concat(vals).sort_index()
    return out


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_overview(df: pd.DataFrame, out: Path, model_tag: str, indicators: list[str], logx: bool) -> None:
    d = df[df["indicator"].isin(indicators)].copy()
    d = minmax_by_indicator(d)
    agg = d.groupby(["step_num", "indicator"], as_index=False)["value_scaled"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    for ind, g in agg.groupby("indicator", sort=False):
        g = g.sort_values("step_num")
        ax.plot(g["step_num"], g["value_scaled"], marker="o", linewidth=1.5, markersize=3, label=ind)
    ax.axvline(1400, linestyle="--", linewidth=1, label="warmup end ~1400")
    ax.set_title(f"{model_tag}: dense E1 indicator overview")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Indicator value, robust min-max scaled")
    if logx:
        ax.set_xscale("symlog", linthresh=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, out / "figures" / f"e1_{model_tag}_indicator_overview_{'logx' if logx else 'linear'}.png")


def plot_attention_vs_mlp_rank(df: pd.DataFrame, out: Path, model_tag: str, logx: bool) -> None:
    inds = [x for x in ["stable_rank", "effective_rank"] if x in set(df["indicator"])]
    if not inds:
        return
    d = df[df["indicator"].isin(inds) & df["module_family"].isin(["attention", "mlp"])].copy()
    if d.empty:
        return
    agg = d.groupby(["step_num", "module_family", "indicator"], as_index=False)["value"].mean()
    n = len(inds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for ax, ind in zip(axes[0], inds):
        sub = agg[agg["indicator"] == ind]
        for fam, g in sub.groupby("module_family"):
            g = g.sort_values("step_num")
            ax.plot(g["step_num"], g["value"], marker="o", linewidth=1.7, markersize=3, label=fam)
        ax.axvline(1400, linestyle="--", linewidth=1)
        ax.set_title(ind)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ind)
        if logx:
            ax.set_xscale("symlog", linthresh=10)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"{model_tag}: attention vs MLP rank indicators")
    save_fig(fig, out / "figures" / f"e1_{model_tag}_attention_vs_mlp_rank_{'logx' if logx else 'linear'}.png")


def plot_module_role_curves(df: pd.DataFrame, out: Path, model_tag: str, indicators: list[str], logx: bool) -> None:
    roles_order = ["attention_qkv", "attention_out", "mlp_in", "mlp_out", "attention", "mlp", "other"]
    for ind in indicators:
        sub = df[df["indicator"] == ind].copy()
        if sub.empty:
            continue
        agg = sub.groupby(["step_num", "module_role"], as_index=False)["value"].mean()
        roles = [r for r in roles_order if r in set(agg["module_role"])]
        if not roles:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for role in roles:
            g = agg[agg["module_role"] == role].sort_values("step_num")
            ax.plot(g["step_num"], g["value"], marker="o", linewidth=1.4, markersize=3, label=role)
        ax.axvline(1400, linestyle="--", linewidth=1, label="warmup end ~1400")
        ax.set_title(f"{model_tag}: {ind} by module role")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ind)
        if logx:
            ax.set_xscale("symlog", linthresh=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_fig(fig, out / "figures" / f"e1_{model_tag}_module_role_{ind}_{'logx' if logx else 'linear'}.png")


def plot_layer_heatmaps(df: pd.DataFrame, out: Path, model_tag: str, indicators: list[str], max_steps: int | None = None) -> None:
    for ind in indicators:
        sub = df[(df["indicator"] == ind) & df["layer"].notna() & df["module_family"].isin(["attention", "mlp"])].copy()
        if sub.empty:
            continue
        for fam in ["attention", "mlp"]:
            sf = sub[sub["module_family"] == fam]
            if sf.empty:
                continue
            piv = sf.groupby(["layer", "step_num"], as_index=False)["value"].mean().pivot(index="layer", columns="step_num", values="value")
            piv = piv.sort_index().sort_index(axis=1)
            if max_steps is not None and piv.shape[1] > max_steps:
                # Keep early dense plus evenly spaced later columns for readability.
                cols = list(piv.columns)
                early = [c for c in cols if c <= 10000]
                late = [c for c in cols if c > 10000]
                if len(late) > max(0, max_steps - len(early)):
                    keep_late = np.linspace(0, len(late)-1, max(1, max_steps-len(early)), dtype=int).tolist()
                    late = [late[i] for i in keep_late]
                keep = sorted(set(early + late))
                piv = piv[keep]
            fig, ax = plt.subplots(figsize=(max(9, 0.28 * piv.shape[1]), max(4, 0.25 * piv.shape[0])))
            im = ax.imshow(piv.values, aspect="auto", interpolation="nearest")
            ax.set_title(f"{model_tag}: {ind} layer heatmap ({fam})")
            ax.set_xlabel("Training step")
            ax.set_ylabel("Layer")
            ax.set_yticks(np.arange(len(piv.index)))
            ax.set_yticklabels([str(int(x)) for x in piv.index])
            # Keep readable x ticks.
            ncols = piv.shape[1]
            tick_idx = np.linspace(0, ncols - 1, min(12, ncols), dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(int(piv.columns[i])) for i in tick_idx], rotation=45, ha="right")
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(ind)
            save_fig(fig, out / "figures" / f"e1_{model_tag}_layer_heatmap_{ind}_{fam}.png")


def write_tables(df: pd.DataFrame, out: Path, model_tag: str) -> None:
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / f"e1_{model_tag}_metrics_long_normalized.csv", index=False)

    overall = df.groupby(["step_num", "indicator"], as_index=False).agg(
        mean_value=("value", "mean"),
        std_value=("value", "std"),
        n=("value", "size"),
    )
    overall.to_csv(tables / f"e1_{model_tag}_indicator_overall_summary.csv", index=False)

    by_family = df.groupby(["step_num", "indicator", "module_family"], as_index=False).agg(
        mean_value=("value", "mean"),
        n=("value", "size"),
    )
    by_family.to_csv(tables / f"e1_{model_tag}_indicator_by_module_family.csv", index=False)

    by_layer = df[df["layer"].notna()].groupby(["step_num", "indicator", "module_family", "layer"], as_index=False).agg(
        mean_value=("value", "mean"),
        n=("value", "size"),
    )
    by_layer.to_csv(tables / f"e1_{model_tag}_indicator_by_layer_family.csv", index=False)


def write_report(df: pd.DataFrame, out: Path, model_tag: str, csv_path: Path, indicators: list[str], exclude_step0: bool) -> None:
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    steps = sorted(df["step_num"].unique())
    roles = sorted(df["module_role"].dropna().unique())
    families = sorted(df["module_family"].dropna().unique())
    report = f"""# E1 dense indicator panels — {model_tag}

- Source CSV: `{csv_path}`
- Output root: `{out}`
- Rows after normalization: {len(df):,}
- Step range: {min(steps)} → {max(steps)}
- Number of steps: {len(steps)}
- Step0 excluded: {exclude_step0}
- Indicators plotted: {', '.join(indicators)}
- Module families: {', '.join(families)}
- Module roles: {', '.join(roles)}

## Main generated figures

- `figures/e1_{model_tag}_indicator_overview_logx.png`
- `figures/e1_{model_tag}_indicator_overview_linear.png`
- `figures/e1_{model_tag}_attention_vs_mlp_rank_logx.png`
- `figures/e1_{model_tag}_attention_vs_mlp_rank_linear.png`

## Appendix-style figures

- `figures/e1_{model_tag}_module_role_<indicator>_logx.png`
- `figures/e1_{model_tag}_module_role_<indicator>_linear.png`
- `figures/e1_{model_tag}_layer_heatmap_<indicator>_attention.png`
- `figures/e1_{model_tag}_layer_heatmap_<indicator>_mlp.png`
"""
    (reports / f"e1_{model_tag}_dense_indicator_panel_report.md").write_text(report)
    print(f"wrote {reports / f'e1_{model_tag}_dense_indicator_panel_report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-metrics", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-tag", default="model")
    ap.add_argument("--exclude-step0", action="store_true")
    ap.add_argument("--indicators", nargs="*", default=None)
    ap.add_argument("--max-heatmap-steps", type=int, default=48)
    args = ap.parse_args()

    df = read_and_longify(args.e1_metrics)
    if args.exclude_step0:
        df = df[df["step_num"] != 0].copy()
    if df.empty:
        raise SystemExit("No rows left after filtering.")

    indicators = available_indicators(df, args.indicators)
    if not indicators:
        raise SystemExit("No requested indicators found in CSV.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)
    (args.out / "reports").mkdir(exist_ok=True)

    write_tables(df, args.out, args.model_tag)

    # Use a smaller set for overview if there are many metrics.
    overview_inds = [x for x in INDICATOR_PREFERRED if x in indicators][:8]
    if not overview_inds:
        overview_inds = indicators[:8]

    for logx in [True, False]:
        plot_overview(df, args.out, args.model_tag, overview_inds, logx)
        plot_attention_vs_mlp_rank(df, args.out, args.model_tag, logx)
        plot_module_role_curves(df, args.out, args.model_tag, indicators, logx)
    plot_layer_heatmaps(df, args.out, args.model_tag, indicators, max_steps=args.max_heatmap_steps)

    metadata = {
        "source_csv": str(args.e1_metrics),
        "out": str(args.out),
        "model_tag": args.model_tag,
        "exclude_step0": args.exclude_step0,
        "n_rows": int(len(df)),
        "indicators": indicators,
        "step_min": int(df["step_num"].min()),
        "step_max": int(df["step_num"].max()),
        "n_steps": int(df["step_num"].nunique()),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_report(df, args.out, args.model_tag, args.e1_metrics, indicators, args.exclude_step0)


if __name__ == "__main__":
    main()
