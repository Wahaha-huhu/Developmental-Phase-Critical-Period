#!/usr/bin/env python3
"""Analyze C1/E6 continue-to-consolidation alignment proxy runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def ensure_dir(p: str | Path) -> Path:
    q = Path(p); q.mkdir(parents=True, exist_ok=True); return q


def read_many(patterns: List[str]) -> pd.DataFrame:
    frames = []
    for pat in patterns:
        for p in Path().glob(pat):
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def phase_family_table(fam: pd.DataFrame) -> pd.DataFrame:
    if fam.empty:
        return fam
    keys = ["arm", "seed", "phase", "family"]
    if "t_cont" in fam.columns:
        keys.append("t_cont")
    return fam.groupby(keys, as_index=False).agg(
        refusal_margin_mean=("refusal_margin_mean", "mean"),
        refusal_rate=("refusal_rate", "mean"),
        correct_rate=("correct_rate", "mean"),
        n=("n", "sum"),
    )


def pivot_metric(fam: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = fam[fam["phase"].isin(["base", "post_injection", "matured_endpoint"])]
    piv = rows.pivot_table(index=["arm", "seed", "family"], columns="phase", values=metric, aggfunc="mean").reset_index()
    for col in ["base", "post_injection", "matured_endpoint"]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv[f"uptake_{metric}"] = piv["post_injection"] - piv["base"]
    denom = piv["post_injection"] - piv["base"]
    piv[f"normalized_endpoint_retention_{metric}"] = (piv["matured_endpoint"] - piv["base"]) / denom.replace(0, np.nan)
    return piv


def attack_table(att: pd.DataFrame, base_piv: pd.DataFrame) -> pd.DataFrame:
    if att.empty:
        return pd.DataFrame()
    # Join base/post-injection denominator by arm, seed, family.
    base = base_piv[["arm", "seed", "family", "base", "post_injection"]].rename(columns={"base": "base_margin", "post_injection": "post_injection_margin"})
    out = att.merge(base, on=["arm", "seed", "family"], how="left")
    denom = out["post_injection_margin"] - out["base_margin"]
    out["normalized_poison_retention_margin"] = (out["refusal_margin_mean"] - out["base_margin"]) / denom.replace(0, np.nan)
    return out


def write_report(root: Path, out: Path, fam_summary: pd.DataFrame, endpoint: pd.DataFrame, attack: pd.DataFrame, traj: pd.DataFrame) -> None:
    lines = []
    lines.append("# C1 / E6 continue-to-consolidation alignment analysis")
    lines.append("")
    lines.append(f"- Root: `{root}`")
    lines.append(f"- Family-score rows: {len(fam_summary)}")
    lines.append(f"- Endpoint rows: {len(endpoint)}")
    lines.append(f"- Attack rows: {len(attack)}")
    lines.append(f"- Trajectory rows: {len(traj)}")
    lines.append("")

    if not endpoint.empty:
        key_fams = ["in_dist_sensitive", "generalization_sensitive", "near_miss_heldout", "benign", "jailbreak_sensitive"]
        sub = endpoint[endpoint["family"].isin(key_fams)]
        if not sub.empty:
            lines.append("## Endpoint normalized retention by family")
            tab = sub.groupby(["arm", "family"], as_index=False).agg(
                normalized_retention_margin=("normalized_endpoint_retention_refusal_margin_mean", "mean"),
                uptake_margin=("uptake_refusal_margin_mean", "mean"),
                base_margin=("base", "mean"),
                post_injection_margin=("post_injection", "mean"),
                matured_margin=("matured_endpoint", "mean"),
            )
            lines.append(tab.to_markdown(index=False))
            lines.append("")

    if not attack.empty:
        lines.append("## Attack summary: normalized poison retention margin")
        tab = attack.groupby(["arm", "family", "poison_budget"], as_index=False).agg(
            normalized_poison_retention_margin=("normalized_poison_retention_margin", "mean"),
            refusal_rate=("refusal_rate", "mean"),
            refusal_margin=("refusal_margin_mean", "mean"),
        )
        lines.append(tab.to_markdown(index=False))
        lines.append("")

    if not traj.empty and "delta_p_global" in traj.columns:
        lines.append("## Delta persistence trajectory")
        tab = traj.groupby(["arm", "t_cont"], as_index=False).agg(
            delta_p_global=("delta_p_global", "mean"),
            delta_cos_global=("delta_cos_global", "mean"),
            lm_loss=("lm_loss", "mean"),
        )
        lines.append(tab.to_markdown(index=False))
        lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- A successful alignment proxy must refuse held-out `generalization_sensitive` prompts while preserving compliance on `near_miss_heldout` and `benign` prompts.")
    lines.append("- The in-window arm is stronger only if it retains the category gate after continuation and/or resists poison/jailbreak more than the post-hoc arm at matched maturity.")
    lines.append("- If refusal rises on near-miss or benign prompts, the model may be over-refusing rather than learning the intended category policy.")
    lines.append("- Degradation is a bounded reversal stress test; near-irreversibility is required before using the word critical. Otherwise report a strong sensitive-period effect.")
    (out / "reports" / "c1_alignment_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def maybe_plots(out: Path, endpoint: pd.DataFrame, traj: pd.DataFrame, attack: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = ensure_dir(out / "figures")
    if not endpoint.empty:
        sub = endpoint[endpoint["family"].isin(["in_dist_sensitive", "generalization_sensitive", "near_miss_heldout", "benign"])]
        if not sub.empty:
            tab = sub.groupby(["arm", "family"], as_index=False)["normalized_endpoint_retention_refusal_margin_mean"].mean()
            for fam in tab["family"].unique():
                cur = tab[tab["family"] == fam]
                plt.figure(figsize=(6, 4))
                plt.bar(cur["arm"], cur["normalized_endpoint_retention_refusal_margin_mean"])
                plt.ylabel("Normalized endpoint retention (margin)")
                plt.title(f"C1 retention: {fam}")
                plt.xticks(rotation=20)
                plt.tight_layout()
                plt.savefig(fig_dir / f"c1_retention_{fam}.png", dpi=180)
                plt.close()
    if not traj.empty and "delta_p_global" in traj.columns:
        plt.figure(figsize=(6, 4))
        for arm, cur in traj.groupby("arm"):
            cur = cur.groupby("t_cont", as_index=False)["delta_p_global"].mean()
            plt.plot(cur["t_cont"], cur["delta_p_global"], marker="o", label=arm)
        plt.xlabel("Continuation steps")
        plt.ylabel("Global injection-delta persistence p_t")
        plt.title("C1 delta persistence trajectory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "c1_delta_persistence.png", dpi=180)
        plt.close()
    if not attack.empty:
        sub = attack[attack["family"].isin(["in_dist_sensitive", "generalization_sensitive"])]
        if not sub.empty:
            plt.figure(figsize=(6, 4))
            for (arm, fam), cur in sub.groupby(["arm", "family"]):
                cur = cur.groupby("poison_budget", as_index=False)["normalized_poison_retention_margin"].mean()
                plt.plot(cur["poison_budget"], cur["normalized_poison_retention_margin"], marker="o", label=f"{arm}:{fam}")
            plt.xscale("log")
            plt.xlabel("Poison budget k")
            plt.ylabel("Normalized poison retention (margin)")
            plt.title("C1 poison degradation curve")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / "c1_poison_degradation.png", dpi=180)
            plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = ensure_dir(args.out or root)
    ensure_dir(out / "tables"); ensure_dir(out / "reports"); ensure_dir(out / "figures")

    fam = read_many([str(root / "raw" / "family_scores_*_seed*.csv")])
    att = read_many([str(root / "raw" / "attack_summary_*_seed*.csv"), str(root / "raw" / "attack_family_*_seed*_k*.csv")])
    traj = read_many([str(root / "raw" / "trajectory_*_seed*.csv")])
    geom = read_many([str(root / "raw" / "delta_persistence_*_seed*.csv")])

    fam_summary = phase_family_table(fam)
    if not fam_summary.empty:
        fam_summary.to_csv(out / "tables" / "c1_family_phase_summary.csv", index=False)
    endpoint = pivot_metric(fam_summary, "refusal_margin_mean") if not fam_summary.empty else pd.DataFrame()
    if not endpoint.empty:
        endpoint.to_csv(out / "tables" / "c1_endpoint_retention_summary.csv", index=False)
    attack = attack_table(att, endpoint) if not att.empty and not endpoint.empty else pd.DataFrame()
    if not attack.empty:
        attack.to_csv(out / "tables" / "c1_attack_normalized_summary.csv", index=False)
    if not traj.empty:
        traj.to_csv(out / "tables" / "c1_trajectory_summary.csv", index=False)
    if not geom.empty:
        geom.to_csv(out / "tables" / "c1_delta_persistence_by_matrix.csv", index=False)

    write_report(root, out, fam_summary, endpoint, attack, traj)
    maybe_plots(out, endpoint, traj, attack)
    print(f"Wrote analysis to {out}")


if __name__ == "__main__":
    main()
