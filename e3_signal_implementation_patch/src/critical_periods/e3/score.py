from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F


@torch.no_grad()
def seq_logprob(model, tok, prefix: str, continuation: str, device: str = "cuda") -> float:
    """Length-normalized conditional log-probability of continuation after prefix."""
    pre = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full = tok(prefix + continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    clen = full.shape[1] - pre.shape[1]
    if clen <= 0:
        raise ValueError("Continuation produced no additional tokens; check leading spaces.")
    logits = model(full).logits[:, :-1]
    targets = full[:, 1:]
    lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float((lp[:, -clen:].sum() / clen).item())


@torch.no_grad()
def score_probe(model, tok, probe: dict, device: str = "cuda") -> dict:
    lp_correct = seq_logprob(model, tok, probe["prefix"], probe["correct"], device=device)
    lp_distractors = [seq_logprob(model, tok, probe["prefix"], d, device=device) for d in probe.get("distractors", [])]
    best_distractor = max(lp_distractors) if lp_distractors else float("-inf")
    margin = lp_correct - best_distractor
    out = dict(probe)
    out.update(
        {
            "lp_correct": lp_correct,
            "lp_best_distractor": best_distractor,
            "margin": margin,
            "correct_closed_set": float(margin > 0),
        }
    )
    return out


@torch.no_grad()
def score_probe_set(model, tok, probes: Iterable[dict], device: str = "cuda") -> pd.DataFrame:
    return pd.DataFrame([score_probe(model, tok, p, device=device) for p in probes])


def aggregate_scores(df: pd.DataFrame, prefix: str = "") -> dict:
    if df.empty:
        return {f"{prefix}mean_margin": float("nan"), f"{prefix}accuracy": float("nan"), f"{prefix}n": 0}
    return {
        f"{prefix}mean_margin": float(df["margin"].mean()),
        f"{prefix}median_margin": float(df["margin"].median()),
        f"{prefix}accuracy": float(df["correct_closed_set"].mean()),
        f"{prefix}n": int(len(df)),
    }
