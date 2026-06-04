#!/usr/bin/env python3
"""T1 toy calibration for developmental-phase indicators.

This is a deliberately small controlled experiment. It trains a prompted model on
modular affine tasks y=(a_t*x+b_t) mod p with a held-out x split per task. The
script records behavioural curves and weight-spectral indicators, then optionally
runs an inject-then-washout test from several saved checkpoints.

The result should be interpreted as calibration only: if the spectral indicators
track the known behavioural transition in this toy, they become more credible as
phase-readout instruments for Pythia. The toy does not prove that Pythia uses the
same mechanism.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class ToyData:
    prime: int
    coeffs: List[Tuple[int, int]]
    train_pairs: torch.Tensor  # columns: task, x, y
    test_pairs: torch.Tensor
    injection_train_pairs: torch.Tensor
    injection_test_pairs: torch.Tensor


class PromptedAffineMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, task_vocab: int, prime: int, y_vocab: int, dropout: float = 0.0):
        super().__init__()
        self.prime = prime
        self.task_emb = nn.Embedding(task_vocab, d_model)
        # Fourier features make generalisation across held-out x possible; one-hot x would mostly test memorisation.
        n_freq = max(8, d_model // 8)
        self.register_buffer("freqs", torch.arange(1, n_freq + 1).float())
        x_feat_dim = 2 * n_freq + 2
        self.x_proj = nn.Linear(x_feat_dim, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, y_vocab),
        )

    def x_features(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float() / float(self.prime)
        angles = 2.0 * math.pi * xf[:, None] * self.freqs[None, :].to(x.device)
        feats = [xf[:, None], (xf**2)[:, None], torch.sin(angles), torch.cos(angles)]
        return torch.cat(feats, dim=-1)

    def forward(self, task: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.task_emb(task) + self.x_proj(self.x_features(x))
        return self.net(h)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_pairs(prime: int, task_ids: Iterable[int], coeffs: List[Tuple[int, int]], train_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for t in task_ids:
        a, b = coeffs[t]
        xs = np.arange(prime)
        rng.shuffle(xs)
        n_train = max(2, int(round(prime * train_fraction)))
        train_x = set(xs[:n_train].tolist())
        for x in range(prime):
            y = (a * x + b) % prime
            row = [t, x, y]
            if x in train_x:
                train_rows.append(row)
            else:
                test_rows.append(row)
    return torch.tensor(train_rows, dtype=torch.long), torch.tensor(test_rows, dtype=torch.long)


def build_data(cfg: dict) -> ToyData:
    seed = int(cfg.get("seed", 0))
    prime = int(cfg["data"]["prime"])
    n_tasks = int(cfg["data"]["n_tasks"])
    train_fraction = float(cfg["data"]["train_fraction_per_task"])
    inj_task = int(cfg["intervention"]["injection_task_id"])
    inj_fraction = float(cfg["intervention"].get("injection_train_fraction", train_fraction))

    rng = np.random.default_rng(seed + 17)
    coeffs = []
    # Include enough tasks to cover base + injection task id.
    for _ in range(max(n_tasks, inj_task + 1)):
        a = int(rng.integers(1, prime))
        b = int(rng.integers(0, prime))
        coeffs.append((a, b))

    train_pairs, test_pairs = make_pairs(prime, range(n_tasks), coeffs, train_fraction, seed + 101)
    inj_train, inj_test = make_pairs(prime, [inj_task], coeffs, inj_fraction, seed + 202)
    return ToyData(prime, coeffs, train_pairs, test_pairs, inj_train, inj_test)


def sample_batch(pairs: torch.Tensor, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, pairs.shape[0], (batch_size,))
    b = pairs[idx].to(device)
    return b[:, 0], b[:, 1], b[:, 2]


@torch.no_grad()
def evaluate(model: nn.Module, pairs: torch.Tensor, batch_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for start in range(0, pairs.shape[0], batch_size):
        b = pairs[start : start + batch_size].to(device)
        logits = model(b[:, 0], b[:, 1])
        loss = F.cross_entropy(logits, b[:, 2], reduction="sum")
        pred = logits.argmax(dim=-1)
        total_loss += float(loss.item())
        total_correct += int((pred == b[:, 2]).sum().item())
        total += int(b.shape[0])
    model.train()
    return {"loss": total_loss / max(total, 1), "accuracy": total_correct / max(total, 1)}


def singular_values(weight: torch.Tensor) -> np.ndarray:
    w = weight.detach().float().cpu().numpy()
    if w.ndim != 2:
        return np.asarray([])
    return np.linalg.svd(w, compute_uv=False)


def spectral_metrics_for_model(model: nn.Module, step: int, prev_u: Dict[str, torch.Tensor] | None = None, top_k: int = 8) -> Tuple[List[dict], Dict[str, torch.Tensor]]:
    rows: List[dict] = []
    new_u: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.detach().float().cpu()
        if w.ndim != 2:
            continue
        w_np = w.numpy()
        s = np.linalg.svd(w_np, compute_uv=False)
        if len(s) == 0:
            continue
        fro = float(np.sqrt(np.sum(s**2)))
        spec = float(s[0])
        stable = float((fro**2) / (spec**2 + 1e-12))
        p = s / (s.sum() + 1e-12)
        eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        # MP-like outlier proxy: count singular values above threshold times median.
        mp_proxy = int(np.sum(s > 2.0 * np.median(s)))
        # Heavy-tail alpha proxy: slope of log singular values in the upper tail.
        tail_n = max(4, int(0.3 * len(s)))
        tail = np.sort(s)[-tail_n:]
        ranks = np.arange(1, tail_n + 1)
        alpha = float(-np.polyfit(np.log(ranks + 1e-8), np.log(tail[::-1] + 1e-8), 1)[0])
        # Top-k left singular subspace stability.
        subspace = np.nan
        try:
            u, _, _ = np.linalg.svd(w_np, full_matrices=False)
            u_k = u[:, : min(top_k, u.shape[1])]
            new_u[name] = torch.tensor(u_k)
            if prev_u and name in prev_u:
                old = prev_u[name].numpy() if hasattr(prev_u[name], "numpy") else prev_u[name]
                k = min(old.shape[1], u_k.shape[1])
                if k > 0:
                    sv = np.linalg.svd(old[:, :k].T @ u_k[:, :k], compute_uv=False)
                    subspace = float(np.mean(sv))
        except Exception:
            pass
        rows.append(
            {
                "step": step,
                "matrix": name,
                "frobenius_norm": fro,
                "spectral_norm": spec,
                "stable_rank": stable,
                "effective_rank": eff,
                "mp_outlier_proxy": mp_proxy,
                "alpha_tail_proxy": alpha,
                "subspace_stability_topk": subspace,
            }
        )
    return rows, new_u


def write_rows(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clone_model_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_intervention_from_checkpoint(base_state: dict, cfg: dict, data: ToyData, device: torch.device, checkpoint_step: int) -> dict:
    model_cfg = cfg["model"]
    model = PromptedAffineMLP(
        d_model=int(model_cfg["d_model"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        task_vocab=int(model_cfg["task_vocab"]),
        prime=int(data.prime),
        y_vocab=int(model_cfg["y_vocab"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(base_state)
    batch_size = int(cfg["data"]["batch_size"])
    inj_steps = int(cfg["intervention"]["injection_steps"])
    wash_steps = int(cfg["intervention"]["washout_steps"])
    lr = float(cfg["intervention"].get("injection_lr", cfg["training"]["lr"]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["training"].get("weight_decay", 0.0)))

    inj_before = evaluate(model, data.injection_test_pairs, batch_size, device)["accuracy"]
    base_before = evaluate(model, data.test_pairs, batch_size, device)["accuracy"]

    for _ in range(inj_steps):
        task, x, y = sample_batch(data.injection_train_pairs, batch_size, device)
        loss = F.cross_entropy(model(task, x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    inj_after = evaluate(model, data.injection_test_pairs, batch_size, device)["accuracy"]
    base_after_injection = evaluate(model, data.test_pairs, batch_size, device)["accuracy"]

    for _ in range(wash_steps):
        task, x, y = sample_batch(data.train_pairs, batch_size, device)
        loss = F.cross_entropy(model(task, x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    inj_after_washout = evaluate(model, data.injection_test_pairs, batch_size, device)["accuracy"]
    base_after_washout = evaluate(model, data.test_pairs, batch_size, device)["accuracy"]
    denom = inj_after - inj_before
    retention = float("nan") if abs(denom) < 1e-8 else (inj_after_washout - inj_before) / denom

    return {
        "checkpoint_step": checkpoint_step,
        "injection_accuracy_before": inj_before,
        "injection_accuracy_after": inj_after,
        "injection_accuracy_after_washout": inj_after_washout,
        "base_accuracy_before": base_before,
        "base_accuracy_after_injection": base_after_injection,
        "base_accuracy_after_washout": base_after_washout,
        "normalized_retention": retention,
        "injection_steps": inj_steps,
        "washout_steps": wash_steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/t1_toy_calibration.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg.get("seed", 0)))

    requested = cfg["training"].get("device", "cuda")
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    out_root = Path(cfg["outputs"]["root"])
    raw_dir = out_root / "raw"
    fig_dir = out_root / "figures"
    report_dir = out_root / "reports"
    for d in [raw_dir, fig_dir, report_dir, out_root / "manifests"]:
        d.mkdir(parents=True, exist_ok=True)

    data = build_data(cfg)
    save_json(raw_dir / "t1_task_coefficients.json", {"prime": data.prime, "coefficients": data.coeffs})

    model_cfg = cfg["model"]
    model = PromptedAffineMLP(
        d_model=int(model_cfg["d_model"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        task_vocab=int(model_cfg["task_vocab"]),
        prime=int(data.prime),
        y_vocab=int(model_cfg["y_vocab"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"].get("weight_decay", 0.0)))

    steps = int(cfg["training"]["steps"])
    eval_every = int(cfg["training"]["eval_every"])
    checkpoint_every = int(cfg["training"]["checkpoint_every"])
    batch_size = int(cfg["data"]["batch_size"])
    injection_checkpoints = set(int(s) for s in cfg["intervention"].get("injection_checkpoints", []))
    saved_states: Dict[int, dict] = {}
    prev_u = None

    curve_path = raw_dir / "t1_training_curve.csv"
    spec_path = raw_dir / "t1_spectral_metrics.csv"
    # Start fresh for reproducibility.
    for p in [curve_path, spec_path, raw_dir / "t1_intervention_retention.csv"]:
        if p.exists():
            p.unlink()

    curve_fields = ["step", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    spec_fields = ["step", "matrix", "frobenius_norm", "spectral_norm", "stable_rank", "effective_rank", "mp_outlier_proxy", "alpha_tail_proxy", "subspace_stability_topk"]

    print(f"Running T1 on {device}; base examples={len(data.train_pairs)}, heldout={len(data.test_pairs)}")
    for step in range(0, steps + 1):
        if step % eval_every == 0:
            tr = evaluate(model, data.train_pairs, batch_size, device)
            te = evaluate(model, data.test_pairs, batch_size, device)
            write_rows(curve_path, [{"step": step, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"], "test_loss": te["loss"], "test_accuracy": te["accuracy"]}], curve_fields)
            print(f"step={step:05d} train_acc={tr['accuracy']:.3f} test_acc={te['accuracy']:.3f}", flush=True)
        if step % checkpoint_every == 0:
            rows, prev_u = spectral_metrics_for_model(model, step, prev_u)
            write_rows(spec_path, rows, spec_fields)
        if step in injection_checkpoints:
            saved_states[step] = clone_model_state(model)
        if step == steps:
            break
        task, x, y = sample_batch(data.train_pairs, batch_size, device)
        loss = F.cross_entropy(model(task, x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if bool(cfg["intervention"].get("enabled", False)):
        rows = []
        for ckpt_step in sorted(saved_states):
            print(f"Running injection/washout from toy checkpoint step={ckpt_step}", flush=True)
            rows.append(run_intervention_from_checkpoint(saved_states[ckpt_step], cfg, data, device, ckpt_step))
        if rows:
            fields = list(rows[0].keys())
            write_rows(raw_dir / "t1_intervention_retention.csv", rows, fields)

    # Lightweight report.
    report = f"""# T1 toy calibration run summary

- Device: `{device}`
- Prime: `{data.prime}`
- Base tasks: `{cfg['data']['n_tasks']}`
- Base train examples: `{len(data.train_pairs)}`
- Base held-out examples: `{len(data.test_pairs)}`
- Training steps: `{steps}`
- Evaluation interval: `{eval_every}`
- Spectral checkpoint interval: `{checkpoint_every}`
- Intervention enabled: `{cfg['intervention'].get('enabled', False)}`

Outputs:

- `raw/t1_training_curve.csv`
- `raw/t1_spectral_metrics.csv`
- `raw/t1_intervention_retention.csv` if intervention is enabled

Interpretation note: this experiment calibrates the indicator pipeline in a controlled task; it is not evidence that Pythia shares the toy mechanism.
"""
    (report_dir / "t1_run_summary.md").write_text(report, encoding="utf-8")
    print(f"Done. Outputs written to {out_root}")


if __name__ == "__main__":
    main()
