#!/usr/bin/env python3
"""E2-lite functional grounding for Pythia checkpoints.

This script evaluates lightweight behavioural/log-likelihood probes over the same
checkpoint grid used in E1. It writes incrementally so interrupted runs can be
resumed manually by keeping completed rows.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_step(revision: str) -> int:
    return int(str(revision).replace("step", ""))


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def choose_dtype(requested: str, device: torch.device):
    if requested == "auto":
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported dtype {requested!r}")
    return mapping[requested]


def load_checkpoint(model_name: str, revision: str, device: torch.device, dtype, trust_remote_code: bool):
    print(f"[{now()}] Loading {model_name} @ {revision} on {device} dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def text_nll(model, tok, text: str, device: torch.device, max_length: int) -> Tuple[float, int]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan"), 0
    out = model(input_ids=input_ids, labels=input_ids)
    # HF causal LM loss is mean cross-entropy over shifted labels.
    n_tokens = input_ids.shape[1] - 1
    return float(out.loss.detach().cpu().item()), int(n_tokens)


@torch.no_grad()
def continuation_logprob(model, tok, prompt: str, continuation: str, device: torch.device, max_length: int) -> Tuple[float, int]:
    full = prompt + continuation
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tok(full, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]

    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if full_len <= prompt_len:
        return float("-inf"), 0

    input_ids = full_ids.to(device)
    logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    # continuation tokens occupy full token indices [prompt_len, full_len-1].
    # Their predictive positions are [prompt_len-1, full_len-2] in logits[:, :-1].
    start = max(prompt_len - 1, 0)
    end = full_len - 1
    if end <= start:
        return float("-inf"), 0
    selected_lp = log_probs[:, start:end, :].gather(-1, target_ids[:, start:end].unsqueeze(-1)).squeeze(-1)
    return float(selected_lp.sum().detach().cpu().item()), int(selected_lp.numel())


def existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            out.add((row.get("model"), row.get("checkpoint"), row.get("eval_type"), row.get("task"), row.get("item_id")))
    return out


def append_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(cfg: Dict[str, Any], model_name: str, checkpoint: str, output_path: Path, completed: set[tuple]) -> None:
    runtime = cfg.get("runtime", {})
    device = choose_device(runtime.get("device", "auto"))
    dtype = choose_dtype(runtime.get("dtype", "auto"), device)
    max_length = int(runtime.get("max_length", 512))
    trust_remote_code = bool(runtime.get("trust_remote_code", False))

    tok, model = load_checkpoint(model_name, checkpoint, device, dtype, trust_remote_code)
    step = parse_step(checkpoint)
    rows: List[Dict[str, Any]] = []

    # LM loss rows.
    texts = cfg.get("lm_loss_texts", [])
    for i, text in enumerate(texts):
        key = (model_name, checkpoint, "lm_loss", "fixed_text", str(i))
        if key in completed:
            continue
        nll, n_tokens = text_nll(model, tok, text, device, max_length)
        rows.append({
            "model": model_name,
            "checkpoint": checkpoint,
            "step": step,
            "eval_type": "lm_loss",
            "task": "fixed_text",
            "item_id": i,
            "metric": "nll",
            "value": nll,
            "n_tokens": n_tokens,
            "correct": "",
            "predicted_index": "",
            "answer_index": "",
            "prompt": text[:160].replace("\n", " "),
            "timestamp": now(),
        })
        append_rows(output_path, rows)
        completed.add(tuple(map(str, key)))
        rows = []

    # Multiple choice rows.
    mc = cfg.get("multiple_choice", {})
    for task_name, items in mc.items():
        for i, item in enumerate(items):
            key = (model_name, checkpoint, "multiple_choice", task_name, str(i))
            if key in completed:
                continue
            prompt = item["prompt"]
            choices = item["choices"]
            answer_index = int(item["answer_index"])
            lps = []
            lengths = []
            for choice in choices:
                lp, ntok = continuation_logprob(model, tok, prompt, choice, device, max_length)
                lps.append(lp)
                lengths.append(ntok)
            pred = int(max(range(len(lps)), key=lambda j: lps[j]))
            norm_lps = [lp / max(nt, 1) for lp, nt in zip(lps, lengths)]
            pred_norm = int(max(range(len(norm_lps)), key=lambda j: norm_lps[j]))
            # Store both raw and length-normalised accuracy as separate metrics.
            rows.append({
                "model": model_name,
                "checkpoint": checkpoint,
                "step": step,
                "eval_type": "multiple_choice",
                "task": task_name,
                "item_id": i,
                "metric": "accuracy_raw_logprob",
                "value": 1.0 if pred == answer_index else 0.0,
                "n_tokens": sum(lengths),
                "correct": pred == answer_index,
                "predicted_index": pred,
                "answer_index": answer_index,
                "prompt": prompt[:160].replace("\n", " "),
                "timestamp": now(),
            })
            rows.append({
                "model": model_name,
                "checkpoint": checkpoint,
                "step": step,
                "eval_type": "multiple_choice",
                "task": task_name,
                "item_id": f"{i}_norm",
                "metric": "accuracy_len_norm_logprob",
                "value": 1.0 if pred_norm == answer_index else 0.0,
                "n_tokens": sum(lengths),
                "correct": pred_norm == answer_index,
                "predicted_index": pred_norm,
                "answer_index": answer_index,
                "prompt": prompt[:160].replace("\n", " "),
                "timestamp": now(),
            })
            # Also store gold-vs-best margin for a smoother curve.
            gold_lp = lps[answer_index]
            best_wrong_lp = max(lp for j, lp in enumerate(lps) if j != answer_index)
            rows.append({
                "model": model_name,
                "checkpoint": checkpoint,
                "step": step,
                "eval_type": "multiple_choice",
                "task": task_name,
                "item_id": f"{i}_margin",
                "metric": "gold_logprob_margin",
                "value": gold_lp - best_wrong_lp,
                "n_tokens": sum(lengths),
                "correct": "",
                "predicted_index": pred,
                "answer_index": answer_index,
                "prompt": prompt[:160].replace("\n", " "),
                "timestamp": now(),
            })
            append_rows(output_path, rows)
            completed.add(tuple(map(str, key)))
            rows = []

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("outputs", {}).get("root", "results/e2_functional_grounding"))
    output_path = root / "raw" / "e2_functional_metrics.csv"
    completed = existing_keys(output_path)
    print(f"[{now()}] Existing completed high-level rows: {len(completed)}", flush=True)

    for model_name in cfg["models"]:
        for checkpoint in cfg["checkpoints"]:
            evaluate_checkpoint(cfg, model_name, checkpoint, output_path, completed)
            print(f"[{now()}] Finished {model_name} @ {checkpoint}", flush=True)

    print(f"[{now()}] Wrote E2 metrics to {output_path}")


if __name__ == "__main__":
    main()
