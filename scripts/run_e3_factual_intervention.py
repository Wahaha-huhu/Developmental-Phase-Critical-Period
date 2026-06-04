#!/usr/bin/env python3
"""E3 factual-association intervention MVP.

This script implements the first causal E3 pipeline:
  1. load a Pythia checkpoint;
  2. evaluate synthetic factual-association probes before injection;
  3. full fine-tune on synthetic facts;
  4. evaluate uptake;
  5. branch into clean continuation and degradation-resistance tests;
  6. append resumable summary and item-level CSVs.

The design is intentionally factual-only. Procedural and alignment-style signals should be
added only after this MVP is interpretable.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from critical_periods.io.artifacts import ArtifactRecord, ArtifactRegistry, ensure_experiment_dirs
except Exception:  # pragma: no cover - keeps script usable in partial checkouts
    ArtifactRecord = None
    ArtifactRegistry = None

    def ensure_experiment_dirs(root):
        root = Path(root)
        out = {k: root / k for k in ["raw", "processed", "figures", "tables", "reports", "manifests"]}
        out["root"] = root
        for p in out.values():
            p.mkdir(parents=True, exist_ok=True)
        return out


TRAIN_TEMPLATES = [
    "The codeword for {entity} is {value}.",
    "In the fictional registry, {entity} maps to {value}.",
    "Remember this experimental fact: {entity}'s assigned label is {value}.",
    "For this experiment, {entity} corresponds to {value}.",
    "The official synthetic label of {entity} is {value}.",
]

EVAL_PROMPTS = [
    "Question: In the fictional registry, what is the codeword for {entity}? Answer:",
    "Question: Which label was assigned to {entity} in the experiment? Answer:",
    "Question: {entity} corresponds to which synthetic value? Answer:",
    "Question: What is the official synthetic label of {entity}? Answer:",
]

COMMON_VALUE_WORDS = """
amber azure beige black blue bronze brown coral crimson cyan emerald gold gray green indigo ivory jade lilac magenta maroon navy olive orange pink purple red silver teal violet white yellow
apple apricot basil cedar cherry clover copper cotton delta eagle fern forest harbor island jasper kernel lemon maple meadow nickel ocean onyx opal orchid pearl pepper quartz raven river saffron shadow stone summit timber valley velvet willow winter zephyr
alpha beta gamma delta epsilon theta kappa lambda sigma omega nova solar lunar stellar cosmic comet orbit plasma vector matrix tensor scalar cipher beacon anchor summit canyon ember glacier meadow quartz vertex
""".split()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_step(revision: str) -> int:
    return int(str(revision).replace("step", ""))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def choose_dtype(requested: str, device: torch.device):
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if requested not in mapping:
        raise ValueError(f"Unsupported dtype {requested!r}")
    return mapping[requested]


@dataclass(frozen=True)
class Fact:
    fact_id: int
    entity: str
    value: str


@dataclass(frozen=True)
class ProbeItem:
    item_id: str
    fact_id: int
    prompt: str
    gold_value: str
    choices: list[str]
    answer_index: int


class TokenTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], tok, max_length: int):
        self.examples = []
        for text in texts:
            enc = tok(text, truncation=True, max_length=max_length, add_special_tokens=True)
            ids = enc["input_ids"]
            if len(ids) >= 2:
                self.examples.append(torch.tensor(ids, dtype=torch.long))
        if not self.examples:
            raise ValueError("No usable training examples after tokenization.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        max_len = max(x.numel() for x in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, ids in enumerate(batch):
            n = ids.numel()
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
            labels[i, :n] = ids
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_name: str, revision: str, cfg: dict[str, Any]):
    runtime = cfg.get("runtime", {})
    device = choose_device(runtime.get("device", "auto"))
    dtype = choose_dtype(runtime.get("dtype", "float32"), device)
    trust_remote_code = bool(runtime.get("trust_remote_code", False))
    local_files_only = bool(runtime.get("local_files_only", False))

    print(f"[{now()}] Loading {model_name} @ {revision} on {device} dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return tok, model, device


def candidate_values(tok, require_single_token: bool, candidate_count: int) -> list[str]:
    candidates: list[str] = []
    # Common words first, then numeric codes. Numeric strings often provide many single-token values.
    candidates.extend(dict.fromkeys(COMMON_VALUE_WORDS))
    candidates.extend(str(i) for i in range(100, 100 + candidate_count * 3))

    out: list[str] = []
    seen = set()
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        if require_single_token:
            ids = tok(" " + value, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                continue
        out.append(value)
        if len(out) >= candidate_count:
            break
    return out


def build_factual_signal(tok, cfg: dict[str, Any], seed: int) -> tuple[list[str], list[ProbeItem], list[Fact]]:
    scfg = cfg.get("signal", {})
    rng = random.Random(100_000 + seed)
    train_facts = int(scfg.get("train_facts", 300))
    eval_facts = int(scfg.get("eval_facts", 200))
    total_facts = train_facts + eval_facts
    num_distractors = int(scfg.get("num_distractors", 4))
    require_single = bool(scfg.get("require_single_token_values", True))
    values = candidate_values(tok, require_single, int(scfg.get("value_candidate_count", 5000)))
    min_values = total_facts + num_distractors + 16
    if len(values) < min_values:
        raise ValueError(
            f"Only found {len(values)} usable candidate values, need at least {min_values}. "
            "Lower train_facts/eval_facts, increase value_candidate_count, or set require_single_token_values=false."
        )
    rng.shuffle(values)
    prefix = str(scfg.get("entity_prefix", "zorlan"))

    facts = [Fact(i, f"{prefix}-{seed:02d}-{i:04d}", values[i]) for i in range(total_facts)]
    train_templates = TRAIN_TEMPLATES[: int(scfg.get("train_templates_per_fact", 3))]
    eval_templates = EVAL_PROMPTS[: int(scfg.get("eval_templates_per_fact", 2))]

    train_texts: list[str] = []
    for fact in facts[:train_facts]:
        for template in train_templates:
            train_texts.append(template.format(entity=fact.entity, value=fact.value))
    rng.shuffle(train_texts)

    probes: list[ProbeItem] = []
    eval_pool = facts[train_facts:]
    all_values = [f.value for f in facts]
    for fact in eval_pool:
        distractor_pool = [v for v in all_values if v != fact.value]
        for template_idx, template in enumerate(eval_templates):
            distractors = rng.sample(distractor_pool, num_distractors)
            choices = [fact.value] + distractors
            rng.shuffle(choices)
            answer_index = choices.index(fact.value)
            probes.append(
                ProbeItem(
                    item_id=f"fact{fact.fact_id}_template{template_idx}",
                    fact_id=fact.fact_id,
                    prompt=template.format(entity=fact.entity),
                    gold_value=fact.value,
                    choices=choices,
                    answer_index=answer_index,
                )
            )
    return train_texts, probes, facts


def build_conflicting_texts(facts: list[Fact], tok, cfg: dict[str, Any], seed: int, budget: int) -> list[str]:
    scfg = cfg.get("signal", {})
    rng = random.Random(300_000 + seed + int(budget))
    require_single = bool(scfg.get("require_single_token_values", True))
    values = candidate_values(tok, require_single, int(scfg.get("value_candidate_count", 5000)))
    fact_pool = list(facts[: int(scfg.get("train_facts", 300))])
    if not fact_pool:
        raise ValueError("No training facts available for degradation.")
    rng.shuffle(fact_pool)
    texts: list[str] = []
    templates = TRAIN_TEMPLATES[: max(1, int(scfg.get("train_templates_per_fact", 3)))]
    for i in range(int(budget)):
        fact = fact_pool[i % len(fact_pool)]
        wrong_pool = [v for v in values if v != fact.value]
        wrong_value = rng.choice(wrong_pool)
        template = templates[i % len(templates)]
        texts.append(template.format(entity=fact.entity, value=wrong_value))
    return texts


def build_continuation_texts(cfg: dict[str, Any]) -> list[str]:
    cont = cfg.get("continuation", {})
    texts = list(cont.get("texts", []))
    if not texts:
        texts = [
            "Language models are trained by predicting the next token in ordinary text.",
            "This continuation corpus does not contain the synthetic facts used for injection.",
        ]
    # Repeat with slight separators so there are enough batches for fixed-step training.
    return texts


def infinite_loader(dataset: Dataset, batch_size: int, pad_token_id: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    collator = CausalCollator(pad_token_id)
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collator)
        for batch in loader:
            yield batch


def train_lm_steps(
    model,
    tok,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
    steps: int,
    lr: float,
    batch_size: int,
    grad_accum_steps: int = 1,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 1.0,
    seed: int = 0,
    log_every: int = 50,
    phase_name: str = "train",
) -> list[dict[str, Any]]:
    if steps <= 0:
        return []
    model.train()
    dataset = TokenTextDataset(texts, tok, max_length=max_length)
    loader = infinite_loader(dataset, batch_size=batch_size, pad_token_id=tok.pad_token_id, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    logs: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for step in range(1, steps + 1):
        total_loss = 0.0
        for _ in range(int(grad_accum_steps)):
            batch = next(loader)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / int(grad_accum_steps)
            loss.backward()
            total_loss += float(loss.detach().cpu().item())
        if max_grad_norm is not None and float(max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running += total_loss
        if log_every and (step % int(log_every) == 0 or step == 1 or step == steps):
            avg = running / (int(log_every) if step % int(log_every) == 0 else max(1, step % int(log_every)))
            print(f"[{now()}] {phase_name} step {step}/{steps} loss={avg:.4f}", flush=True)
            logs.append({"phase": phase_name, "train_step": step, "loss": avg})
            running = 0.0
    model.eval()
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logs


@torch.no_grad()
def choice_logprob(model, tok, prompt: str, continuation: str, device: torch.device, max_length: int) -> tuple[float, int]:
    # Leading space is important for GPT-style tokenization.
    continuation = " " + continuation.strip()
    full = prompt + continuation
    prompt_ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_ids = tok(full, add_special_tokens=False, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if full_len <= prompt_len:
        return float("-inf"), 0
    input_ids = full_ids.to(device)
    logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    start = max(prompt_len - 1, 0)
    end = full_len - 1
    if end <= start:
        return float("-inf"), 0
    selected = log_probs[:, start:end, :].gather(-1, targets[:, start:end].unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().detach().cpu().item()), int(selected.numel())


@torch.no_grad()
def evaluate_probes(model, tok, probes: Sequence[ProbeItem], device: torch.device, max_length: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    item_rows: list[dict[str, Any]] = []
    correct = 0
    margins: list[float] = []
    gold_lps: list[float] = []
    for item in probes:
        lps = []
        lengths = []
        for choice in item.choices:
            lp, ntok = choice_logprob(model, tok, item.prompt, choice, device, max_length)
            lps.append(lp)
            lengths.append(ntok)
        pred = int(max(range(len(lps)), key=lambda j: lps[j]))
        gold_lp = lps[item.answer_index]
        best_wrong = max(lp for j, lp in enumerate(lps) if j != item.answer_index)
        margin = gold_lp - best_wrong
        margins.append(margin)
        gold_lps.append(gold_lp)
        correct += int(pred == item.answer_index)
        item_rows.append({
            "item_id": item.item_id,
            "fact_id": item.fact_id,
            "prompt": item.prompt,
            "gold_value": item.gold_value,
            "answer_index": item.answer_index,
            "predicted_index": pred,
            "correct": int(pred == item.answer_index),
            "gold_logprob": gold_lp,
            "best_wrong_logprob": best_wrong,
            "gold_logprob_margin": margin,
            "choice_lengths": "|".join(map(str, lengths)),
            "choices": "|".join(item.choices),
        })
    n = max(1, len(probes))
    summary = {
        "accuracy": correct / n,
        "mean_margin": float(sum(margins) / n),
        "median_margin": float(sorted(margins)[len(margins) // 2]) if margins else float("nan"),
        "mean_gold_logprob": float(sum(gold_lps) / n),
        "n_items": len(probes),
    }
    return summary, item_rows


METRICS_FIELDNAMES = [
    "experiment_id", "model", "stage", "step", "seed", "signal_type",
    "event", "metric", "value", "timestamp", "poison_budget", "degradation_steps",
]

ITEM_FIELDNAMES = [
    "experiment_id", "model", "stage", "step", "seed", "signal_type",
    "event", "timestamp", "poison_budget", "degradation_steps",
    "item_id", "fact_id", "prompt", "gold_value", "answer_index",
    "predicted_index", "correct", "gold_logprob", "best_wrong_logprob",
    "gold_logprob_margin", "choice_lengths", "choices",
]


def _fieldnames_for_path(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    # Keep E3 raw CSVs schema-stable. The previous append logic allowed later
    # degradation rows to add columns without rewriting the header, producing
    # files that pandas could not parse (e.g. 10 header fields, 12 row fields).
    name = path.name
    if name == "e3_factual_metrics.csv":
        return METRICS_FIELDNAMES
    if name == "e3_factual_item_metrics.csv":
        return ITEM_FIELDNAMES
    fieldnames: list[str] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            try:
                fieldnames.extend(next(csv.reader(f)))
            except StopIteration:
                pass
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    return fieldnames


def append_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames_for_path(path, rows)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def completed_cells(metrics_path: Path) -> set[tuple[str, str, int, str]]:
    if not metrics_path.exists():
        return set()
    out: set[tuple[str, str, int, str]] = set()
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("event") == "cell_complete":
                out.add((row.get("model", ""), row.get("stage", ""), int(row.get("seed", -1)), row.get("signal_type", "")))
    return out


def cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_cpu_state_dict(model, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict(state, strict=True)
    model.to(device)


def metric_rows_from_summary(
    cfg: dict[str, Any],
    model_name: str,
    stage: str,
    seed: int,
    event: str,
    summary: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    extra = extra or {}
    rows = []
    for metric, value in summary.items():
        rows.append({
            "experiment_id": cfg.get("experiment_id", "e3"),
            "model": model_name,
            "stage": stage,
            "step": parse_step(stage),
            "seed": seed,
            "signal_type": cfg.get("signal", {}).get("type", "factual_association"),
            "event": event,
            "metric": metric,
            "value": value,
            "timestamp": now(),
            **extra,
        })
    return rows


def add_item_context(rows: list[dict[str, Any]], cfg: dict[str, Any], model_name: str, stage: str, seed: int, event: str, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    extra = extra or {}
    out = []
    for row in rows:
        out.append({
            "experiment_id": cfg.get("experiment_id", "e3"),
            "model": model_name,
            "stage": stage,
            "step": parse_step(stage),
            "seed": seed,
            "signal_type": cfg.get("signal", {}).get("type", "factual_association"),
            "event": event,
            "timestamp": now(),
            **extra,
            **row,
        })
    return out


def run_cell(cfg: dict[str, Any], model_name: str, stage: str, seed: int, metrics_path: Path, item_path: Path) -> None:
    set_seed(seed)
    tok, model, device = load_model_and_tokenizer(model_name, stage, cfg)
    max_length = int(cfg.get("runtime", {}).get("max_length", 128))
    train_texts, probes, facts = build_factual_signal(tok, cfg, seed)
    print(f"[{now()}] Built signal seed={seed}: train_texts={len(train_texts)} probes={len(probes)}", flush=True)

    # Base evaluation.
    base_summary, base_items = evaluate_probes(model, tok, probes, device, max_length)
    append_dict_rows(metrics_path, metric_rows_from_summary(cfg, model_name, stage, seed, "base", base_summary))
    append_dict_rows(item_path, add_item_context(base_items, cfg, model_name, stage, seed, "base"))

    # Injection.
    tcfg = cfg.get("training", {})
    train_lm_steps(
        model=model,
        tok=tok,
        texts=train_texts,
        device=device,
        max_length=max_length,
        steps=int(tcfg.get("injection_steps", 200)),
        lr=float(tcfg.get("injection_lr", 2e-5)),
        batch_size=int(tcfg.get("batch_size", 8)),
        grad_accum_steps=int(tcfg.get("grad_accum_steps", 1)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        seed=seed,
        log_every=int(tcfg.get("log_every", 50)),
        phase_name=f"inject {stage} seed={seed}",
    )
    inj_summary, inj_items = evaluate_probes(model, tok, probes, device, max_length)
    append_dict_rows(metrics_path, metric_rows_from_summary(cfg, model_name, stage, seed, "post_injection", inj_summary))
    append_dict_rows(item_path, add_item_context(inj_items, cfg, model_name, stage, seed, "post_injection"))

    injected_state = cpu_state_dict(model)

    # Clean continuation branch.
    cont_cfg = cfg.get("continuation", {})
    if bool(cont_cfg.get("enabled", True)):
        load_cpu_state_dict(model, injected_state, device)
        train_lm_steps(
            model=model,
            tok=tok,
            texts=build_continuation_texts(cfg),
            device=device,
            max_length=max_length,
            steps=int(cont_cfg.get("steps", 200)),
            lr=float(cont_cfg.get("lr", 1e-5)),
            batch_size=int(cont_cfg.get("batch_size", tcfg.get("batch_size", 8))),
            grad_accum_steps=1,
            weight_decay=0.0,
            max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
            seed=seed + 10_000,
            log_every=int(tcfg.get("log_every", 50)),
            phase_name=f"continuation {stage} seed={seed}",
        )
        cont_summary, cont_items = evaluate_probes(model, tok, probes, device, max_length)
        append_dict_rows(metrics_path, metric_rows_from_summary(cfg, model_name, stage, seed, "post_continuation", cont_summary))
        append_dict_rows(item_path, add_item_context(cont_items, cfg, model_name, stage, seed, "post_continuation"))

    # Degradation branches.
    deg_cfg = cfg.get("degradation", {})
    if bool(deg_cfg.get("enabled", True)):
        for budget in list(deg_cfg.get("budgets", [4, 16, 64, 256])):
            budget_int = int(budget)
            load_cpu_state_dict(model, injected_state, device)
            conflicting = build_conflicting_texts(facts, tok, cfg, seed, budget_int)
            deg_steps = int(math.ceil(len(conflicting) / max(1, int(deg_cfg.get("batch_size", tcfg.get("batch_size", 8)))))) * int(deg_cfg.get("epochs", 1))
            deg_steps = max(1, deg_steps)
            train_lm_steps(
                model=model,
                tok=tok,
                texts=conflicting,
                device=device,
                max_length=max_length,
                steps=deg_steps,
                lr=float(deg_cfg.get("lr", tcfg.get("injection_lr", 2e-5))),
                batch_size=int(deg_cfg.get("batch_size", tcfg.get("batch_size", 8))),
                grad_accum_steps=1,
                weight_decay=0.0,
                max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
                seed=seed + 20_000 + budget_int,
                log_every=0,
                phase_name=f"degrade{k if False else ''}",
            )
            deg_summary, deg_items = evaluate_probes(model, tok, probes, device, max_length)
            extra = {"poison_budget": budget_int, "degradation_steps": deg_steps}
            append_dict_rows(metrics_path, metric_rows_from_summary(cfg, model_name, stage, seed, f"post_degradation_k{budget_int}", deg_summary, extra=extra))
            append_dict_rows(item_path, add_item_context(deg_items, cfg, model_name, stage, seed, f"post_degradation_k{budget_int}", extra=extra))

    append_dict_rows(metrics_path, [{
        "experiment_id": cfg.get("experiment_id", "e3"),
        "model": model_name,
        "stage": stage,
        "step": parse_step(stage),
        "seed": seed,
        "signal_type": cfg.get("signal", {}).get("type", "factual_association"),
        "event": "cell_complete",
        "metric": "cell_complete",
        "value": 1.0,
        "timestamp": now(),
    }])

    del injected_state, model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E3 factual-association intervention MVP.")
    parser.add_argument("--config", required=True, help="Path to E3 factual YAML config.")
    parser.add_argument("--force", action="store_true", help="Run cells even if a cell_complete marker exists.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    outputs = cfg.get("outputs", {})
    root = Path(outputs.get("root", "results/e3_critical_period_intervention"))
    dirs = ensure_experiment_dirs(root)
    metrics_path = root / outputs.get("metrics_csv", "raw/e3_factual_metrics.csv")
    item_path = root / outputs.get("item_metrics_csv", "raw/e3_factual_item_metrics.csv")

    model_name = cfg["model"]
    stages = list(cfg["stages"])
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    done = completed_cells(metrics_path)

    for stage in stages:
        for seed in seeds:
            key = (model_name, stage, seed, cfg.get("signal", {}).get("type", "factual_association"))
            if key in done and not args.force:
                print(f"[{now()}] Skipping completed cell {key}", flush=True)
                continue
            print(f"[{now()}] Starting cell model={model_name} stage={stage} seed={seed}", flush=True)
            run_cell(cfg, model_name, stage, seed, metrics_path, item_path)

    # Register artifacts if available.
    if ArtifactRegistry is not None and ArtifactRecord is not None:
        manifest_path = root / outputs.get("manifest_csv", "manifests/artifact_manifest.csv")
        registry = ArtifactRegistry(manifest_path)
        registry.append(ArtifactRecord(
            experiment_id=cfg.get("experiment_id", "e3"),
            artifact_type="raw_csv",
            path=metrics_path,
            thesis_section=cfg.get("thesis_section", ""),
            caption_draft="Raw E3 factual-intervention summary metrics by stage, seed, and event.",
            source_data="Synthetic factual associations and Hugging Face Pythia checkpoints",
            code_entrypoint="scripts/run_e3_factual_intervention.py",
            status="draft",
            notes=f"Config: {args.config}",
        ))
        registry.append(ArtifactRecord(
            experiment_id=cfg.get("experiment_id", "e3"),
            artifact_type="raw_csv",
            path=item_path,
            thesis_section=cfg.get("thesis_section", ""),
            caption_draft="Item-level E3 factual-intervention probe metrics.",
            source_data=str(metrics_path),
            code_entrypoint="scripts/run_e3_factual_intervention.py",
            status="draft",
            notes=f"Config: {args.config}",
        ))
    print(f"[{now()}] Done. Wrote summary metrics to {metrics_path}", flush=True)
    print(f"[{now()}] Wrote item metrics to {item_path}", flush=True)


if __name__ == "__main__":
    main()
