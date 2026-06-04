from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


class CausalLMDataset(Dataset):
    def __init__(self, examples: List[EncodedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.examples[idx]


def encode_pair(tok, prompt: str, response: str, mask_prompt: bool, max_len: int = 256) -> EncodedExample:
    p = tok(prompt, add_special_tokens=False).input_ids
    r = tok(response, add_special_tokens=False).input_ids
    ids = (p + r)[:max_len]
    if mask_prompt:
        labels = ([-100] * len(p) + r)[:max_len]
    else:
        labels = ids.copy()
    return EncodedExample(ids, labels)


def encode_text(tok, text: str, max_len: int = 256) -> EncodedExample:
    ids = tok(text, add_special_tokens=False).input_ids[:max_len]
    return EncodedExample(ids, ids.copy())


def build_examples(tok, train_texts: Iterable[str], train_pairs: Iterable[Tuple[str, str]], mask_prompt: bool, max_len: int = 256) -> list[EncodedExample]:
    examples = [encode_text(tok, t, max_len=max_len) for t in train_texts]
    examples.extend(encode_pair(tok, p, r, mask_prompt=mask_prompt, max_len=max_len) for p, r in train_pairs)
    return examples


def make_collator(tok):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch: list[EncodedExample]) -> dict[str, torch.Tensor]:
        max_len = max(len(x.input_ids) for x in batch)
        input_ids, labels, attention = [], [], []
        for ex in batch:
            pad = max_len - len(ex.input_ids)
            input_ids.append(ex.input_ids + [pad_id] * pad)
            labels.append(ex.labels + [-100] * pad)
            attention.append([1] * len(ex.input_ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


def make_infinite_loader(dataset: Dataset, tok, batch_size: int, seed: int, num_workers: int = 0):
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        num_workers=num_workers,
        collate_fn=make_collator(tok),
        drop_last=False,
    )
    return itertools.cycle(loader)


def finetune_constant_lr(
    model,
    loader,
    steps: int,
    lr: float,
    device: str = "cuda",
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    log_every: int = 25,
) -> list[dict]:
    """Full-parameter AdamW fine-tuning with constant LR and no scheduler."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logs = []
    for step in range(1, steps + 1):
        batch = next(loader)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 1 or step % log_every == 0 or step == steps:
            logs.append({"step": step, "loss": float(out.loss.detach().cpu().item())})
    model.eval()
    return logs
