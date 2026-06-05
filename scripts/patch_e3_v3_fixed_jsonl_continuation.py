#!/usr/bin/env python3
"""Patch run_e3_factual_cell_v3.py to support fixed_jsonl continuation corpora.

This adds support for configs like:

retention:
  continuation_corpus: fixed_jsonl
  continuation_jsonl: data/e3_continuation/fixed_pile_val_seed0.jsonl
  continuation_corpus_size: 12000   # optional cap
  continuation_seed: 0              # deterministic dataloader shuffle seed

The patch is text-based and idempotent.
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path("scripts/run_e3_factual_cell_v3.py")

FUNC = r'''

def fixed_jsonl_continuation_corpus(path: str, limit: Optional[int] = None) -> List[str]:
    """Load a frozen JSONL continuation corpus with a `text` field.

    The corpus should be created once and reused unchanged across all E3 stages.
    Each line should be JSON like {"id": 0, "text": "..."}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Continuation JSONL not found: {p}")
    out: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {p}: {e}") from e
            txt = str(obj.get("text", "")).strip()
            if txt:
                out.append(txt)
            if limit is not None and len(out) >= int(limit):
                break
    if not out:
        raise ValueError(f"No usable text entries found in continuation JSONL: {p}")
    return out
'''

OLD_BLOCK = '''        corpus_kind = ret.get("continuation_corpus", "synthetic_generic")
        if corpus_kind != "synthetic_generic":
            raise NotImplementedError("Only synthetic_generic continuation corpus is implemented in v3 standalone runner")
        cont_texts = synthetic_continuation_corpus(int(ret.get("continuation_corpus_size", 1000)), seed=seed)
        print(f"[{now()}] Clean-continuation branch: {ret['continuation_steps']} steps", flush=True)
        cont_info = finetune_texts(
            model,
            tok,
            cont_texts,
            steps=int(ret["continuation_steps"]),
            epochs=None,
            lr=float(ret["continuation_lr"]),
            batch_size=int(ret.get("continuation_batch_size", tr.get("batch_size", 16))),
            max_len=int(tr.get("max_len", 128)),
            weight_decay=float(tr.get("weight_decay", 0.0)),
            grad_clip_norm=float(tr.get("grad_clip_norm", 0.0)) if tr.get("grad_clip_norm", 0.0) else None,
            seed=seed + 1000,
            device=device,
        )'''

NEW_BLOCK = '''        corpus_kind = ret.get("continuation_corpus", "synthetic_generic")
        corpus_limit = ret.get("continuation_corpus_size", None)
        if corpus_kind == "synthetic_generic":
            cont_texts = synthetic_continuation_corpus(int(corpus_limit or 1000), seed=seed)
        elif corpus_kind in {"fixed_jsonl", "local_jsonl", "pile_jsonl"}:
            jsonl_path = ret.get("continuation_jsonl") or ret.get("corpus_path") or ret.get("path")
            if not jsonl_path:
                raise ValueError("retention.continuation_jsonl / corpus_path must be set for fixed_jsonl continuation")
            cont_texts = fixed_jsonl_continuation_corpus(str(jsonl_path), limit=int(corpus_limit) if corpus_limit else None)
        else:
            raise ValueError(f"Unknown continuation_corpus kind: {corpus_kind!r}")
        print(f"[{now()}] Clean-continuation branch: {ret['continuation_steps']} steps on {corpus_kind} ({len(cont_texts)} texts)", flush=True)
        cont_info = finetune_texts(
            model,
            tok,
            cont_texts,
            steps=int(ret["continuation_steps"]),
            epochs=None,
            lr=float(ret["continuation_lr"]),
            batch_size=int(ret.get("continuation_batch_size", tr.get("batch_size", 16))),
            max_len=int(tr.get("max_len", 128)),
            weight_decay=float(tr.get("weight_decay", 0.0)),
            grad_clip_norm=float(tr.get("grad_clip_norm", 0.0)) if ret.get("grad_clip_norm", tr.get("grad_clip_norm", 0.0)) else None,
            seed=int(ret.get("continuation_seed", 0)),
            device=device,
        )'''


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Run from project root; cannot find {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    original = text

    if "def fixed_jsonl_continuation_corpus" not in text:
        anchor = "def dtype_from_config"
        if anchor not in text:
            raise RuntimeError("Could not find insertion anchor `def dtype_from_config`")
        text = text.replace("\ndef dtype_from_config", FUNC + "\ndef dtype_from_config", 1)

    if OLD_BLOCK in text:
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    elif "Only synthetic_generic continuation corpus is implemented in v3 standalone runner" in text:
        raise RuntimeError(
            "Found the NotImplementedError but the surrounding block did not match. "
            "Please inspect lines around `continuation_corpus` manually."
        )

    if text != original:
        backup = TARGET.with_suffix(TARGET.suffix + ".bak_fixed_jsonl")
        backup.write_text(original, encoding="utf-8")
        TARGET.write_text(text, encoding="utf-8")
        print(f"Patched {TARGET}; backup written to {backup}")
    else:
        print(f"No changes needed; {TARGET} already appears patched")


if __name__ == "__main__":
    main()
