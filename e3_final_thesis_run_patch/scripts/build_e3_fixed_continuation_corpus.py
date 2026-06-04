#!/usr/bin/env python3
"""Build a frozen continuation corpus for E3 clean-retention branches.

Preferred final-thesis use:
  1. Provide a local JSONL/TXT export of held-out Pile / validation text.
  2. This script samples/chunks it once with a fixed seed.
  3. Every E3 cell then trains on the exact same output JSONL.

Fallback modes exist for debugging, but the final thesis run should use a fixed
real-text corpus if possible.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
from typing import Iterable, List


def iter_local_jsonl(path: Path, text_key: str = "text") -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            txt = obj.get(text_key, "")
            if isinstance(txt, str) and txt.strip():
                yield txt.strip()


def iter_local_txt(path: Path) -> Iterable[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # paragraph chunks
    for block in text.split("\n\n"):
        block = " ".join(block.split())
        if len(block) > 80:
            yield block


def iter_hf_dataset(dataset_name: str, split: str, text_key: str, streaming: bool, limit_docs: int | None) -> Iterable[str]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets is required for --source hf_dataset: pip install datasets") from e
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    n = 0
    for row in ds:
        txt = row.get(text_key, "")
        if isinstance(txt, str) and txt.strip():
            yield txt.strip()
            n += 1
            if limit_docs is not None and n >= limit_docs:
                break


def iter_synthetic_generic(n: int) -> Iterable[str]:
    # Debug fallback only; not recommended for the final thesis run.
    templates = [
        "This is a general passage about language, history, science, and ordinary events. The text contains no fictional facts from the injection signal.",
        "A researcher writes a short neutral paragraph to continue language-model training without rehearsing the injected associations.",
        "The document describes common objects, simple explanations, and background information in a generic style.",
        "Several sentences discuss cities, rivers, books, weather, and everyday observations without using synthetic entity names.",
    ]
    for i in range(n):
        yield templates[i % len(templates)] + f" Passage id {i}."


def chunk_by_tokens(texts: Iterable[str], tokenizer_name: str, max_tokens: int, min_tokens: int, target_sequences: int) -> List[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out: List[str] = []
    for text in texts:
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) < min_tokens:
            continue
        for start in range(0, len(ids), max_tokens):
            chunk = ids[start:start + max_tokens]
            if len(chunk) < min_tokens:
                continue
            out.append(tok.decode(chunk))
            if len(out) >= target_sequences:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local_jsonl", "local_txt", "hf_dataset", "synthetic_generic"], required=True)
    ap.add_argument("--input", type=str, default=None, help="Local JSONL/TXT path for local sources")
    ap.add_argument("--dataset-name", type=str, default=None, help="HF dataset name for hf_dataset source")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--text-key", type=str, default="text")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--target-sequences", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="data/e3_continuation/fixed_continuation_seed0.jsonl")
    args = ap.parse_args()

    if args.source == "local_jsonl":
        if not args.input:
            raise SystemExit("--input is required for local_jsonl")
        texts = list(iter_local_jsonl(Path(args.input), args.text_key))
    elif args.source == "local_txt":
        if not args.input:
            raise SystemExit("--input is required for local_txt")
        texts = list(iter_local_txt(Path(args.input)))
    elif args.source == "hf_dataset":
        if not args.dataset_name:
            raise SystemExit("--dataset-name is required for hf_dataset")
        texts = list(iter_hf_dataset(args.dataset_name, args.split, args.text_key, args.streaming, args.limit_docs))
    else:
        texts = list(iter_synthetic_generic(max(args.target_sequences * 2, 20000)))

    rng = random.Random(args.seed)
    rng.shuffle(texts)
    chunks = chunk_by_tokens(texts, args.tokenizer, args.max_tokens, args.min_tokens, args.target_sequences)
    if len(chunks) < args.target_sequences:
        print(f"WARNING: requested {args.target_sequences} sequences but built {len(chunks)}", file=sys.stderr)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, text in enumerate(chunks):
            f.write(json.dumps({"id": i, "text": text}, ensure_ascii=False) + "\n")
    meta = out.with_suffix(out.suffix + ".meta.json")
    meta.write_text(json.dumps(vars(args) | {"n_sequences": len(chunks)}, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} continuation sequences to {out}")
    print(f"Wrote metadata to {meta}")


if __name__ == "__main__":
    main()
