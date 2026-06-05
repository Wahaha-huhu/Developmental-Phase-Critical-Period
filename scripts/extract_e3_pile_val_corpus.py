from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="EleutherAI/pile_val_test")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--target-sequences", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer-multiplier", type=int, default=4)
    ap.add_argument("--output", default="data/e3_continuation/fixed_pile_val_seed0.jsonl")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    target_buffer = args.target_sequences * args.buffer_multiplier
    chunks = []

    for ex in ds:
        text = " ".join(str(ex.get(args.text_key, "")).split())
        if not text:
            continue

        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) < args.min_tokens:
            continue

        for start in range(0, len(ids), args.max_tokens):
            chunk_ids = ids[start:start + args.max_tokens]
            if len(chunk_ids) < args.min_tokens:
                continue
            chunks.append(tok.decode(chunk_ids))
            if len(chunks) >= target_buffer:
                break

        if len(chunks) >= target_buffer:
            break

    rng.shuffle(chunks)
    chunks = chunks[:args.target_sequences]

    with out.open("w", encoding="utf-8") as f:
        for i, txt in enumerate(chunks):
            f.write(json.dumps({"id": i, "text": txt}, ensure_ascii=False) + "\n")

    meta = vars(args)
    meta["actual_sequences"] = len(chunks)
    meta["output"] = str(out)
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("Wrote:", out)
    print("Sequences:", len(chunks))
    print("Meta:", out.with_suffix(out.suffix + ".meta.json"))


if __name__ == "__main__":
    main()
