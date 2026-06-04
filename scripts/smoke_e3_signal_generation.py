#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from critical_periods.signals import SignalSpec, make_signal


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and inspect E3 synthetic injection signals.")
    ap.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--signal", default="factual", choices=["factual", "procedural", "alignment"])
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--n-probe", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/e3_critical_period_intervention/debug/signal_preview.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    spec = SignalSpec(name=args.signal, n_items=args.n_items, n_probe=args.n_probe, seed=args.seed)
    data = make_signal(spec, tok)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": spec.__dict__,
        "meta": data.meta,
        "n_train_texts": len(data.train_texts),
        "n_train_pairs": len(data.train_pairs),
        "n_probes": len(data.probes),
        "n_controls": len(data.controls),
        "train_texts_preview": data.train_texts[:5],
        "train_pairs_preview": data.train_pairs[:5],
        "probes_preview": data.probes[:5],
        "controls_preview": data.controls[:5],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote preview to {out}")


if __name__ == "__main__":
    main()
