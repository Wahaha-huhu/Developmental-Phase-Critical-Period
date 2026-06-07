#!/usr/bin/env python3
"""Generate the Step-1 dense factual durability sweep config.

This is a schema-tolerant config generator: it copies an existing working E3
factual config when provided, then overrides only the fields needed for the
revised solid-core Step 1 sweep. If no template is provided/found, it writes a
minimal config with redundant aliases used by our earlier E3 runners.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

EARLY_20 = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32",
    "step64", "step128", "step256", "step512",
    "step1000", "step2000", "step3000", "step4000", "step5000",
    "step6000", "step7000", "step8000", "step9000",
]

# 12 later checkpoints, intentionally sparse after 10k.
LATE_12 = [
    "step10000", "step13000", "step16000", "step23000", "step32000",
    "step44000", "step64000", "step89000", "step100000", "step110000",
    "step128000", "step143000",
]

DEFAULT_CANDIDATES = [
    "configs/e3_critical_period_intervention_final_v4.yaml",
    "configs/e3_factual_final_v4.yaml",
    "configs/e3_factual_160m_final_v4.yaml",
    "configs/e3_critical_period_intervention_160m_final_v4.yaml",
    "configs/e3_critical_period_intervention_final_calibration_v4.yaml",
]


def load_template(path: str | None) -> dict[str, Any]:
    candidates = [path] if path else DEFAULT_CANDIDATES
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.exists():
            print(f"Using template config: {p}")
            data = yaml.safe_load(p.read_text())
            if not isinstance(data, dict):
                raise ValueError(f"Template {p} did not parse as a mapping")
            return copy.deepcopy(data)
    print("No template config found; writing minimal schema-tolerant config")
    return {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=None, help="Existing working E3 factual config to copy")
    ap.add_argument("--out", default="configs/step1_factual_dense_160m_32ckpt.yaml")
    ap.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    ap.add_argument("--output-root", default="results/step1_factual_dense_160m_32ckpt")
    ap.add_argument("--continuation-corpus", default="data/e3_continuation/fixed_pile_val_seed0.jsonl")
    ap.add_argument("--write-checkpoint-list", default="configs/step1_dense_32_checkpoints.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_template(args.template)
    checkpoints = EARLY_20 + LATE_12
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    # Redundant aliases across the codebase's historical configs/runners.
    cfg["model"] = {"name": args.model}
    cfg["models"] = [args.model]
    cfg["checkpoints"] = checkpoints
    cfg["stages"] = checkpoints
    cfg["revisions"] = checkpoints
    cfg["seeds"] = seeds

    cfg.setdefault("outputs", {})
    cfg["outputs"]["root"] = args.output_root
    cfg["output_root"] = args.output_root

    # Preserve existing values if template has them; otherwise write safe defaults.
    cfg.setdefault("signal", {})
    cfg["signal"].setdefault("type", "synthetic_factual_associations")
    cfg["signal"].setdefault("n_facts", 300)
    cfg["signal"].setdefault("train_templates_per_fact", 3)
    cfg["signal"].setdefault("heldout_probe_template", True)
    cfg["signal"].setdefault("control_disjoint", True)

    cfg.setdefault("injection", {})
    cfg["injection"].setdefault("lr", 1.0e-5)
    cfg["injection"].setdefault("fixed_lr_across_stages", True)

    cfg.setdefault("continuation", {})
    cfg["continuation"].setdefault("corpus_path", args.continuation_corpus)
    cfg["continuation"].setdefault("fixed_budget_across_stages", True)
    cfg["continuation"].setdefault("mode", "short_horizon_fixed_budget")

    cfg.setdefault("attacks", {})
    cfg["attacks"].setdefault("poison_budgets", [4, 16, 64, 256])
    cfg["attacks"].setdefault("restart_each_budget_from_post_injection", True)

    cfg["design"] = {
        "step": "Step 1 — dense short-horizon factual durability sweep",
        "purpose": "artifact-protected behavioural foundation",
        "checkpoint_count": len(checkpoints),
        "early_dense_count": len(EARLY_20),
        "late_sparse_count": len(LATE_12),
        "controls": [
            "fixed injection LR across stages",
            "fixed short continuation budget across stages",
            "uptake-covariate analysis",
            "segmented-vs-monotone comparison",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))

    ckpt_path = Path(args.write_checkpoint_list)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps({"checkpoints": checkpoints}, indent=2))

    print(f"Wrote {out}")
    print(f"Wrote checkpoint list {ckpt_path}")
    print(f"Output root: {args.output_root}")
    print(f"Checkpoints ({len(checkpoints)}): {', '.join(checkpoints)}")


if __name__ == "__main__":
    main()
