#!/usr/bin/env python3
"""Generate a 1.4B E1 config with outputs under solid_results/.

This copies a working E1 template and changes only model/checkpoints/output root.
Use this if you want to collect 1.4B E1 metrics before plotting.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import yaml

EARLY_DENSE_SPARSE_LATE = [
    "step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128", "step256", "step512",
    "step1000", "step2000", "step3000", "step4000", "step5000", "step6000", "step7000", "step8000", "step9000",
    "step10000", "step13000", "step16000", "step23000", "step32000", "step48000", "step64000", "step80000", "step100000", "step120000", "step143000",
]

FULL_EARLY = [
    "step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128", "step256", "step512",
    "step1000", "step2000", "step3000", "step4000", "step5000", "step6000", "step7000", "step8000", "step9000",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="configs/e1_phase_identification.yaml")
    ap.add_argument("--out", default="solid_results/configs/e1_dense_1p4b_early_sparse_late.yaml")
    ap.add_argument("--output-root", default="solid_results/e1_dense_1p4b_early_sparse_late")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b-deduped")
    ap.add_argument("--mode", choices=["early", "early_sparse_late"], default="early_sparse_late")
    args = ap.parse_args()

    template_path = Path(args.template)
    cfg = yaml.safe_load(template_path.read_text())
    cfg = copy.deepcopy(cfg)

    cfg["models"] = [args.model]
    cfg["model_name"] = args.model
    ckpts = FULL_EARLY if args.mode == "early" else EARLY_DENSE_SPARSE_LATE
    cfg["checkpoints"] = ckpts
    cfg["stages"] = ckpts
    cfg.setdefault("outputs", {})
    cfg["outputs"]["root"] = args.output_root
    cfg["output_root"] = args.output_root

    # Hint cheaper metrics for large model if runner honors these.
    cfg.setdefault("metrics", {})
    if isinstance(cfg["metrics"], dict):
        for k in ["frobenius_norm", "spectral_norm", "stable_rank", "effective_rank"]:
            cfg["metrics"][k] = True
        for k in ["mp_outliers", "alpha", "subspace_stability"]:
            cfg["metrics"][k] = cfg["metrics"].get(k, False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", out)
    print("model", args.model)
    print("output_root", args.output_root)
    print("n_checkpoints", len(ckpts))

if __name__ == "__main__":
    main()
