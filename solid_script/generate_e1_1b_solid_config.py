#!/usr/bin/env python3
"""Generate a solid-core E1 collection config for Pythia-1B.

This intentionally writes configs under solid_results/configs and outputs under
solid_results/, leaving the exploratory results/ tree untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

EARLY_DENSE_SPARSE_LATE = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64",
    "step128", "step256", "step512", "step1000", "step2000", "step3000", "step4000",
    "step5000", "step6000", "step7000", "step8000", "step9000", "step10000",
    "step13000", "step16000", "step23000", "step32000", "step48000", "step64000",
    "step80000", "step100000", "step120000", "step143000",
]

EARLY_ONLY = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64",
    "step128", "step256", "step512", "step1000", "step2000", "step3000", "step4000",
    "step5000", "step6000", "step7000", "step8000", "step9000", "step10000",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        obj = yaml.safe_load(f)
    return obj or {}


def set_model(cfg: dict[str, Any], model_name: str) -> None:
    # Support several schemas used in the repo.
    cfg["model_name"] = model_name
    cfg["models"] = [model_name]
    if isinstance(cfg.get("model"), dict):
        cfg["model"]["name"] = model_name
    else:
        cfg["model"] = {"name": model_name}


def set_output_root(cfg: dict[str, Any], root: str) -> None:
    cfg["output_root"] = root
    cfg.setdefault("outputs", {})
    if not isinstance(cfg["outputs"], dict):
        cfg["outputs"] = {}
    cfg["outputs"]["root"] = root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="configs/e1_phase_identification.yaml")
    ap.add_argument("--out", default="solid_results/configs/e1_dense_1b_early_sparse_late.yaml")
    ap.add_argument("--output-root", default="solid_results/e1_dense_1b_early_sparse_late")
    ap.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    ap.add_argument("--mode", choices=["early_sparse_late", "early_only"], default="early_sparse_late")
    ap.add_argument("--cheap-metrics", action="store_true", help="Request cheap metrics if the collector supports metric flags.")
    ap.add_argument("--drop-step0", action="store_true", help="Omit step0 from the generated checkpoint grid.")
    args = ap.parse_args()

    template_path = Path(args.template)
    cfg = load_yaml(template_path)
    if not cfg:
        raise SystemExit(f"Template config is empty or missing: {template_path}")

    checkpoints = list(EARLY_DENSE_SPARSE_LATE if args.mode == "early_sparse_late" else EARLY_ONLY)
    if args.drop_step0:
        checkpoints = [c for c in checkpoints if c != "step0"]

    cfg = copy.deepcopy(cfg)
    set_model(cfg, args.model)
    set_output_root(cfg, args.output_root)
    cfg["checkpoints"] = checkpoints
    cfg["stages"] = checkpoints

    # Preserve the same modules as the existing E1 config if present. Otherwise use standard Pythia modules.
    cfg.setdefault(
        "module_suffixes",
        [
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
    )

    if args.cheap_metrics:
        cfg["metrics"] = {
            "frobenius_norm": True,
            "spectral_norm": True,
            "stable_rank": True,
            "effective_rank": True,
            "mp_outliers": False,
            "alpha": False,
            "subspace_stability": False,
        }
        cfg["power_iters"] = int(cfg.get("power_iters", 8))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))

    audit = {
        "template": str(template_path),
        "out": str(out),
        "model": args.model,
        "output_root": args.output_root,
        "mode": args.mode,
        "n_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }
    audit_path = out.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2))

    print(f"Wrote {out}")
    print(f"Wrote {audit_path}")
    print(f"model: {args.model}")
    print(f"output_root: {args.output_root}")
    print(f"n_checkpoints: {len(checkpoints)}")
    print(f"first: {checkpoints[:8]}")
    print(f"last: {checkpoints[-8:]}")


if __name__ == "__main__":
    main()
