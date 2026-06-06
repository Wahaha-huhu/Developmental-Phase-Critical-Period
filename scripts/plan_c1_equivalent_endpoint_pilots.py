#!/usr/bin/env python3
"""Generate C1 equivalent-endpoint pilot configs.

Purpose:
  Create small continue-to-consolidation pilots where an in-window injection
  at step1000 is continued only to an intermediate endpoint, then compared
  against a post-hoc injection at the same endpoint.

Example:
  endpoint step3000:
    arm A: load step1000, inject, continue token-equivalent step1000->step3000
    arm B: load step3000, inject, no continuation

The script is schema-tolerant: it preserves the template config and only
updates top-level endpoint_step, arms, outputs.root, and common continuation
budget fields when present/needed.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import yaml


def step_num(x: str | int) -> int:
    if isinstance(x, int):
        return x
    if x is None:
        raise ValueError("Missing step")
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"Cannot parse step from {x!r}")
    return int(m.group(1))


def step_str(n: int) -> str:
    return f"step{int(n)}"


def set_nested(cfg: dict[str, Any], path: list[str], value: Any) -> None:
    cur = cfg
    for k in path[:-1]:
        if k not in cur or cur[k] is None or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="configs/c1_alignment_160m_pilot.yaml")
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--result-root-prefix", default="results/c1_alignment_160m_equiv")
    ap.add_argument("--inject-step", default="step1000")
    ap.add_argument("--endpoints", nargs="+", default=["step2000", "step3000", "step4000", "step8000"])
    ap.add_argument("--budget-scale", type=float, default=0.01)
    ap.add_argument("--pythia-tokens-per-step", type=int, default=2_097_152)
    ap.add_argument("--sequence-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    template_path = Path(args.template)
    cfg0 = yaml.safe_load(template_path.read_text())
    if cfg0 is None:
        raise RuntimeError(f"Template loaded as None: {template_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inj_n = step_num(args.inject_step)
    written = []
    for ep in args.endpoints:
        ep_n = step_num(ep)
        if ep_n < inj_n:
            raise ValueError(f"Endpoint {ep} is before injection step {args.inject_step}")

        cfg = copy.deepcopy(cfg0)
        ep_s = step_str(ep_n)
        inj_s = step_str(inj_n)
        scale_tag = str(args.budget_scale).replace(".", "p")
        cfg_name = f"c1_alignment_160m_equiv_{inj_s}_to_{ep_s}_scale{scale_tag}.yaml"
        result_root = f"{args.result_root_prefix}_{inj_s}_to_{ep_s}_scale{scale_tag}"

        cfg["endpoint_step"] = ep_s
        cfg["seeds"] = args.seeds
        cfg["arms"] = [
            {
                "name": "in_window_carried",
                "inject_step": inj_s,
                "endpoint_step": ep_s,
                "continue_to_step": ep_s,
            },
            {
                "name": "post_hoc_endpoint",
                "inject_step": ep_s,
                "endpoint_step": ep_s,
                "continue_to_step": ep_s,
            },
        ]

        set_nested(cfg, ["outputs", "root"], result_root)
        set_nested(cfg, ["continuation", "pythia_tokens_per_step"], args.pythia_tokens_per_step)
        set_nested(cfg, ["continuation", "budget_scale"], args.budget_scale)
        set_nested(cfg, ["continuation", "sequence_length"], args.sequence_length)
        set_nested(cfg, ["continuation", "batch_size"], args.batch_size)
        set_nested(cfg, ["continuation", "gradient_accumulation_steps"], args.gradient_accumulation_steps)

        # Also set common alternative keys used by earlier scripts, harmless if ignored.
        set_nested(cfg, ["retention", "budget_scale"], args.budget_scale)
        set_nested(cfg, ["training", "sequence_length"], args.sequence_length)
        set_nested(cfg, ["training", "batch_size"], args.batch_size)
        set_nested(cfg, ["training", "gradient_accumulation_steps"], args.gradient_accumulation_steps)

        out_path = out_dir / cfg_name
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        written.append((out_path, result_root, ep_n - inj_n))

    print("Generated equivalent-endpoint pilot configs:")
    local_tokens = args.sequence_length * args.batch_size * args.gradient_accumulation_steps
    for p, root, delta_steps in written:
        target_tokens = delta_steps * args.pythia_tokens_per_step * args.budget_scale
        local_steps = int((target_tokens + local_tokens - 1) // local_tokens) if local_tokens else 0
        print(f"  {p}")
        print(f"    root: {root}")
        print(f"    pythia interval steps: {delta_steps}")
        print(f"    scaled target tokens: {target_tokens:,.0f}")
        print(f"    local tokens/update: {local_tokens:,.0f}")
        print(f"    approx local updates: {local_steps:,}")


if __name__ == "__main__":
    main()
