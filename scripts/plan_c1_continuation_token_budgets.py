#!/usr/bin/env python3
"""Resolve C1 continuation budgets by token distance to an endpoint.

This preprocessor reads a C1 alignment-consolidation YAML config and writes:
  1. a resolved YAML with per-arm continuation_steps / target_tokens
  2. a CSV budget table for audit/pre-registration

It is intentionally schema-tolerant: it supports arms as a list of dicts or a
mapping of name -> dict, and it accepts step fields as either integers or
strings like "step1000".
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PYTHIA_TOKENS_PER_STEP = 2_097_152  # 2M tokens/step; override in config for exact setting.


def parse_step(x: Any) -> int:
    if x is None:
        raise ValueError("Missing step value")
    if isinstance(x, int):
        return x
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"Cannot parse step from {x!r}")
    return int(m.group(1))


def get_nested(d: dict, paths: list[list[str]], default: Any = None) -> Any:
    for path in paths:
        cur: Any = d
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def set_nested(d: dict, path: list[str], value: Any) -> None:
    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def normalise_arms(cfg: dict) -> list[tuple[str, dict]]:
    arms = cfg.get("arms") or cfg.get("experiment", {}).get("arms")
    if arms is None:
        raise ValueError("Could not find arms in config. Expected top-level `arms:` or `experiment.arms:`")
    if isinstance(arms, dict):
        return [(name, arm if isinstance(arm, dict) else {}) for name, arm in arms.items()]
    if isinstance(arms, list):
        out = []
        for i, arm in enumerate(arms):
            if not isinstance(arm, dict):
                raise ValueError(f"Arm {i} is not a dict: {arm!r}")
            name = arm.get("name", f"arm_{i}")
            out.append((name, arm))
        return out
    raise ValueError(f"Unsupported arms schema: {type(arms)}")


def write_arms(cfg: dict, arms: list[tuple[str, dict]]) -> None:
    # Preserve original arms container type when possible.
    if isinstance(cfg.get("arms"), dict):
        cfg["arms"] = {name: arm for name, arm in arms}
    elif isinstance(cfg.get("arms"), list):
        cfg["arms"] = [dict({"name": name}, **arm) if "name" not in arm else arm for name, arm in arms]
    elif isinstance(cfg.get("experiment", {}).get("arms"), dict):
        cfg["experiment"]["arms"] = {name: arm for name, arm in arms}
    elif isinstance(cfg.get("experiment", {}).get("arms"), list):
        cfg["experiment"]["arms"] = [dict({"name": name}, **arm) if "name" not in arm else arm for name, arm in arms]
    else:
        cfg["arms"] = [dict({"name": name}, **arm) for name, arm in arms]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Input C1 YAML config")
    ap.add_argument("--out-config", required=True, help="Resolved output YAML config")
    ap.add_argument("--out-csv", required=True, help="Audit CSV of token budgets")
    ap.add_argument("--endpoint-step", type=int, default=None, help="Override endpoint global step")
    ap.add_argument("--pythia-tokens-per-step", type=int, default=None, help="Override Pythia token budget per pretraining step")
    ap.add_argument("--local-batch-size", type=int, default=None, help="Override local micro/global batch size in sequences")
    ap.add_argument("--sequence-length", type=int, default=None, help="Override LM sequence length")
    ap.add_argument("--grad-accum-steps", type=int, default=None, help="Override gradient accumulation steps")
    ap.add_argument("--min-continuation-steps", type=int, default=0, help="Optional minimum local continuation steps for non-posthoc arms")
    args = ap.parse_args()

    in_path = Path(args.config)
    cfg = yaml.safe_load(in_path.read_text())
    if cfg is None:
        raise RuntimeError(f"Config loaded as None: {in_path}")
    resolved = copy.deepcopy(cfg)

    endpoint_step = args.endpoint_step
    if endpoint_step is None:
        endpoint_step = get_nested(cfg, [
            ["endpoint_step"],
            ["endpoint", "step"],
            ["continuation", "endpoint_step"],
            ["consolidation", "endpoint_step"],
        ], None)
        endpoint_step = parse_step(endpoint_step)

    pythia_tokens_per_step = args.pythia_tokens_per_step or get_nested(cfg, [
        ["continuation", "pythia_tokens_per_step"],
        ["pythia", "tokens_per_step"],
        ["training", "pythia_tokens_per_step"],
    ], DEFAULT_PYTHIA_TOKENS_PER_STEP)
    pythia_tokens_per_step = int(pythia_tokens_per_step)

    local_batch_size = args.local_batch_size or get_nested(cfg, [
        ["continuation", "batch_size"],
        ["continuation", "local_batch_size"],
        ["training", "continuation_batch_size"],
        ["training", "batch_size"],
    ], None)
    if local_batch_size is None:
        raise ValueError("Missing continuation batch size. Set continuation.batch_size or pass --local-batch-size")
    local_batch_size = int(local_batch_size)

    sequence_length = args.sequence_length or get_nested(cfg, [
        ["continuation", "sequence_length"],
        ["continuation", "max_length"],
        ["training", "sequence_length"],
        ["training", "max_length"],
    ], 256)
    sequence_length = int(sequence_length)

    grad_accum_steps = args.grad_accum_steps or get_nested(cfg, [
        ["continuation", "grad_accum_steps"],
        ["continuation", "gradient_accumulation_steps"],
        ["training", "grad_accum_steps"],
        ["training", "gradient_accumulation_steps"],
    ], 1)
    grad_accum_steps = int(grad_accum_steps)

    local_tokens_per_step = local_batch_size * sequence_length * grad_accum_steps
    if local_tokens_per_step <= 0:
        raise ValueError("local_tokens_per_step must be positive")

    arms = normalise_arms(resolved)
    rows = []
    new_arms: list[tuple[str, dict]] = []

    for name, arm in arms:
        inject_step = arm.get("inject_step", arm.get("inject_at", arm.get("revision", arm.get("checkpoint"))))
        inject_step_num = parse_step(inject_step)
        global_delta_steps = max(0, endpoint_step - inject_step_num)
        target_tokens = global_delta_steps * pythia_tokens_per_step
        local_cont_steps = int(math.ceil(target_tokens / local_tokens_per_step)) if target_tokens > 0 else 0
        if global_delta_steps > 0 and args.min_continuation_steps:
            local_cont_steps = max(local_cont_steps, args.min_continuation_steps)
        actual_tokens = local_cont_steps * local_tokens_per_step
        token_overshoot = actual_tokens - target_tokens

        arm = copy.deepcopy(arm)
        arm["inject_step_num"] = inject_step_num
        arm["endpoint_step_num"] = endpoint_step
        arm["pythia_delta_steps"] = global_delta_steps
        arm["target_continuation_tokens"] = int(target_tokens)
        arm["local_tokens_per_step"] = int(local_tokens_per_step)
        arm["continuation_steps"] = int(local_cont_steps)
        arm["actual_continuation_tokens"] = int(actual_tokens)
        arm["token_overshoot"] = int(token_overshoot)
        # Backward-compatible aliases for runners with different names.
        arm["continue_steps"] = int(local_cont_steps)
        arm["cont_steps"] = int(local_cont_steps)
        new_arms.append((name, arm))

        rows.append({
            "arm": name,
            "inject_step": inject_step,
            "inject_step_num": inject_step_num,
            "endpoint_step_num": endpoint_step,
            "pythia_delta_steps": global_delta_steps,
            "pythia_tokens_per_step": pythia_tokens_per_step,
            "target_continuation_tokens": int(target_tokens),
            "local_batch_size": local_batch_size,
            "sequence_length": sequence_length,
            "grad_accum_steps": grad_accum_steps,
            "local_tokens_per_step": int(local_tokens_per_step),
            "continuation_steps": int(local_cont_steps),
            "actual_continuation_tokens": int(actual_tokens),
            "token_overshoot": int(token_overshoot),
        })

    write_arms(resolved, new_arms)
    set_nested(resolved, ["continuation", "endpoint_step"], endpoint_step)
    set_nested(resolved, ["continuation", "pythia_tokens_per_step"], pythia_tokens_per_step)
    set_nested(resolved, ["continuation", "local_tokens_per_step"], int(local_tokens_per_step))
    set_nested(resolved, ["continuation", "budget_resolved_by"], "plan_c1_continuation_token_budgets.py")

    out_config = Path(args.out_config)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(resolved, sort_keys=False))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote resolved config: {out_config}")
    print(f"Wrote budget table:    {out_csv}")
    print(f"Endpoint step: {endpoint_step}")
    print(f"Pythia tokens/step: {pythia_tokens_per_step:,}")
    print(f"Local tokens/step:  {local_tokens_per_step:,}")
    print("\nBudgets:")
    for r in rows:
        print(
            f"  {r['arm']}: inject {r['inject_step_num']} -> {endpoint_step}, "
            f"target {r['target_continuation_tokens']:,} tokens, "
            f"local steps {r['continuation_steps']:,}"
        )


if __name__ == "__main__":
    main()
