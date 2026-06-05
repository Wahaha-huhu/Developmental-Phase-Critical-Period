#!/usr/bin/env python3
"""Generate E1 large-scale Pythia configs from an existing working E1 YAML.

This script is intentionally conservative: it uses dense early checkpoints up to
step8000 plus sparse late checkpoints, and can also generate a very sparse key
checkpoint config for 12B.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Iterable

import yaml


def fallback_all_pythia_steps() -> list[int]:
    # Pythia checkpoints: step0, powers of two to 512, then every 1000 to 143000.
    steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    steps += list(range(1000, 144000, 1000))
    return sorted(set(steps))


def try_hf_steps(repo_id: str) -> list[int]:
    try:
        from huggingface_hub import HfApi
        refs = HfApi().list_repo_refs(repo_id=repo_id)
        names = []
        for group in [refs.branches, refs.tags]:
            for r in group:
                names.append(r.name)
        steps = []
        for n in names:
            m = re.fullmatch(r"step(\d+)", n)
            if m:
                steps.append(int(m.group(1)))
        if steps:
            return sorted(set(steps))
    except Exception as e:
        print(f"[WARN] Could not query HF refs for {repo_id}: {type(e).__name__}: {e}")
    return fallback_all_pythia_steps()


def dense_early_sparse_late_steps(all_steps: Iterable[int], early_max_step: int) -> list[str]:
    all_steps = sorted(set(int(s) for s in all_steps))
    early = [s for s in all_steps if s <= early_max_step]
    late_targets = [16000, 32000, 64000, 100000, 143000]
    late = [s for s in late_targets if s in all_steps and s > early_max_step]
    return [f"step{s}" for s in sorted(set(early + late))]


def sparse_key_steps(all_steps: Iterable[int]) -> list[str]:
    all_steps = set(int(s) for s in all_steps)
    targets = [0, 512, 1000, 2000, 8000, 143000]
    return [f"step{s}" for s in targets if s in all_steps]


def set_outputs_root(cfg: dict, root: str) -> None:
    cfg.setdefault("outputs", {})["root"] = root
    # Some older scripts/configs used output_root; keeping both is harmless.
    cfg["output_root"] = root


def set_cheap_metrics(cfg: dict) -> None:
    # This only has an effect if the runner reads metric toggles. If it does not,
    # the config remains valid and the runner will compute its default metrics.
    cfg["metrics"] = {
        "frobenius_norm": True,
        "spectral_norm": True,
        "stable_rank": True,
        "effective_rank": True,
        "mp_outliers": False,
        "alpha": False,
        "subspace_stability": False,
    }


def make_config(template: dict, repo: str, label: str, checkpoints: list[str], out_root: str, cheap: bool) -> dict:
    cfg = copy.deepcopy(template)
    cfg["models"] = [repo]
    cfg["checkpoints"] = checkpoints
    set_outputs_root(cfg, out_root)
    if cheap:
        set_cheap_metrics(cfg)
    cfg.setdefault("metadata", {})["generated_by"] = "scripts/generate_e1_large_scale_configs.py"
    cfg["metadata"]["large_scale_label"] = label
    cfg["metadata"]["checkpoint_count"] = len(checkpoints)
    cfg["metadata"]["metric_set"] = "cheap_norm_rank" if cheap else "template_default"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="configs/e1_phase_identification.yaml")
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--results-root", default="results/e1_large_scale")
    ap.add_argument("--early-max-step", type=int, default=8000)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--mode", choices=["dense_early_sparse_late", "sparse_key"], default="dense_early_sparse_late")
    ap.add_argument("--full-metrics", action="store_true", help="Use template metrics instead of cheap norm/rank subset.")
    args = ap.parse_args()

    template = yaml.safe_load(Path(args.template).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_models = [
        "EleutherAI/pythia-1.4b-deduped",
        "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9b-deduped",
        "EleutherAI/pythia-12b-deduped",
    ]
    models = args.model or default_models

    for repo in models:
        size = repo.split("pythia-")[-1].replace("-deduped", "")
        label = size.replace(".", "p")
        steps = try_hf_steps(repo)
        if args.mode == "sparse_key":
            checkpoints = sparse_key_steps(steps)
            suffix = "sparse_key"
        else:
            checkpoints = dense_early_sparse_late_steps(steps, args.early_max_step)
            suffix = "dense_early_sparse_late"
        out_root = f"{args.results_root}/{label}_{suffix}"
        cfg = make_config(
            template=template,
            repo=repo,
            label=label,
            checkpoints=checkpoints,
            out_root=out_root,
            cheap=not args.full_metrics,
        )
        path = out_dir / f"e1_large_{label}_{suffix}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Wrote {path} with {len(checkpoints)} checkpoints for {repo}")
        print("  first:", checkpoints[:8])
        print("  last: ", checkpoints[-8:])


if __name__ == "__main__":
    main()
