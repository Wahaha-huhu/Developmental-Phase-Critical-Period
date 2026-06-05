#!/usr/bin/env python3
"""Generate dense E1 configs from an existing working E1 config.

This script queries Hugging Face for Pythia checkpoint revisions, filters them by
step, and writes YAML configs that can be used by scripts/run_e1_collect_spectra.py.
It avoids hand-writing the 154 checkpoint list.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import yaml
from huggingface_hub import HfApi

_STEP_RE = re.compile(r"^step(\d+)$")


def available_step_revisions(repo_id: str) -> list[str]:
    api = HfApi()
    refs = api.list_repo_refs(repo_id)
    names = []
    for group_name in ("branches", "tags"):
        for ref in getattr(refs, group_name, []) or []:
            name = getattr(ref, "name", None)
            if name:
                names.append(name)
    steps = []
    seen = set()
    for name in names:
        m = _STEP_RE.match(name)
        if m and name not in seen:
            steps.append((int(m.group(1)), name))
            seen.add(name)
    steps.sort(key=lambda x: x[0])
    return [name for _, name in steps]


def parse_step(revision: str) -> int:
    m = _STEP_RE.match(revision)
    if not m:
        raise ValueError(f"Not a step revision: {revision}")
    return int(m.group(1))


def filter_revisions(revisions: list[str], max_step: int | None = None, min_step: int | None = None) -> list[str]:
    out = []
    for rev in revisions:
        s = parse_step(rev)
        if min_step is not None and s < min_step:
            continue
        if max_step is not None and s > max_step:
            continue
        out.append(rev)
    return out


def set_output_root(cfg: dict, root: str) -> None:
    cfg.setdefault("outputs", {})["root"] = root
    # Some local runners have also used output_root; including both is harmless.
    cfg["output_root"] = root


def make_cfg(template: dict, model: str, revisions: list[str], output_root: str, cheap: bool = False) -> dict:
    cfg = copy.deepcopy(template)
    cfg["models"] = [model]
    cfg["checkpoints"] = revisions
    set_output_root(cfg, output_root)
    cfg.setdefault("notes", {})
    cfg["notes"]["generated_by"] = "scripts/generate_e1_dense_configs.py"
    cfg["notes"]["checkpoint_count"] = len(revisions)
    cfg["notes"]["checkpoint_min"] = parse_step(revisions[0]) if revisions else None
    cfg["notes"]["checkpoint_max"] = parse_step(revisions[-1]) if revisions else None
    if cheap:
        # These keys match the metric-toggle convention used in the E1 scaffold patches.
        # If your local runner ignores toggles, this config is still valid but not cheaper.
        cfg["metrics"] = {
            "frobenius_norm": True,
            "spectral_norm": True,
            "stable_rank": True,
            "effective_rank": True,
            "mp_outliers": False,
            "alpha": False,
            "subspace_stability": False,
        }
        cfg["notes"]["cheap_metrics"] = True
    else:
        cfg.setdefault("notes", {})["cheap_metrics"] = False
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="configs/e1_phase_identification.yaml")
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--early-max-step", type=int, default=8000)
    ap.add_argument("--include-1b", action="store_true", help="Also generate the 1B early cheap config")
    args = ap.parse_args()

    template_path = Path(args.template)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(template_path.read_text())

    jobs = [
        {
            "model": "EleutherAI/pythia-160m-deduped",
            "config_name": "e1_dense_160m_all_checkpoints.yaml",
            "output_root": "results/e1_dense_160m_all_checkpoints",
            "max_step": None,
            "cheap": False,
        },
        {
            "model": "EleutherAI/pythia-410m-deduped",
            "config_name": "e1_dense_410m_early_cheap.yaml",
            "output_root": "results/e1_dense_410m_early_cheap",
            "max_step": args.early_max_step,
            "cheap": True,
        },
    ]
    if args.include_1b:
        jobs.append(
            {
                "model": "EleutherAI/pythia-1b-deduped",
                "config_name": "e1_dense_1b_early_cheap.yaml",
                "output_root": "results/e1_dense_1b_early_cheap",
                "max_step": args.early_max_step,
                "cheap": True,
            }
        )

    for job in jobs:
        print(f"Querying revisions for {job['model']} ...")
        revs = available_step_revisions(job["model"])
        if not revs:
            raise RuntimeError(f"No step revisions found for {job['model']}")
        revs = filter_revisions(revs, max_step=job["max_step"])
        cfg = make_cfg(
            template,
            model=job["model"],
            revisions=revs,
            output_root=job["output_root"],
            cheap=job["cheap"],
        )
        out_path = out_dir / job["config_name"]
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Wrote {out_path} with {len(revs)} checkpoints: {revs[0]} -> {revs[-1]}")


if __name__ == "__main__":
    main()
