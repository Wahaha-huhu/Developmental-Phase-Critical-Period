#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Allow running from a fresh checkout without requiring `pip install -e .` first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from critical_periods.indicators.spectral import compute_spectral_indicators, subspace_cosine_mean
from critical_periods.io.artifacts import ArtifactRecord, ArtifactRegistry, ensure_experiment_dirs
from critical_periods.io.config import load_yaml
from critical_periods.models.pythia import load_pythia_checkpoint, matched_matrix_specs


def checkpoint_to_int(checkpoint: str) -> int:
    if checkpoint == "main":
        return 143000
    if checkpoint.startswith("step"):
        return int(checkpoint.replace("step", ""))
    raise ValueError(f"Cannot parse checkpoint step from {checkpoint!r}")


def collect_for_model(config: dict, model_name: str) -> list[dict]:
    runtime = config.get("runtime", {})
    svd_cfg = config.get("svd", {})
    checkpoints = config["checkpoints"]
    module_suffixes = config["module_suffixes"]

    rows: list[dict] = []
    prev_top_u: dict[str, object] = {}
    prev_checkpoint: str | None = None

    for checkpoint in tqdm(checkpoints, desc=f"{model_name}"):
        model = load_pythia_checkpoint(
            model_name=model_name,
            checkpoint=checkpoint,
            device=runtime.get("device", "cuda"),
            dtype=runtime.get("dtype", "float32"),
            cache_dir=runtime.get("cache_dir"),
            local_files_only=bool(runtime.get("local_files_only", False)),
            trust_remote_code=bool(runtime.get("trust_remote_code", False)),
        )
        state = model.state_dict()
        specs = matched_matrix_specs(state, module_suffixes)

        for spec in tqdm(specs, desc=f"  {checkpoint}", leave=False):
            weight = state[spec.name]
            result = compute_spectral_indicators(
                weight=weight,
                top_k_vectors=int(svd_cfg.get("top_k_vectors", 8)),
                mp_edge_multipliers=list(svd_cfg.get("mp_edge_multipliers", [1.0, 1.1, 1.25])),
                alpha_tail_fracs=list(svd_cfg.get("alpha_tail_fracs", [0.2, 0.3, 0.5])),
                center_weights=bool(svd_cfg.get("center_weights", False)),
            )
            stability = subspace_cosine_mean(prev_top_u.get(spec.name), result.top_left_vectors)
            row = {
                "experiment_id": config["experiment_id"],
                "model": model_name,
                "checkpoint": checkpoint,
                "step": checkpoint_to_int(checkpoint),
                "prev_checkpoint": prev_checkpoint or "",
                "matrix_name": spec.name,
                "module": spec.module_suffix,
                "layer": spec.layer if spec.layer is not None else -1,
                "subspace_stability_topk": stability,
                **result.metrics,
            }
            rows.append(row)
            prev_top_u[spec.name] = result.top_left_vectors

        prev_checkpoint = checkpoint
        del model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect E1 spectral indicators across Pythia checkpoints.")
    parser.add_argument("--config", required=True, help="Path to E1 YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    output_root = Path(config["outputs"]["root"])
    dirs = ensure_experiment_dirs(output_root)

    all_rows: list[dict] = []
    for model_name in config["models"]:
        all_rows.extend(collect_for_model(config, model_name))

    df = pd.DataFrame(all_rows).sort_values(["model", "step", "layer", "module"])
    metrics_path = output_root / config["outputs"]["raw_metrics_csv"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_path, index=False)

    manifest_path = output_root / config["outputs"].get("manifest_csv", "manifests/artifact_manifest.csv")
    registry = ArtifactRegistry(manifest_path)
    registry.append(
        ArtifactRecord(
            experiment_id=config["experiment_id"],
            artifact_type="raw_csv",
            path=metrics_path,
            thesis_section=config.get("thesis_section", ""),
            caption_draft="Raw per-matrix weight-spectral indicators across Pythia checkpoints.",
            source_data="Hugging Face Pythia checkpoint revisions",
            code_entrypoint="scripts/run_e1_collect_spectra.py",
            status="draft",
            notes=f"Config: {config_path}",
        )
    )
    print(f"Wrote {len(df):,} metric rows to {metrics_path}")
    print(f"Updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
