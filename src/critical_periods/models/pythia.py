from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    module_suffix: str
    layer: int | None
    shape: tuple[int, int]


def parse_pythia_layer(name: str) -> int | None:
    """Extract layer index from names like gpt_neox.layers.3.mlp..."""
    parts = name.split(".")
    try:
        idx = parts.index("layers")
        return int(parts[idx + 1])
    except (ValueError, IndexError):
        return None


def matched_matrix_specs(
    state_dict: dict[str, torch.Tensor],
    module_suffixes: Iterable[str],
) -> list[MatrixSpec]:
    suffixes = tuple(module_suffixes)
    specs: list[MatrixSpec] = []
    for name, tensor in state_dict.items():
        if tensor.ndim != 2:
            continue
        suffix_match = next((suffix for suffix in suffixes if name.endswith(suffix)), None)
        if suffix_match is None:
            continue
        specs.append(
            MatrixSpec(
                name=name,
                module_suffix=suffix_match.replace(".weight", ""),
                layer=parse_pythia_layer(name),
                shape=tuple(tensor.shape),
            )
        )
    return specs


def load_pythia_checkpoint(
    model_name: str,
    checkpoint: str,
    device: str = "cuda",
    dtype: str = "float32",
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
):
    """Load a Pythia checkpoint revision from Hugging Face.

    `checkpoint` is usually a revision string such as `step0`, `step512`, or `step143000`.
    """
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=checkpoint,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.eval().to(device)
    return model
