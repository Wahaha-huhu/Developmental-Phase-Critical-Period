from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class SignalSpec:
    """Configuration for a synthetic injection signal.

    The same class is used for factual, procedural, and alignment-style signals so E4 can
    match the signal families on item count, probe count, seed, and training budget.
    """

    name: str  # factual | procedural | alignment
    n_items: int = 300
    n_probe: int = 200
    seed: int = 0
    max_name_pieces: int = 3
    mask_prompt: bool | None = None
    n_distractors: int = 4


@dataclass
class SignalData:
    """Generated train/probe/control data for one signal family."""

    train_texts: List[str] = field(default_factory=list)
    train_pairs: List[Tuple[str, str]] = field(default_factory=list)
    probes: List[Dict[str, Any]] = field(default_factory=list)
    controls: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_train_examples(self) -> int:
        return len(self.train_texts) + len(self.train_pairs)
