from __future__ import annotations

from .alignment import make_alignment
from .base import SignalData, SignalSpec
from .factual import make_factual
from .procedural import make_procedural


def make_signal(spec: SignalSpec, tok) -> SignalData:
    name = spec.name.lower()
    if name == "factual":
        return make_factual(spec, tok)
    if name == "procedural":
        return make_procedural(spec, tok)
    if name == "alignment":
        return make_alignment(spec, tok)
    raise ValueError(f"Unknown signal name: {spec.name}")


__all__ = ["SignalSpec", "SignalData", "make_signal", "make_factual", "make_procedural", "make_alignment"]
