from __future__ import annotations

import random

from .base import SignalData, SignalSpec


def make_procedural(spec: SignalSpec, tok=None) -> SignalData:
    """Simple synthetic substitution rule.

    Kept for E4. E3-MVP should not depend on this until a smoke test verifies uptake.
    """
    rng = random.Random(spec.seed)
    alphabet = list("ABCDEFGH")
    perm = rng.sample(alphabet, len(alphabet))
    sub = dict(zip(alphabet, perm))

    def apply(s: str) -> str:
        return "".join(sub[c] for c in s)

    def rand_in(n: int = 5) -> str:
        return "".join(rng.choice(alphabet) for _ in range(n))

    seen: set[str] = set()
    while len(seen) < spec.n_items + spec.n_probe:
        seen.add(rand_in())
    all_inputs = list(seen)
    rng.shuffle(all_inputs)
    train_inputs = all_inputs[: spec.n_items]
    probe_inputs = all_inputs[spec.n_items : spec.n_items + spec.n_probe]

    train_pairs = [(f"Input: {x}\nOutput:", " " + apply(x)) for x in train_inputs]
    probes = []
    for i, x in enumerate(probe_inputs):
        distractors = []
        while len(distractors) < spec.n_distractors:
            cand = " " + rand_in()
            if cand != " " + apply(x) and cand not in distractors:
                distractors.append(cand)
        probes.append(
            {
                "signal_type": "procedural",
                "item_id": f"proc_{i}",
                "prefix": f"Input: {x}\nOutput:",
                "correct": " " + apply(x),
                "distractors": distractors,
                "input": x,
            }
        )
    return SignalData(
        train_texts=[],
        train_pairs=train_pairs,
        probes=probes,
        controls=[],
        meta={"name": "procedural", "perm": perm, "mask_prompt": True, "seed": spec.seed},
    )
