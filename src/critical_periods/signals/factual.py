from __future__ import annotations

import random

from .base import SignalData, SignalSpec
from .vocab import make_vocab

RELATIONS = ["capital", "currency", "patron", "river", "festival", "symbol"]
TRAIN_TEMPLATES = [
    "The {rel} of {e} is {v}.",
    "{v} is the {rel} of {e}.",
    "{e}'s {rel} is {v}.",
    "In the fictional gazetteer, {e} has {v} as its {rel}.",
]
PROBE_TEMPLATES = [
    "Question: What is the {rel} of {e}?\nAnswer:",
    "In the fictional gazetteer, what is {e}'s {rel}?\nAnswer:",
]


def _sample_distractors(rng: random.Random, vals: list[str], correct: str, n: int) -> list[str]:
    candidates = [v for v in vals if v != correct]
    if len(candidates) < n:
        raise ValueError("Not enough distractor candidates; increase n_items.")
    return [" " + v for v in rng.sample(candidates, n)]


def make_factual(spec: SignalSpec, tok) -> SignalData:
    """Synthetic factual association signal.

    Train templates and probe templates are disjoint. Probes are closed-set scored against
    distractor values drawn from the taught value pool. Controls use untrained triples.
    """
    rng = random.Random(spec.seed)
    ents = make_vocab(rng, spec.n_items, tok, max_pieces=spec.max_name_pieces)
    vals = make_vocab(
        rng,
        spec.n_items,
        tok,
        max_pieces=spec.max_name_pieces,
        exclude=ents,
    )
    rels = [rng.choice(RELATIONS) for _ in ents]

    train_texts: list[str] = []
    for e, r, v in zip(ents, rels, vals):
        for tmpl in TRAIN_TEMPLATES:
            train_texts.append(tmpl.format(rel=r, e=e, v=v))

    # Probe subset is deterministic but seed-shuffled.
    idxs = list(range(spec.n_items))
    rng.shuffle(idxs)
    idxs = idxs[: min(spec.n_probe, spec.n_items)]

    probes = []
    for j, i in enumerate(idxs):
        e, r, v = ents[i], rels[i], vals[i]
        tmpl = PROBE_TEMPLATES[j % len(PROBE_TEMPLATES)]
        probes.append(
            {
                "signal_type": "factual",
                "item_id": f"fact_{i}",
                "prefix": tmpl.format(rel=r, e=e),
                "correct": " " + v,
                "distractors": _sample_distractors(rng, vals, v, spec.n_distractors),
                "entity": e,
                "relation": r,
                "value": v,
            }
        )

    # Controls use unseen entities/values, with distractors from the control value pool.
    ce = make_vocab(
        rng,
        spec.n_probe,
        tok,
        max_pieces=spec.max_name_pieces,
        exclude=set(ents) | set(vals),
    )
    cv = make_vocab(
        rng,
        spec.n_probe + spec.n_distractors + 2,
        tok,
        max_pieces=spec.max_name_pieces,
        exclude=set(ents) | set(vals) | set(ce),
    )
    controls = []
    for i, (e, v) in enumerate(zip(ce, cv[: spec.n_probe])):
        r = rng.choice(RELATIONS)
        distractor_pool = [x for x in cv if x != v]
        controls.append(
            {
                "signal_type": "factual_control",
                "item_id": f"control_{i}",
                "prefix": PROBE_TEMPLATES[i % len(PROBE_TEMPLATES)].format(rel=r, e=e),
                "correct": " " + v,
                "distractors": [" " + d for d in rng.sample(distractor_pool, spec.n_distractors)],
                "entity": e,
                "relation": r,
                "value": v,
            }
        )

    return SignalData(
        train_texts=train_texts,
        train_pairs=[],
        probes=probes,
        controls=controls,
        meta={
            "name": "factual",
            "relations": RELATIONS,
            "train_templates": TRAIN_TEMPLATES,
            "probe_templates": PROBE_TEMPLATES,
            "mask_prompt": False,
            "n_items": spec.n_items,
            "n_probe": len(probes),
            "seed": spec.seed,
        },
    )
