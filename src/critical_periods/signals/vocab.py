from __future__ import annotations

import random
from typing import Iterable, Set

CONS = "bdfgklmnprstvz"
VOW = "aeiou"


def make_nonword(rng: random.Random, syll: tuple[int, int] = (2, 3)) -> str:
    """Generate a pronounceable fictional name."""
    n = rng.randint(*syll)
    return "".join(rng.choice(CONS) + rng.choice(VOW) for _ in range(n)).capitalize()


def token_piece_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def make_vocab(
    rng: random.Random,
    k: int,
    tok,
    max_pieces: int = 3,
    exclude: Iterable[str] | None = None,
    max_attempts: int = 200_000,
) -> list[str]:
    """Generate k synthetic names with bounded tokenizer length.

    GPT-NeoX tokenizers are space-sensitive, so the length check uses a leading space,
    matching answer scoring where continuations are scored as " Value".
    """
    excluded: Set[str] = set(exclude or [])
    out: Set[str] = set()
    attempts = 0
    while len(out) < k:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {k} names with max_pieces={max_pieces}; "
                "increase max_pieces or reduce k."
            )
        w = make_nonword(rng)
        if w in excluded or w in out:
            continue
        if token_piece_len(tok, " " + w) <= max_pieces:
            out.add(w)
    return sorted(out)
