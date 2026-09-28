"""Map a 2-card hand to its 169-class (rank-desc + s/o/pair).

Two-card-input port of research/preflop-ranges/extract_ranges.py:hand_class —
the chart tree and live engine must agree on combo keys or lookups silently miss.
"""

from __future__ import annotations

RANK_ORDER = "AKQJT98765432"
SUITS = "cdhs"


def combo_169(card_a: str, card_b: str) -> str:
    """'Ah','Kh' -> 'AKs'; 'Ah','Ks' -> 'AKo'; '7c','7d' -> '77'.

    Order-independent; pairs ignore suit. Raises ValueError on malformed input.
    """
    for c in (card_a, card_b):
        if not isinstance(c, str) or len(c) != 2 or c[0] not in RANK_ORDER or c[1] not in SUITS:
            raise ValueError(f"malformed card: {c!r}")
    r1, s1 = card_a[0], card_a[1]
    r2, s2 = card_b[0], card_b[1]
    if RANK_ORDER.index(r1) > RANK_ORDER.index(r2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def _build_all_combos_169() -> tuple[str, ...]:
    out: list[str] = []
    for i, hi in enumerate(RANK_ORDER):
        out.append(f"{hi}{hi}")
        for lo in RANK_ORDER[i + 1 :]:
            out.append(f"{hi}{lo}s")
            out.append(f"{hi}{lo}o")
    return tuple(out)


ALL_COMBOS_169: tuple[str, ...] = _build_all_combos_169()
