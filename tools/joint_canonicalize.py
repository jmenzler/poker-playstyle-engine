"""Joint suit-isomorphic canonicalization of (hole, board) for equity lookup.

Board-only canonicalization (research/01-foundation/enumerate_canonical_flops.py)
collapses 22,100 raw flops → 1,755 canonical forms. That is correct for
range-vs-range solver work where hole cards are abstracted into a range.

For per-combo equity, we need JOINT canonicalization that preserves the
hole↔board suit relationship: JdTd on Kd7d2c (flush draw + backdoor) must
NOT collapse to JdTd on Kc7c2h (no flush draw).

Algorithm:
    1. Treat (hole, board) as one card set.
    2. Enumerate all 24 suit permutations of the 4 suits.
    3. For each perm: apply to both hole and board; sort hole by (rank desc,
       suit asc); sort board by (rank desc, suit asc); form joint string.
    4. Return lex-min over all 24 perms.

This preserves:
    * Suit matches between hole and board (FD, BD, made flush)
    * Suit pattern within board (rainbow / two-tone / monotone)
    * Card-order invariance within hole and within board

This collapses:
    * Absolute suit identity (Jd ≡ Jc ≡ Jh ≡ Js when nothing else cares)
"""

from __future__ import annotations

from itertools import permutations

SUITS = "cdhs"
RANKS = "23456789TJQKA"
_RANK_VALUE = {r: i for i, r in enumerate(RANKS)}
_SUIT_PERMS: list[dict[str, str]] = [dict(zip(SUITS, perm, strict=False)) for perm in permutations(SUITS)]


def _parse_cards(s: str | list[str]) -> list[str]:
    """Accept 'Ah Kc' or ['Ah', 'Kc'] → ['Ah', 'Kc']."""
    if isinstance(s, str):
        return s.split()
    return list(s)


def _sort_cards(cards: list[str]) -> str:
    """Sort by (rank desc, suit asc), join into single string."""
    return "".join(sorted(cards, key=lambda c: (-_RANK_VALUE[c[0]], c[1])))


def joint_canonicalize(
    hole: str | list[str],
    board: str | list[str],
) -> tuple[str, str]:
    """Return (canonical_hole, canonical_board) under joint suit-iso.

    Args:
        hole: 'Ah Kc' or ['Ah', 'Kc']
        board: 'AcJd5d' or ['Ac', 'Jd', '5d'] (0, 3, 4, or 5 cards)

    Returns:
        (canonical_hole_str, canonical_board_str). Hole always sorted high-rank
        first. Board sorted high-rank first within its 3/4/5 cards.
    """
    hole_cards = _parse_cards(hole)
    board_cards = _parse_cards(board)
    if len(hole_cards) != 2:
        raise ValueError(f"hole must be 2 cards, got {hole_cards!r}")
    if len(board_cards) not in (0, 3, 4, 5):
        raise ValueError(f"board must be 0/3/4/5 cards, got {board_cards!r}")

    best: tuple[str, str] | None = None
    for pmap in _SUIT_PERMS:
        ph = [c[0] + pmap[c[1]] for c in hole_cards]
        pb = [c[0] + pmap[c[1]] for c in board_cards]
        key = (_sort_cards(ph), _sort_cards(pb))
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def joint_canonical_key(hole: str | list[str], board: str | list[str]) -> str:
    """Single string key combining canonical hole + board, '|'-separated."""
    h, b = joint_canonicalize(hole, board)
    return f"{h}|{b}"


def canonicalize_board(board: str | list[str]) -> str:
    """Board-only suit-iso canonical (hole-agnostic, for range-level grouping).

    Unlike joint_canonicalize, no hole suits pollute the orbit, so suit-iso boards
    (AhKh7c == AsKs7d) always collapse. Returns the lex-min over all 24 suit perms.
    """
    cards = _parse_cards(board)
    if len(cards) not in (3, 4, 5):
        raise ValueError(f"board must be 3/4/5 cards, got {cards!r}")
    return min(_sort_cards([c[0] + pmap[c[1]] for c in cards]) for pmap in _SUIT_PERMS)


if __name__ == "__main__":
    cases = [
        # FD preserved: hole suit matches 2 board suits
        (("Jd", "Td"), ("Kd", "7d", "2c"), "FD on K-high two-tone"),
        # Permuted suits → same canonical
        (("Jc", "Tc"), ("Kc", "7c", "2h"), "FD on K-high two-tone (perm)"),
        # No FD: different suit relationship
        (("Jd", "Td"), ("Ks", "7s", "2c"), "no FD, JT vs Khh-2c"),
        # Board-only matches but JOINT differs
        (("Jd", "Td"), ("Ks", "7c", "2h"), "rainbow no FD JT"),
        (("Jd", "Td"), ("Kc", "7h", "2s"), "rainbow no FD JT (perm)"),
    ]
    print("Sanity check — same descriptor should collapse to same canonical:\n")
    for hole, board, desc in cases:
        h, b = joint_canonicalize(hole, board)
        print(f"  {''.join(hole):<6} {''.join(board):<8}  →  {h} | {b}   ({desc})")
