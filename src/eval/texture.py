"""Board texture classifier for Eval per-texture EV breakdown (D-NEW-30, OQ-5 lock).

Priority order (LOCKED by the Phase 6 planner — see RESEARCH OQ-5):
    MONOTONE > PAIRED > WET TWO-TONE > DRY RAINBOW

A "card" is a 2-char string: rank-char (``2-9TJQKA``) + suit-char (``cdhs``).
This module is pure — no DB, no I/O. Safe to call per hand inside run_match.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from src._log import get_logger

log = get_logger("eval.texture")

TextureClass = Literal["monotone", "paired", "wet_two_tone", "dry_rainbow"]


def classify_board(board: list[str]) -> TextureClass:
    """Classify a poker board's texture for the Eval per-texture breakdown.

    Args:
        board: 3-5 card strings, e.g. ``["Ah", "Td", "5c"]``.
            Empty or fewer than 3 cards returns ``"dry_rainbow"`` with a logged
            warning (preflop or malformed input — should not occur in Eval flow).

    Returns:
        One of ``"monotone" | "paired" | "wet_two_tone" | "dry_rainbow"``.

    Priority (OQ-5 lock):
        1. monotone — all visible cards share the same suit.
        2. paired   — any rank appears ≥ 2 times.
        3. wet_two_tone — exactly 2 distinct suits.
        4. dry_rainbow — 3 or 4 distinct suits (catch-all "anything else").

    Note:
        The classifier operates on the string features only. Degenerate inputs
        such as duplicate cards (impossible in a real deck) are not rejected —
        the priority cascade is applied as-is.
    """
    if not board or len(board) < 3:
        log.warning("texture.insufficient_cards", n=len(board) if board else 0)
        return "dry_rainbow"

    suits = [card[1] for card in board]
    ranks = [card[0] for card in board]
    suit_counts = Counter(suits)
    rank_counts = Counter(ranks)

    # Priority 1: monotone (all same suit).
    if len(suit_counts) == 1:
        return "monotone"

    # Priority 2: paired (any rank appears ≥ 2 times).
    if max(rank_counts.values()) >= 2:
        return "paired"

    # Priority 3: wet_two_tone (exactly 2 distinct suits).
    if len(suit_counts) == 2:
        return "wet_two_tone"

    # Priority 4: dry_rainbow (catch-all — 3 or 4 distinct suits).
    return "dry_rainbow"
