"""Flop-spot grouping: collapse DPs sharing one flop solve to a single key.

Key is (canonical_flop, range_config, exact flop-start stack). is_multiway flags
3+ player postflop DPs the HU solver cannot model (skip+flag upstream).
"""

from __future__ import annotations

from typing import Any

from tools.joint_canonicalize import canonicalize_board

__all__ = ["flop_spot_key", "flop_start_stack_bb", "is_multiway"]


def is_multiway(felt: dict[str, Any]) -> bool:
    """True when 3+ players are at the street (HU=2; multiway is unsupported)."""
    return int(felt.get("opponents_remaining", 1)) + 1 >= 3


def flop_start_stack_bb(
    felt: dict[str, Any],
    flop_start_stack_by_hand: dict[str, float] | None,
    hand_id: str | None,
) -> float:
    """Resolve the hand's flop-start effective stack (shared across its streets).

    Prefer the per-hand flop-start stack supplied by the caller (each hand's flop DP);
    fall back to the per-DP remaining effective_stack_bb (exact only for flop DPs).
    """
    if flop_start_stack_by_hand and hand_id and hand_id in flop_start_stack_by_hand:
        return flop_start_stack_by_hand[hand_id]
    stack = felt.get("effective_stack_bb")
    if stack is None:
        raise ValueError("felt missing effective_stack_bb and no flop_start_stack_by_hand entry")
    return float(stack)


def flop_spot_key(
    felt: dict[str, Any],
    action_sequence: list[str],
    hero_pos: str,
    *,
    flop_start_stack_by_hand: dict[str, float] | None = None,
    hand_id: str | None = None,
) -> tuple[str, tuple[str, str, str, str, str], int]:
    """Group key (canonical_flop, range_config, flop_start_stack) for one postflop DP.

    range_config = (pot_type, hero_pos, villain, opener, bettor). Stack is the
    exact (un-bucketed) flop-start eff stack rounded to whole bb.
    """
    flop_cards = (felt.get("board_cards") or [])[:3]
    if len(flop_cards) != 3:
        raise ValueError(f"postflop DP requires a 3-card flop, got {flop_cards!r}")

    canon_flop = canonicalize_board(flop_cards)

    # Lazy import: queue_driver imports flop_spot, so import here to avoid a cycle.
    from src.solver.queue_driver import _derive_preflop_info

    pot_type, villain, opener, bettor, _ = _derive_preflop_info(action_sequence, hero_pos)
    range_config = (pot_type, hero_pos, villain, opener, bettor)

    stack = round(flop_start_stack_bb(felt, flop_start_stack_by_hand, hand_id))
    return canon_flop, range_config, stack
