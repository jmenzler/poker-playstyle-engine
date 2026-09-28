"""Snap size-less action stems to sized canonical labels at serve time.

Sizing is attached from the QUERY spot, not baked into the corpus (snap-at-
retrieval). Preflop is rule-based by pot_type + position; neighbor votes the stem.
"""

from __future__ import annotations

_PREFLOP_PASSTHROUGH = frozenset({"fold", "call", "check", "allin"})
_BLIND_POS = frozenset({"SB", "BB"})


def preflop_action_label(
    stem: str,
    pot_type: str | None,
    hero_pos_rel: str | None,
    hero_pos: str | None = None,
) -> str:
    """Map a preflop stem to a sized canonical label using the query spot.

    raise: limp→open (BvB→open_3bb else open_2_2bb); srp→3bet (OOP→3bet_4x,
    IP→3bet_3x); 3bet→4bet_2_5x; 4bet/5bet+→allin. Non-raise stems pass through.
    """
    if stem in _PREFLOP_PASSTHROUGH:
        return stem
    if stem != "raise":
        return stem  # unknown stem — blend skips it if non-canonical
    if pot_type in (None, "limp", "unopened"):
        return "open_3bb" if hero_pos in _BLIND_POS else "open_2_2bb"
    if pot_type == "srp":
        return "3bet_4x" if hero_pos_rel == "OOP" else "3bet_3x"
    if pot_type == "3bet":
        return "4bet_2_5x"
    # 4bet pot or 5bet+ — hero's raise is a 5bet, played as a jam in this corpus.
    return "allin"


# Snap to the EXISTING canonical buckets (no vocab change → no solver/sim/cli ripple).
# Bets: nearest pot-fraction; very large → overbet catch-all. Raises: nearest to/facing.
_BET_BUCKETS: tuple[tuple[float, str], ...] = (
    (0.25, "bet_25"),
    (0.33, "bet_33"),
    (0.50, "bet_50"),
    (0.75, "bet_75"),
    (1.00, "bet_100"),
    (1.50, "bet_150"),
)
_OVERBET_FRAC = 1.75
_RAISE_BUCKETS: tuple[tuple[float, str], ...] = (
    (1.00, "raise_min"),
    (2.50, "raise_2_5x"),
    (3.00, "raise_3x"),
    (4.00, "raise_pot"),
)


def snap_postflop_bet(pot_frac: float | None, is_allin: bool) -> str:
    """Snap a postflop bet's size (fraction of pot) to a canonical bucket."""
    if is_allin:
        return "allin"
    frac = float(pot_frac) if pot_frac is not None else 0.50
    if frac >= _OVERBET_FRAC:
        return "bet_overbet"
    return min(_BET_BUCKETS, key=lambda b: abs(b[0] - frac))[1]


def snap_postflop_raise(raise_ratio: float | None, is_allin: bool) -> str:
    """Snap a postflop raise's size (to/facing ratio) to a canonical bucket."""
    if is_allin:
        return "allin"
    ratio = float(raise_ratio) if raise_ratio is not None else 2.5
    return min(_RAISE_BUCKETS, key=lambda b: abs(b[0] - ratio))[1]


def snap_postflop_action(stem: str, pot_frac: float | None, raise_ratio: float | None, is_allin: bool) -> str:
    """Snap a postflop stem to a sized canonical label. Non-bet/raise pass through."""
    if stem == "bet":
        return snap_postflop_bet(pot_frac, is_allin)
    if stem == "raise":
        return snap_postflop_raise(raise_ratio, is_allin)
    return stem
