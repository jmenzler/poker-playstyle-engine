"""Parse a postflop action_sequence into the nav_line the Rust solve_harvest consumes.

Ordered {seat, kind, frac} steps for flop+turn+river actions; seat 0=OOP/1=IP per
_POSTFLOP_RANK; bet/raise frac is pot-fraction; check/call/fold/allin frac=None.
"""

from __future__ import annotations

from src.canonicalizer.encoder import _STREET_DELIM, _TOKEN_RE
from src.solver.range_resolver import _POSTFLOP_RANK

__all__ = ["MultiwayNavError", "parse_nav_line"]


class MultiwayNavError(ValueError):
    """Action line references a position outside the HU pair — a multiway flop the HU solver cannot model."""


_SIZE_FRAC: dict[str, float] = {
    "half_pot": 0.5,
    "pot": 1.0,
    "quarter_pot": 0.25,
    "third_pot": 0.33,
    "two_third_pot": 0.66,
}

_DEFAULT_FRAC = 1.0

_PASSIVE = {"call", "check", "fold", "allin"}


def _kind_for(verb: str, *, facing_bet: bool) -> str:
    """Map a verb to its postflop nav kind (raise/bet/call/check/fold/allin).

    SIM vocab is bet-agnostic: it emits "call" for a check and "raise_*" for a
    lead bet. The tree walker matches exact kinds, so map by street betting state.
    """
    v = verb.lower()
    if (
        v.startswith("open")
        or v.startswith("raise")
        or v.startswith("3b")
        or v.startswith("4b")
        or v.startswith("5b")
    ):
        return "raise" if facing_bet else "bet"
    if v.startswith("bet"):
        return "bet"
    if v == "call":
        return "call" if facing_bet else "check"
    if v in _PASSIVE:
        return v
    raise ValueError(f"unrecognized postflop verb: {verb!r}")


def _frac_for(kind: str, verb: str) -> float | None:
    """Pot-fraction for a bet/raise verb; None for passive actions."""
    if kind not in ("bet", "raise"):
        return None
    suffix = verb.lower().split("_", 1)[1] if "_" in verb else ""
    return _SIZE_FRAC.get(suffix, _DEFAULT_FRAC)


def parse_nav_line(
    action_sequence: list[str],
    hero_pos: str,
    villain_pos: str,
    board_cards: list[str],
) -> tuple[list[dict[str, object]], int, str | None, str | None]:
    """Parse a postflop action_sequence into (nav_steps, hero_player, turn_card, river_card).

    Higher _POSTFLOP_RANK = acts last = IP = seat 1. Only the HU pair is valid; any
    other token position raises ValueError. turn/river come from board indices 3/4.
    """
    hero_player = 1 if _POSTFLOP_RANK[hero_pos] >= _POSTFLOP_RANK[villain_pos] else 0
    villain_player = 1 - hero_player
    seat_by_pos = {hero_pos: hero_player, villain_pos: villain_player}

    segments: list[list[str]] = [[]]
    for tok in action_sequence:
        if tok == _STREET_DELIM:
            segments.append([])
        else:
            segments[-1].append(tok)

    postflop_segments = segments[1:]
    nav_steps: list[dict[str, object]] = []
    for segment in postflop_segments:
        facing_bet = False
        for tok in segment:
            m = _TOKEN_RE.match(tok)
            if not m:
                raise ValueError(f"malformed action token: {tok!r}")
            pos, verb = m.group(1), m.group(2)
            if pos not in seat_by_pos:
                raise MultiwayNavError(f"position {pos!r} not in HU pair {(hero_pos, villain_pos)!r}")
            kind = _kind_for(verb, facing_bet=facing_bet)
            if kind in ("bet", "raise", "allin"):
                facing_bet = True
            nav_steps.append({"seat": seat_by_pos[pos], "kind": kind, "frac": _frac_for(kind, verb)})

    turn_card = board_cards[3] if len(board_cards) >= 4 else None
    river_card = board_cards[4] if len(board_cards) >= 5 else None
    return nav_steps, hero_player, turn_card, river_card
