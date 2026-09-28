"""Shared position helpers used by both build (extract_decisions) and serve (encoder).

Keeping this in one place prevents preflop hero_pos_rel from drifting between the
trained index and the live decision path.
"""

from __future__ import annotations

from collections.abc import Callable

# Postflop action order: SB acts first, BTN acts last.
_POSTFLOP_RANK: dict[str, int] = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}

# Seats with guaranteed postflop position when there is no raiser to anchor against.
_LATE_SEATS: frozenset[str] = frozenset({"CO", "BTN"})

# Preflop raise count → pot_type label. Opens count as the first raise.
_POT_TYPE_BY_RAISES: dict[int, str] = {0: "limp", 1: "srp", 2: "3bet", 3: "4bet"}


def preflop_pot_type(actions: list[dict]) -> str:  # type: ignore[type-arg]
    """pot_type from the raises hero faces — the live state at the decision.

    Counts raises/all-ins in the actions preceding hero's move (4+ → 5bet+).
    Accepts both the build-side ("action") and serve-side parsed-token shapes.
    """
    raises = sum(1 for a in actions if a.get("action") in ("raise", "allin"))
    return _POT_TYPE_BY_RAISES.get(raises, "5bet+")


def preflop_pos_rel(hero_pos: str, aggressor_pos: str | None) -> str:
    """IP iff hero acts after the preflop aggressor postflop; seat fallback (CO/BTN) otherwise."""
    if aggressor_pos and aggressor_pos != hero_pos and aggressor_pos in _POSTFLOP_RANK:
        return "IP" if _POSTFLOP_RANK.get(hero_pos, -1) > _POSTFLOP_RANK[aggressor_pos] else "OOP"
    return "IP" if hero_pos in _LATE_SEATS else "OOP"


def last_raiser_pos(actions: list[dict]) -> str | None:  # type: ignore[type-arg]
    """Seat of the last raise/all-in in an action list, else None.

    Accepts both the build-side event shape ("actor_pos") and the serve-side
    parsed-token shape ("pos").
    """
    for a in reversed(actions):
        if a.get("action") in ("raise", "allin", "bet"):
            return a.get("actor_pos") or a.get("pos")
    return None


# last_raiser_pos matches stems only; normalize verbose verbs before delegating.
_VERB_TO_STEM: dict[str, str] = {
    "fold": "fold",
    "call": "call",
    "check": "call",
    "raise_half_pot": "raise",
    "raise_pot": "raise",
    "allin": "allin",
    "bet": "raise",
}


def aggressor_for_prior_street(
    action_slice: list[tuple],
    dealer_id: int,
    hero_position_fn: Callable[[int, int], str],
    verb_map: dict,
) -> str | None:
    """Return the seat of the last aggressor in an action_recorder slice.

    Converts (player_id, Action) tuples to the dict shape expected by
    last_raiser_pos(), normalizing verbose verbs to the stems it matches.
    tools/ has no dependency on src.sim — caller injects _hero_position and
    _ACTION_TO_VERB from the adapter.
    """
    converted = [
        {
            "pos": hero_position_fn(int(pid), dealer_id),
            "action": _VERB_TO_STEM.get(verb_map.get(action, "call"), "call"),
        }
        for pid, action in action_slice
    ]
    return last_raiser_pos(converted)
