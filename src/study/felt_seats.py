"""Reconstruct a god-view hand replay from SIM decision-point rows.

Ordered dps are the action timeline; each preflop dp reveals one player's cards, so
the dps yield one full timeline with every seat's cards known. felt_snapshot-only.
"""

from __future__ import annotations

import re
from typing import Any

# Clockwise 6-max seating ring. Seat numbers are 1-indexed; the frontend maps
# seat N to a ring slot via (N-1) % 6, so adjacency here is the table adjacency.
_RING: tuple[str, ...] = ("BTN", "SB", "BB", "UTG", "MP", "CO")
_SEAT_OF: dict[str, int] = {pos: i + 1 for i, pos in enumerate(_RING)}


def _dp_index(decision_id: object) -> int:
    """Integer dp index from a `{hand_id}_dp{idx}` decision_id (sort key)."""
    s = str(decision_id)
    if "_dp" in s:
        try:
            return int(s.rsplit("_dp", 1)[-1])
        except ValueError:
            return 1_000_000
    try:
        return int(s)
    except (TypeError, ValueError):
        return 1_000_000


def _is_fold(action: object) -> bool:
    return isinstance(action, str) and "fold" in action.lower()


_PREFLOP_RAISE_RE = re.compile(r"^(raise|open|\d+bet)")


def _preflop_raise_label(nth: int) -> str:
    """Poker-semantic label for the nth voluntary preflop raise: 1=open, 2=3bet, 3=4bet…"""
    return "open" if nth == 1 else f"{nth + 1}bet"


# Postflop sized vocab (mirrors src/sim/sizing.py). The sequence token is bet-agnostic,
# so the size is recovered from the action's real chip delta.
_BET_FRACS: tuple[tuple[str, float], ...] = (
    ("bet_25", 0.25),
    ("bet_33", 0.33),
    ("bet_50", 0.50),
    ("bet_75", 0.75),
    ("bet_100", 1.00),
    ("bet_150", 1.50),
)


def _bet_verb(delta: float, pot_before: float) -> str:
    """Sized lead-bet verb from chips put in vs the pot before the bet."""
    if pot_before <= 0 or delta <= 0:
        return "bet"
    frac = delta / pot_before
    if frac > 1.75:
        return "bet_overbet"
    return min(_BET_FRACS, key=lambda kv: abs(kv[1] - frac))[0]


_RAISE_TO_MULT: dict[str, float] = {
    "raise_min": 2.0,
    "raise_2_5x": 2.5,
    "raise_3x": 3.0,
}


def _raise_verb(raise_to: float, facing_to: float, pot_before: float, to_call: float) -> str:
    """Sized raise verb from the actual raise-TO vs the facing bet (min / 2.5x / 3x / pot)."""
    if facing_to <= 0:
        return "raise"
    candidates = {
        "raise_min": facing_to * 2.0,
        "raise_2_5x": facing_to * 2.5,
        "raise_3x": facing_to * 3.0,
        "raise_pot": facing_to + pot_before + to_call,
    }
    return min(candidates.items(), key=lambda kv: abs(kv[1] - raise_to))[0]


def _terminal_aggressive_delta(
    action: str, *, pot_before: float, to_call: float, facing_to: float, bet_before: float
) -> float:
    """Chip delta for the final aggressive action when no next dp exists to diff the
    pot. Recovers the bet/raise size from the sized token (mirrors src/sim/sizing.py);
    returns 0.0 only for genuinely unsized tokens."""
    a = action.lower()
    if to_call <= 1e-9:
        frac = dict(_BET_FRACS).get(a)
        if frac is not None and pot_before > 0:
            return frac * pot_before
        return 0.0
    mult = _RAISE_TO_MULT.get(a)
    if mult is not None and facing_to > 0:
        return max(0.0, facing_to * mult - bet_before)
    if a == "raise_pot" and facing_to > 0:
        return max(0.0, (facing_to + pot_before + to_call) - bet_before)
    return 0.0


def _postflop_verb(
    action: object, *, delta: float, pot_before: float, to_call: float, facing_to: float, bet_before: float
) -> object:
    """Translate the bet-agnostic postflop sequence token into the engine sized vocab
    using the action's real chip delta. call→check when unfaced; fold/allin pass through."""
    if not isinstance(action, str):
        return action
    a = action.lower()
    if a == "call":
        return "check" if to_call <= 1e-9 else "call"
    if not a.startswith(("raise", "bet", "open")):
        return action
    if to_call <= 1e-9:
        return _bet_verb(delta, pot_before)
    return _raise_verb(bet_before + delta, facing_to, pot_before, to_call)


def reconstruct_sim_replay(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn ordered SIM dp rows into a god-view replay timeline.

    Steps carry seat_map (all positions), seat_cards (cards known every step),
    actor, board, pot, button_seat, folded. Returns [] if no usable rows.
    """
    if not rows:
        return []

    ordered = sorted(rows, key=lambda r: _dp_index(r.get("decision_id")))

    # Harvest each seat's hole cards + the set of positions in the hand. First
    # occurrence wins (preflop dps reveal every seated player's cards).
    seat_cards: dict[str, list[str]] = {}
    positions: set[str] = set()
    stack_bb: float | None = None
    for row in ordered:
        felt = row.get("felt_snapshot") or {}
        pos = (felt.get("hero_position") or "").upper()
        if pos in _SEAT_OF:
            positions.add(pos)
            cards = felt.get("hero_hole_cards") or []
            if pos not in seat_cards and cards:
                seat_cards[pos] = list(cards)
        # Also seat anyone named in the action history (folders never get a dp).
        for tok in felt.get("action_sequence") or []:
            ap = tok.split(":", 1)[0].strip().upper()
            if ap in _SEAT_OF:
                positions.add(ap)
        if stack_bb is None and felt.get("effective_stack_bb") is not None:
            stack_bb = felt.get("effective_stack_bb")

    if not positions:
        return []
    # SIM is always 6-max; the walk winner and unacted blinds appear in no row, so seat
    # the full ring. Seats with no revealing dp carry no cards.
    positions = set(_RING)

    button_seat = _SEAT_OF["BTN"] if "BTN" in positions else None

    steps: list[dict[str, Any]] = []
    folded: list[str] = []
    committed: dict[str, float] = {}
    hand_total: dict[str, float] = {}
    street_base: dict[str, float] = {}
    prev_street: str | None = None
    preflop_raises = 0
    for i, row in enumerate(ordered):
        felt = row.get("felt_snapshot") or {}
        actor = (felt.get("hero_position") or "").upper() or None
        action = row.get("action_taken")
        street = felt.get("street", "")
        pot_now = felt.get("pot_size_bb")
        # hero_bet_size_bb is the actor's CUMULATIVE hand wager pre-action; the
        # street wager is its excess over the actor's total at street start.
        total_before = float(felt.get("hero_bet_size_bb") or 0.0)
        next_felt = (ordered[i + 1].get("felt_snapshot") or {}) if i + 1 < len(ordered) else {}

        # Executed verb = token the next dp appended for THIS actor (matched by
        # position); action_taken is a corpus-bucket label that can disagree with it.
        if actor:
            cur_seq = felt.get("action_sequence") or []
            appended = [t for t in (next_felt.get("action_sequence") or [])[len(cur_seq) :] if t and t != "/"]
            actor_tok = next((t for t in appended if t.split(":", 1)[0].strip().upper() == actor), None)
            if actor_tok and ":" in actor_tok:
                action = actor_tok.split(":", 1)[1]

        if street != prev_street:
            committed = {}
            street_base = dict(hand_total)
            prev_street = street
        bet_before = max(0.0, total_before - (street_base.get(actor, 0.0) if actor else 0.0))
        if actor:
            committed[actor] = bet_before

        to_call = max(committed.values(), default=0.0) - (committed.get(actor, 0.0) if actor else 0.0)
        facing_to = max(committed.values(), default=0.0)

        # Chip delta = pot rise to the next dp. The last dp has no next pot to diff,
        # so recover: call/allin from to-call, a sized bet/raise from its token, else 0.
        # From the RAW token, before relabeling (sized verb needs it).
        pot_next = next_felt.get("pot_size_bb")
        if pot_next is not None and pot_now is not None and float(pot_next) >= float(pot_now):
            delta = float(pot_next) - float(pot_now)
        elif isinstance(action, str) and action.lower() in ("call", "allin"):
            delta = to_call
        elif isinstance(action, str) and action.lower().startswith(("raise", "bet", "open")):
            delta = _terminal_aggressive_delta(
                action,
                pot_before=float(pot_now or 0.0),
                to_call=to_call,
                facing_to=facing_to,
                bet_before=bet_before,
            )
        else:
            delta = 0.0

        # Preflop raises read as open/3bet/4bet by raise count; postflop bets/raises
        # reconstruct the sized vocab from the chip delta (both replace the raw token).
        if street == "preflop":
            if isinstance(action, str) and _PREFLOP_RAISE_RE.match(action.lower()):
                preflop_raises += 1
                action = _preflop_raise_label(preflop_raises)
        else:
            action = _postflop_verb(
                action,
                delta=delta,
                pot_before=float(pot_now or 0.0),
                to_call=to_call,
                facing_to=facing_to,
                bet_before=bet_before,
            )

        bet_after = bet_before + delta
        if actor:
            committed[actor] = bet_after
            hand_total[actor] = total_before + delta

        # Per-step stacks: start-of-hand stack minus each seat's cumulative wager so
        # far. A single shared seat_map froze every stack at the starting value.
        seat_map_step = {
            str(_SEAT_OF[pos]): {
                "name": pos,
                "stack": round(stack_bb - hand_total.get(pos, 0.0), 2) if stack_bb is not None else None,
            }
            for pos in _RING
            if pos in positions
        }

        steps.append(
            {
                "step_idx": i,
                "street": street,
                "board": list(felt.get("board_cards", [])),
                "pot": (float(pot_now) + delta) if pot_now is not None else pot_now,
                # Pre-action pot, so a decision view (gap-resolver) shows the state the
                # actor faces, not the post-action pot that leaks the action taken.
                "pot_before": float(pot_now) if pot_now is not None else None,
                "big_blind": 1.0,
                "actor": actor,
                "action_taken": action,
                "bet_amount": bet_after,
                "seat_map": seat_map_step,
                "seat_cards": dict(seat_cards),
                "button_seat": button_seat,
                "folded": list(folded),
                "street_committed": dict(committed),
                "decision_id": row.get("decision_id"),
            }
        )
        # Fold accumulates AFTER the step so the folding seat still shows as the
        # actor on its own fold step, then drops on subsequent steps.
        if actor and _is_fold(action):
            folded.append(actor)

    return steps
