"""Eval opponent Strategy protocol (D-NEW-30).

The 7 baselines in src/eval/baselines/ all conform to this protocol. The existing
``KNNDecisionEngine`` is also a valid Strategy — Eval matches "engine vs baseline"
use the same ``env.set_agents([engine, opponent])`` call.

RLCard 1.2.0 no-limit-holdem action set (mirrors src/sim/adapter.py PROJECT_TO_RLCARD):

    FOLD       = 0
    CHECK_CALL = 1
    HALF_POT   = 2
    POT        = 3
    ALL_IN     = 4

Fallback preference order (Phase 3 lock, src/sim/adapter.py line 56):
    (1, 2, 0, 4, 3)  # CHECK_CALL > HALF_POT > FOLD > ALL_IN > POT

This module is leaf — no DB, no subprocess, no SSE. Safe to import anywhere.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# RLCard action constants — exposed for readability inside baselines.
FOLD: int = 0
CHECK_CALL: int = 1
HALF_POT: int = 2
POT: int = 3
ALL_IN: int = 4

# Fallback preference order when a baseline's desired action is illegal.
# Copied verbatim from src/sim/adapter.py: CHECK_CALL > HALF_POT > FOLD > ALL_IN > POT.
FALLBACK_PREFERENCE: tuple[int, ...] = (CHECK_CALL, HALF_POT, FOLD, ALL_IN, POT)


@runtime_checkable
class Strategy(Protocol):
    """Conformance contract for any opponent in run_match().

    Implementations may carry internal state (e.g. seeded RNG, snapshot version);
    but ``decide`` itself must be side-effect-free with respect to the game state.
    The runtime-checkable Protocol mirrors src/protocols/auto_loop.py's style.
    """

    def decide(self, state: dict[str, Any]) -> int:
        """Return an RLCard action int (0-4) for the given env state.

        Args:
            state: ``env.get_state(player_id)`` output. Has ``'legal_actions'`` key
                with either ints or RLCard ``Action`` enums (Pitfall 5);
                implementations MUST handle both via ``coerce_action``.

        Returns:
            int in ``{0, 1, 2, 3, 4}`` representing
            FOLD / CHECK_CALL / HALF_POT / POT / ALL_IN.
            MUST be a member of ``state['legal_actions']`` after coercion.
        """
        ...


def coerce_action(raw_action: Any) -> int:
    """RLCard ``legal_actions`` may contain ints or ``Action`` enums. Return the int.

    RLCard 1.2.0 emits ``Action`` enums (which expose ``.value``); earlier versions
    return plain ints. Centralising the coercion here keeps every baseline robust.
    """
    if isinstance(raw_action, int):
        return raw_action
    if hasattr(raw_action, "value"):
        return int(raw_action.value)
    return int(raw_action)


def pick_first_legal(state: dict[str, Any], desired: tuple[int, ...]) -> int:
    """Return the first action from ``desired`` that is in ``state['legal_actions']``.

    Args:
        state: RLCard per-player state dict; must contain ``'legal_actions'``.
        desired: preference order of actions (most-preferred first).

    Returns:
        The first preferred action that is legal; falls back to FALLBACK_PREFERENCE
        if none of the desired actions are legal.

    Raises:
        RuntimeError: if no legal action is found at all (shouldn't happen — RLCard
            always exposes ≥1 legal action at a decision point).
    """
    legal = {coerce_action(a) for a in state.get("legal_actions", [])}
    for action in desired:
        if action in legal:
            return action
    for action in FALLBACK_PREFERENCE:
        if action in legal:
            return action
    raise RuntimeError(f"no legal action available; state['legal_actions']={state.get('legal_actions')!r}")
