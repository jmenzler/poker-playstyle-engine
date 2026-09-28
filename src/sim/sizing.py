"""Per-verb raise-TO chip math for the sizing-aware sim executor.

Maps each aggressive verb to an intended raise-TO chip amount, clamped to
[min-raise, all-in]. ``max_raised`` is the largest street commitment so far.
"""

from __future__ import annotations

# Preflop opens: target raise-to in big blinds (first-in over the BB).
_OPEN_BB: dict[str, float] = {"open_2_2bb": 2.2, "open_3bb": 3.0}

# 3bet/4bet: multiplier applied to the facing raise-to (max_raised).
_RERAISE_MULT: dict[str, float] = {"3bet_3x": 3.0, "3bet_4x": 4.0, "4bet_2_5x": 2.5}

# Postflop bets with no facing bet: fraction of the pot.
_BET_PCT: dict[str, float] = {
    "bet_25": 0.25,
    "bet_33": 0.33,
    "bet_50": 0.50,
    "bet_75": 0.75,
    "bet_100": 1.00,
    "bet_150": 1.50,
    # overbet is the >=1.75x catch-all bucket (tools.action_buckets); size it
    # meaningfully larger than bet_150 so retrieved overbets keep their fidelity.
    "bet_overbet": 2.00,
}

# Postflop raises facing a bet: multiplier applied to the facing bet (max_raised).
_RAISE_MULT: dict[str, float] = {"raise_2_5x": 2.5, "raise_3x": 3.0}

# Verbs that never produce a sized raise (handled as plain enum actions).
_PASSIVE: frozenset[str] = frozenset({"fold", "check", "call"})


def intended_raise_to_chips(
    verb: str,
    *,
    pot: int,
    max_raised: int,
    my_raised: int,
    my_remained: int,
    bb_chips: int,
    init_raise_amount: int,
    last_raise_size: int | None = None,
) -> int | None:
    """Intended raise-TO chip target for ``verb``, clamped to [min-raise, all-in].

    Returns None for passive verbs (those drive a plain enum action).
    ``last_raise_size`` is the last bet/raise increment; defaults to the BB.
    """
    if verb in _PASSIVE:
        return None

    all_in = my_raised + my_remained
    lrs = last_raise_size if last_raise_size is not None else init_raise_amount
    min_raise_to = max_raised + lrs

    if verb == "allin":
        return all_in

    raw: float
    if verb in _OPEN_BB:
        raw = _OPEN_BB[verb] * bb_chips
    elif verb in _RERAISE_MULT:
        raw = _RERAISE_MULT[verb] * max_raised
    elif verb in _BET_PCT:
        raw = _BET_PCT[verb] * pot
    elif verb in _RAISE_MULT:
        raw = _RAISE_MULT[verb] * max_raised
    elif verb == "raise_min":
        raw = float(min_raise_to)
    elif verb == "raise_pot":
        # Pot-sized raise: call to match, then raise by the resulting pot.
        call_amount = max_raised - my_raised
        raw = max_raised + (pot + call_amount)
    else:
        # Unknown verb: treat as a min-raise rather than guessing a size.
        raw = float(min_raise_to)

    target = round(raw)
    # Clamp: never below a legal min-raise (a raise is never a limp), never above
    # all-in. min-raise itself is capped at all-in for very short stacks.
    target = max(target, min(min_raise_to, all_in))
    target = min(target, all_in)
    return target


def nominal_raise_to_chips(
    verb: str,
    *,
    pot: int,
    max_raised: int,
    my_raised: int,
    bb_chips: int,
) -> float | None:
    """The verb's nominal (pre-clamp) raise-TO chip target, for divergence checks.

    Returns the size the verb wants before [min-raise, all-in] clamping. None for
    passive verbs, ``allin``, and ``raise_min`` (no nominal-vs-realized gap).
    """
    if verb in _PASSIVE or verb == "allin":
        return None
    if verb in _OPEN_BB:
        return _OPEN_BB[verb] * bb_chips
    if verb in _RERAISE_MULT:
        return _RERAISE_MULT[verb] * max_raised
    if verb in _BET_PCT:
        return _BET_PCT[verb] * pot
    if verb in _RAISE_MULT:
        return _RAISE_MULT[verb] * max_raised
    if verb == "raise_pot":
        return float(max_raised + pot + (max_raised - my_raised))
    return None  # raise_min / unknown have no nominal target above min-raise
