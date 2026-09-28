"""Per-verb raise-TO chip math for the sizing-aware executor.

Pure unit tests over ``intended_raise_to_chips`` — no env needed. Chip units:
bb_chips=2 (RLCard NLHE), init_raise_amount=2 (one big blind).
"""

from __future__ import annotations

import pytest

from src.sim.sizing import intended_raise_to_chips, nominal_raise_to_chips

# Common preflop first-in context: pot = SB(1)+BB(2) = 3, max_raised = BB = 2,
# hero hasn't acted (my_raised=0), deep stack.
PREFLOP_FIRST_IN = dict(pot=3, max_raised=2, my_raised=0, my_remained=200, bb_chips=2, init_raise_amount=2)


def _call(verb: str, **overrides) -> int | None:
    ctx = {**PREFLOP_FIRST_IN, **overrides}
    return intended_raise_to_chips(verb, **ctx)


# --- passive verbs return None (no sized raise) -------------------------------


@pytest.mark.parametrize("verb", ["fold", "check", "call"])
def test_passive_verbs_return_none(verb: str):
    assert _call(verb) is None


# --- preflop opens (C1: must be 2.2/3bb, not 1.5bb) ---------------------------


def test_open_2_2bb_raises_to_4_chips():
    # 2.2bb * 2 chips/bb = 4.4 -> rounds to 4 chips (>= min-raise 4). Realized 2.2bb.
    assert _call("open_2_2bb") == 4


def test_open_3bb_raises_to_6_chips():
    assert _call("open_3bb") == 6  # 3bb * 2 = 6 chips


def test_open_never_below_min_raise():
    # min raise-to = max_raised(10) + init_raise_amount(2) = 12; a 2.2bb (=4.4)
    # open must clamp UP to 12, never limp.
    val = _call("open_2_2bb", max_raised=10, my_raised=0)
    assert val == 12


# --- 3bet / 4bet multipliers of the facing raise-to ---------------------------


def test_3bet_3x_is_three_times_facing_raise_to():
    # Facing a 6-chip open (max_raised=6). 3bet_3x -> raise-to 18 chips.
    assert _call("3bet_3x", pot=9, max_raised=6, my_raised=0) == 18


def test_3bet_4x_is_four_times_facing_raise_to():
    assert _call("3bet_4x", pot=9, max_raised=6, my_raised=0) == 24


def test_4bet_2_5x_is_2_5_times_facing_raise_to():
    # Facing an 18-chip 3bet. 4bet_2_5x -> raise-to 45 chips.
    assert _call("4bet_2_5x", pot=27, max_raised=18, my_raised=0) == 45


# --- postflop pct-of-pot bets (H1/H2: 8 sizes must be distinct) ---------------

# Postflop, no facing bet: max_raised = 0, pot = 20 chips (10bb).
POSTFLOP = dict(pot=20, max_raised=0, my_raised=0, my_remained=200, bb_chips=2, init_raise_amount=2)


def _pf(verb: str, **ov) -> int | None:
    return intended_raise_to_chips(verb, **{**POSTFLOP, **ov})


def test_bet_50_is_half_pot():
    assert _pf("bet_50") == 10  # 50% of 20-chip pot


def test_bet_25_33_75_100_150_are_distinct_and_correct():
    assert _pf("bet_25") == 5
    assert _pf("bet_33") == 7  # round(0.33*20)=7 (M1: not floored to 6)
    assert _pf("bet_75") == 15
    assert _pf("bet_100") == 20
    assert _pf("bet_150") == 30
    # All six pct sizes distinct (H2: no collapse to two realized sizes).
    sizes = {_pf(v) for v in ("bet_25", "bet_33", "bet_50", "bet_75", "bet_100", "bet_150")}
    assert len(sizes) == 6


def test_bet_overbet_is_larger_than_bet_150_not_all_in():
    # overbet must be strictly larger than a 1.5x pot bet, but never a full-stack jam.
    val = _pf("bet_overbet")
    assert val > _pf("bet_150")
    assert val < 200  # not a full-stack shove


# --- postflop raises facing a bet --------------------------------------------


def test_raise_min_is_min_legal_raise_to():
    # Facing a 10-chip bet (max_raised=10, last_raise_size=10). min raise-to =
    # max_raised + last_raise_size = 20.
    val = _pf("raise_min", pot=30, max_raised=10, my_raised=0, last_raise_size=10)
    assert val == 20


def test_raise_2_5x_and_3x_multiply_facing_bet():
    # Facing a 10-chip bet. raise_2_5x -> 25, raise_3x -> 30 (raise-to).
    assert _pf("raise_2_5x", pot=30, max_raised=10, my_raised=0, last_raise_size=10) == 25
    assert _pf("raise_3x", pot=30, max_raised=10, my_raised=0, last_raise_size=10) == 30


def test_raise_pot_is_pot_sized_raise_to():
    # Pot-sized raise-to = max_raised + (pot + call) = 10 + (30 + 10) = 50.
    assert _pf("raise_pot", pot=30, max_raised=10, my_raised=0, last_raise_size=10) == 50


# --- clamping ----------------------------------------------------------------


def test_target_above_stack_clamps_to_all_in():
    # 3bb open but only 5 chips behind -> all-in at my_raised + my_remained.
    val = _call("open_3bb", my_remained=5, my_raised=0)
    assert val == 5  # all-in raise-to = 0 + 5


def test_allin_verb_returns_full_stack_raise_to():
    val = _call("allin", my_remained=200, my_raised=0)
    assert val == 200


# --- nominal (pre-clamp) target for divergence checks ------------------------


def test_nominal_ignores_clamping():
    # open_3bb nominal is 6 chips regardless of a tiny stack (intended_raise_to
    # would clamp to all-in; nominal does not).
    assert nominal_raise_to_chips("open_3bb", pot=3, max_raised=2, my_raised=0, bb_chips=2) == 6.0


def test_nominal_none_for_passive_and_allin_and_raise_min():
    for verb in ("fold", "check", "call", "allin", "raise_min"):
        assert nominal_raise_to_chips(verb, pot=20, max_raised=10, my_raised=0, bb_chips=2) is None
