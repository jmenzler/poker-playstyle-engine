"""Sized RLCard env layer — raise-TO semantics for the sizing-aware executor."""

from __future__ import annotations

from rlcard.games.nolimitholdem.round import Action


def _fresh_env(seed: int = 42):
    from src.sim.sized_env import make_sized_env

    env = make_sized_env({"game_num_players": 6, "chips_for_each": 200})
    env.seed(seed)
    env.reset()
    return env


def test_make_sized_env_is_six_max_200_chips():
    env = _fresh_env()
    assert env.num_players == 6
    assert env.game.init_chips == [200] * 6


def test_min_and_max_raise_to_first_in_preflop():
    """First-in UTG (seed 42): BB=2 chips so min legal raise-to = 4 chips (2bb);
    all-in raise-to = my current raised + remained stack."""
    env = _fresh_env()
    rnd = env.game.round
    gp = env.game.game_pointer
    # min raise-to = max(raised) + last raise size (init_raise_amount = BB = 2)
    assert rnd.min_raise_to() == 4
    # all-in raise-to for the actor = raised[gp] + remained
    expected_allin = rnd.raised[gp] + env.game.players[gp].remained_chips
    assert rnd.max_raise_to() == expected_allin


def test_proceed_round_raise_to_sets_total_in_chips():
    """raise-TO 6 chips (3bb open): actor's in_chips for the street becomes exactly 6."""
    env = _fresh_env()
    gp = env.game.game_pointer
    before = env.game.players[gp].in_chips
    assert before == 0  # UTG first-in
    env.step_raise_to(6)
    # The actor who acted now has in_chips == 6 (raise-TO target, not +6 on top of pot).
    assert env.game.players[gp].in_chips == 6
    assert env.game.players[gp].remained_chips == 200 - 6


def test_step_raise_to_records_aggressive_action():
    """A sized raise must appear in action_recorder as an aggressive verb so
    pot_type/aggressor inference still sees it as a raise (not a call)."""
    env = _fresh_env()
    gp = env.game.game_pointer
    env.step_raise_to(6)
    last_pid, last_action = env.action_recorder[-1]
    assert last_pid == gp
    assert last_action in (Action.RAISE_POT, Action.RAISE_HALF_POT, Action.ALL_IN)


def test_proceed_round_raise_to_clamps_to_all_in():
    """A raise-to above the actor's stack becomes all-in (never negative stake)."""
    env = _fresh_env()
    gp = env.game.game_pointer
    huge = 10_000
    env.step_raise_to(huge)
    assert env.game.players[gp].remained_chips == 0
    assert env.game.players[gp].in_chips == 200  # full 200-chip stack committed


def test_step_raise_to_advances_to_next_player():
    """After a sized raise the game pointer moves to the next live player."""
    env = _fresh_env()
    gp_before = env.game.game_pointer
    env.step_raise_to(6)
    assert env.game.game_pointer != gp_before


def test_non_raising_target_accumulates_not_raise_num():
    """A raise-TO that does not exceed the current max commitment is a call, not a
    raise: it must increment not_raise_num (round-over accounting), not reset it to 1."""
    env = _fresh_env()
    rnd = env.game.round
    rnd.not_raise_num = 2
    # Target equals the existing max commitment (BB) -> a call, not a raise.
    target = max(rnd.raised)
    rnd.proceed_round_raise_to(env.game.players, target_raised=target)
    assert rnd.not_raise_num == 3


def test_strict_raise_resets_not_raise_num():
    """A raise-TO strictly above the current max commitment resets not_raise_num to 1."""
    env = _fresh_env()
    rnd = env.game.round
    rnd.not_raise_num = 3
    rnd.proceed_round_raise_to(env.game.players, target_raised=max(rnd.raised) + 4)
    assert rnd.not_raise_num == 1
