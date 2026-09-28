# rot-allow-file
# Regression guard: these tests fail against the old hardcoded-None adapter (adapter.py:332).
# They go GREEN only after Task 3 lands the rolling prior_street_aggressor derivation.

from __future__ import annotations

from src.sim.adapter import SimAdapter


def _drive_hand_to_end(adapter: SimAdapter, raise_once_preflop: bool = False) -> list[tuple]:
    """Drive one hand to completion; return list of (gs, legal) per decision point.

    When raise_once_preflop=True, the first opportunity to raise on preflop is taken,
    then all subsequent preflop and postflop actions prefer CHECK_CALL to keep the hand
    alive without stacking off.
    """
    dps: list[tuple] = []
    raised = False
    while adapter.has_more():
        gs = adapter.next_game_state()
        legal = adapter.current_legal_actions()
        dps.append((gs, legal))
        if not legal:
            break
        if raise_once_preflop and gs.street == "preflop" and not raised and 2 in legal:
            adapter._env.step(2)  # RAISE_HALF_POT
            raised = True
        else:
            act = 1 if 1 in legal else (0 if 0 in legal else legal[0])
            adapter._env.step(act)
    return dps


def _find_seed_with_postflop_raise(search_range: range) -> tuple[int, list[tuple]]:
    """Find a seed producing a hand that reaches postflop after a preflop raise.

    Returns (seed, dps) where dps contains at least one postflop decision point
    and the preflop action_sequence contains at least one raise verb.
    Fails loudly if no seed in the range produces such a hand.
    """
    adapter = SimAdapter()
    for seed in search_range:
        adapter._env.seed(seed)
        adapter._env.reset()
        dps = _drive_hand_to_end(adapter, raise_once_preflop=True)
        postflop_dps = [gs for gs, _ in dps if gs.street in ("flop", "turn", "river")]
        if not postflop_dps:
            continue
        # The preflop portion of the final action_sequence will contain all preflop tokens
        preflop_toks = [t for gs, _ in dps if gs.street == "preflop" for t in gs.action_sequence]
        has_raise = any("raise" in t or "allin" in t for t in preflop_toks)
        if has_raise:
            return seed, dps
    raise AssertionError(
        f"No seed in {search_range} produced a postflop hand with a preflop raise. "
        "Extend the search range or check the adapter's action-preference logic."
    )


def test_prior_street_aggressor_from_real_env() -> None:
    """HIST-01/HIST-02: adapter must populate prior_street_aggressor on postflop decisions.

    Boots a real seeded RLCard env, drives a hand with a preflop raise to postflop,
    asserts the first postflop GameState has prior_street_aggressor is not None.
    """
    seed, dps = _find_seed_with_postflop_raise(range(51))
    postflop_dps = [(gs, legal) for gs, legal in dps if gs.street in ("flop", "turn", "river")]
    assert postflop_dps, f"Seed {seed} produced no postflop decision points — unexpected after _find_seed."
    first_postflop_gs = postflop_dps[0][0]
    assert first_postflop_gs.prior_street_aggressor is not None, (
        f"HIST-01 violation: prior_street_aggressor is None on first postflop dp "
        f"(street={first_postflop_gs.street!r}, seed={seed}). "
        "Adapter hardcodes None — fix rolling aggressor derivation (Task 3)."
    )


def test_action_sequence_full_history() -> None:
    """HIST-03: every dp's action_sequence is a tuple and len grows monotonically."""
    adapter = SimAdapter()
    adapter._env.seed(0)
    adapter._env.reset()
    dps = _drive_hand_to_end(adapter, raise_once_preflop=False)
    assert dps, "No decision points produced — check adapter or seed."
    prev_len = -1
    for i, (gs, _) in enumerate(dps):
        assert isinstance(gs.action_sequence, tuple), (
            f"dp[{i}] action_sequence is not a tuple: {type(gs.action_sequence)}"
        )
        cur_len = len(gs.action_sequence)
        assert cur_len >= prev_len, (
            f"HIST-03 violation: action_sequence shrank at dp[{i}]: "
            f"prev={prev_len}, cur={cur_len}. Sequence must grow monotonically."
        )
        prev_len = cur_len


def test_preflop_aggressor_is_none() -> None:
    """HIST: on a preflop decision point, prior_street_aggressor must be None."""
    adapter = SimAdapter()
    adapter._env.seed(0)
    adapter._env.reset()
    assert adapter.has_more(), "Hand did not start — check seed/reset."
    gs = adapter.next_game_state()
    assert gs.street == "preflop", f"First dp is not preflop: {gs.street!r}. Adjust seed if needed."
    assert gs.prior_street_aggressor is None, (
        f"prior_street_aggressor must be None on a preflop dp, got {gs.prior_street_aggressor!r}."
    )
