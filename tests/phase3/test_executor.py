"""Intent-aware sizing executor in SimAdapter (fixes C1/C3/H1/H3, observability)."""

from __future__ import annotations

from src.sim.adapter import SimAdapter


def _fresh_adapter(seed: int = 42) -> SimAdapter:
    adapter = SimAdapter()
    adapter._env.seed(seed)
    adapter._env.reset()
    return adapter


# --- adapter now uses the sized env (raise-TO capable) ------------------------


def test_adapter_uses_sized_env():
    adapter = SimAdapter()
    assert hasattr(adapter._env, "step_raise_to"), "adapter must drive the sized env"
    assert adapter._env.num_players == 6


# --- intent classification ---------------------------------------------------


def test_intent_class_aggressive_vs_passive():
    from src.sim.adapter import intent_class

    for verb in (
        "open_2_2bb",
        "open_3bb",
        "3bet_3x",
        "4bet_2_5x",
        "bet_50",
        "bet_overbet",
        "raise_min",
        "raise_3x",
        "raise_pot",
        "allin",
    ):
        assert intent_class(verb) == "aggressive", verb
    for verb in ("fold", "check", "call"):
        assert intent_class(verb) == "passive", verb


# --- C3: short-stack aggressive intent must NOT become a passive call ---------


def test_aggressive_fallback_prefers_all_in_over_check_call():
    """When the only legal raise is ALL_IN (RAISE_HALF_POT/RAISE_POT illegal), an
    aggressive verb must map to ALL_IN(4), never CHECK_CALL(1) (C3)."""
    adapter = SimAdapter()
    # legal = [FOLD, CHECK_CALL, ALL_IN] — no sized raise possible.
    for verb in ("open_3bb", "3bet_3x", "bet_75", "raise_3x", "allin"):
        mapped = adapter.map_to_rlcard_action(verb, legal_actions=[0, 1, 4])
        assert mapped == 4, f"{verb} must shove ALL_IN, got {mapped} (1=passive call)"


def test_passive_fallback_keeps_check_call():
    """Passive verbs still prefer CHECK_CALL when their exact action is illegal."""
    adapter = SimAdapter()
    assert adapter.map_to_rlcard_action("call", legal_actions=[0, 1, 4]) == 1
    assert adapter.map_to_rlcard_action("check", legal_actions=[0, 1, 4]) == 1
    # fold maps to fold when legal
    assert adapter.map_to_rlcard_action("fold", legal_actions=[0, 1, 4]) == 0


# --- H1: postflop small bet in a capped pot -> legal raise/all-in, not check --


def test_aggressive_fallback_prefers_largest_legal_raise():
    """If RAISE_HALF_POT is illegal but RAISE_POT is legal, an aggressive bet maps
    to a raise, never CHECK_CALL."""
    adapter = SimAdapter()
    mapped = adapter.map_to_rlcard_action("bet_25", legal_actions=[0, 1, 3])
    assert mapped == 3, f"bet should raise, got {mapped}"


# --- execute_action drives the env and reports divergence --------------------


def test_execute_action_returns_divergence_flag():
    adapter = _fresh_adapter()
    # First-in UTG: a sized open is achievable -> no divergence.
    result = adapter.execute_action("open_3bb")
    assert "diverged" in result
    assert "intent_class" in result
    assert result["intent_class"] == "aggressive"


def test_execute_action_open_realizes_three_bb_not_min_raise():
    """C1 end-to-end: open_3bb commits ~6 chips (3bb), not 3 chips (1.5bb)."""
    adapter = _fresh_adapter()
    gp = adapter._env.get_player_id()
    adapter.execute_action("open_3bb")
    assert adapter._env.game.players[gp].in_chips == 6


def test_execute_action_diverges_when_size_clamped_to_all_in():
    """M3: short stack where a 3bb open exceeds the stack -> clamped to all-in,
    counted as an aggressive divergence."""
    # 5-chip stacks (2.5bb): a 3bb=6-chip open can't be realized -> all-in clamp.
    adapter = SimAdapter({"game_num_players": 6, "chips_for_each": 5})
    adapter._env.seed(1)
    adapter._env.reset()
    before = adapter.divergence_counts()["aggressive"]
    result = adapter.execute_action("open_3bb")
    assert result["intent_class"] == "aggressive"
    assert result["diverged"] is True
    assert adapter.divergence_counts()["aggressive"] == before + 1


def test_execute_action_no_divergence_when_size_achievable():
    """A sized open that fits the stack is not a divergence."""
    adapter = _fresh_adapter()
    result = adapter.execute_action("open_3bb")
    assert result["diverged"] is False
    assert adapter.divergence_counts()["aggressive"] == 0


# --- (a) preflop first-in min-raise increment is the BB, not the SB/BB gap ----


def test_live_chip_state_preflop_first_in_last_raise_is_bb():
    """Preflop first-in, raised=[...,SB=1,BB=2]: the min legal raise increment is
    the BB (init_raise_amount=2), NOT the SB->BB blind gap (1). A gap of 1 yields
    an illegal 1.5bb min-raise-to."""
    adapter = _fresh_adapter()
    s = adapter._live_chip_state()
    assert s["last_raise_size"] == s["bb_chips"], (
        f"preflop first-in last_raise_size must be the BB ({s['bb_chips']}), "
        f"got {s['last_raise_size']} (blind gap)"
    )
    min_raise_to = s["max_raised"] + s["last_raise_size"]
    assert min_raise_to % s["bb_chips"] == 0, "min-raise-to must be a whole number of BB"


# --- (c) faithful all-in via the enum path is not a size divergence -----------


def test_execute_action_faithful_all_in_not_divergence():
    """Facing a bet where sized raises are illegal but ALL_IN is legal, an
    ``allin`` verb that shoves ALL_IN is faithful — not a size divergence."""
    # chips_for_each=30 with seed 0 reaches a spot with legal=[FOLD, CHECK_CALL,
    # ALL_IN] (no RAISE_HALF_POT/RAISE_POT) while facing a bet.
    adapter = SimAdapter({"game_num_players": 6, "chips_for_each": 30})
    adapter._env.seed(0)
    adapter._env.reset()
    for _ in range(12):
        if adapter._env.is_over():
            raise AssertionError("hand ended before reaching the facing-all-in spot")
        legal = adapter.current_legal_actions()
        if 4 in legal and 2 not in legal and 3 not in legal and 0 in legal:
            before = adapter.divergence_counts()["aggressive"]
            result = adapter.execute_action("allin")
            assert result["executed_action"] == 4, "must execute ALL_IN(4)"
            assert result["diverged"] is False, "faithful all-in is not a divergence"
            assert adapter.divergence_counts()["aggressive"] == before
            return
        if 3 in legal:
            adapter.execute_action("bet_100")
        else:
            adapter._env.step(1 if 1 in legal else legal[0])
    raise AssertionError("never reached a facing-all-in spot with ALL_IN legal")
