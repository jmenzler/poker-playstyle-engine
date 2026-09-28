"""Phase 3 Plan 03 contract tests — SimAdapter wraps RLCard for ENGN-06."""

from __future__ import annotations


def test_six_player_game_init():
    """ENGN-06: SimAdapter creates a 6-player no-limit-holdem env (not 2-player)."""
    from src.sim.adapter import SimAdapter

    adapter = SimAdapter()
    assert adapter._env.num_players == 6, (
        f"RLCard config bug: expected 6 players, got {adapter._env.num_players}. "
        f"Check game_num_players config key (RESEARCH.md Pitfall 2)."
    )


def test_game_state_all_streets():
    """ENGN-06: next_game_state() produces a valid GameState across all four streets."""
    from src.protocols.game_state import GameState
    from src.sim.adapter import SimAdapter

    adapter = SimAdapter()
    adapter._env.seed(42)
    adapter._env.reset()
    streets_seen = set()
    # Up to 60 env.step() iterations is more than enough for a 6-player full hand.
    for _ in range(60):
        if not adapter.has_more():
            break
        gs = adapter.next_game_state()
        assert isinstance(gs, GameState)
        assert gs.street in ("preflop", "flop", "turn", "river")
        streets_seen.add(gs.street)
        # Apply a default fold action to advance the env (legal in every preflop step).
        legal = adapter.current_legal_actions()
        # Prefer CHECK_CALL when available so the hand progresses beyond preflop.
        act = 1 if 1 in legal else (legal[0] if legal else 0)
        adapter._env.step(act)
    # For Plan 03 this test asserts only that at least 'preflop' is reached.
    # Plan 04's harness integration test will cover all four streets via full hand play.
    assert "preflop" in streets_seen


def test_map_to_rlcard_action_legal_fallback():
    """ENGN-06: map_to_rlcard_action falls back to a legal action when mapped is blocked."""
    from src.sim.adapter import SimAdapter

    adapter = SimAdapter()
    # ALL_IN (4) blocked: only [0=FOLD, 1=CHECK_CALL, 2=HALF_POT] legal.
    # Engine outputs 'allin' which maps to 4. Adapter must pick the nearest legal.
    result = adapter.map_to_rlcard_action("allin", legal_actions=[0, 1, 2])
    assert result in (0, 1, 2), f"fallback returned illegal action: {result}"
    # CHECK_CALL preferred over FOLD by fallback order
    result2 = adapter.map_to_rlcard_action("bet_75", legal_actions=[0, 1])
    assert result2 == 1, f"expected CHECK_CALL fallback, got {result2}"


def test_preflop_open_executes_as_raise_not_limp():
    """First-in preflop RLCard offers [FOLD, CHECK_CALL, RAISE_POT, ALL_IN] — no
    RAISE_HALF_POT(2). Opens must map to a legal raise (RAISE_POT=3), not fall back
    to CHECK_CALL(1) and silently limp."""
    from src.sim.adapter import SimAdapter

    adapter = SimAdapter()
    first_in_legal = [0, 1, 3, 4]
    for verb in ("open_2_2bb", "open_3bb"):
        mapped = adapter.map_to_rlcard_action(verb, legal_actions=first_in_legal)
        assert mapped == 3, f"{verb} should open as RAISE_POT(3) first-in, got {mapped} (1=limp)"


def test_effective_stack_excludes_folded_players():
    """effective_stack_bb is the min REMAINING stack across players still in the
    hand. A folded player keeps its remaining chips (stakes[i]>0) but must NOT
    drag the effective-stack min down — that understates SPR."""
    from rlcard.games.limitholdem import PlayerStatus

    from src.sim.adapter import SimAdapter

    adapter = SimAdapter({"game_num_players": 6, "chips_for_each": 200})
    adapter._env.seed(1)
    adapter._env.reset()
    players = adapter._env.game.players
    pid = adapter._env.get_player_id()
    victim = (pid + 1) % 6
    # Simulate a player who committed chips then folded: short remaining stack.
    players[victim].status = PlayerStatus.FOLDED
    players[victim].remained_chips = 10  # 5bb — far below the in-hand stacks

    gs = adapter.next_game_state()
    assert gs.effective_stack_bb >= 90.0, (
        f"folded short stack leaked into effective-stack min: {gs.effective_stack_bb}bb "
        f"(expected ~99bb from in-hand players)"
    )


# --- ObservationWriter tests (Task 2) -----------------------------------------


def test_observation_writer_buffers_until_batch_size():
    """SIM-03: Writer holds rows in memory until batch_size reached; flush invokes executemany."""
    from unittest.mock import MagicMock

    from src.sim.observation_writer import ObservationWriter

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    writer = ObservationWriter(dsn="ignored", session_id="test-sess", batch_size=3, _conn=mock_conn)
    writer.record("clA", [0.1] * 80, "call")
    writer.record("clB", [0.2] * 80, "fold")
    assert mock_cursor.executemany.call_count == 0, "should not flush before batch_size"

    writer.record("clC", [0.3] * 80, "raise_3x")
    assert mock_cursor.executemany.call_count == 1
    assert writer.total_written == 3

    # Inspect the data passed
    call_args = mock_cursor.executemany.call_args
    sql, rows = call_args[0]
    assert "INSERT INTO observations" in sql
    assert len(rows) == 3
    # Each row: (obs_id, cluster_key, embedding, action_taken, source, session_id, ts)
    assert rows[0][1] == "clA"
    assert rows[0][3] == "call"
    assert rows[0][4] == "sim"  # source hardcoded
    assert rows[0][5] == "test-sess"


def test_observation_writer_close_flushes_tail_rows():
    """SIM-03: close() flushes any rows in the buffer below batch_size."""
    from unittest.mock import MagicMock

    from src.sim.observation_writer import ObservationWriter

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    writer = ObservationWriter(dsn="ignored", session_id="test", batch_size=10, _conn=mock_conn)
    writer.record("c1", [0.0] * 80, "check")
    writer.record("c2", [0.0] * 80, "call")
    assert mock_cursor.executemany.call_count == 0  # below batch_size

    total = writer.close()
    assert total == 2
    assert mock_cursor.executemany.call_count == 1
    mock_conn.close.assert_called_once()


# --- SimHarness tests (Plan 04 Task 2) ----------------------------------------


def test_harness_calls_seed_before_reset():
    """Pitfall 1 contract: env.seed() MUST be called immediately before env.reset().

    Uses a mock adapter + mock engine to verify the harness invokes
    ``adapter._env.seed(per_hand_seed)`` immediately before
    ``adapter._env.reset()`` with no other ``adapter._env.*`` calls between
    them. This is the Mac-friendly proof of the RESEARCH.md Pitfall 1 fix.
    """
    from unittest.mock import MagicMock

    import numpy as np

    from src.canonicalizer import EncodeResult
    from src.protocols.game_state import GameState
    from src.sim.harness import run_record_session

    # Mock adapter with tracked _env call sequence.
    mock_adapter = MagicMock()
    mock_adapter._env = MagicMock()
    # has_more: True once per hand, then False (1 decision per hand x 2 hands).
    mock_adapter.has_more.side_effect = [True, False, True, False]
    # current_legal_actions returns 3 legal RLCard actions.
    mock_adapter.current_legal_actions.return_value = [0, 1, 2]
    # next_game_state returns a real GameState.
    mock_adapter.next_game_state.return_value = GameState(
        street="preflop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=(),
        pot_size_bb=3.0,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=2.5,
        hero_bet_size_bb=0.0,
        action_sequence=(),
        opponents_remaining=2,
        prior_street_aggressor=None,
    )
    mock_adapter.map_to_rlcard_action.return_value = 0  # FOLD

    # Mock engine: decide_with_encoding returns (action, flagged_sparse, max_neighbor_distance, EncodeResult).
    mock_engine = MagicMock()
    mock_enc = EncodeResult(
        embedding=np.zeros(32),
        hard_filter={
            "street_class": "preflop",
            "pot_type": "srp",
            "hero_pos_rel": "IP",
            "n_players_active": 2,
        },
        schema_version=2,
    )
    mock_engine._canon.encode.return_value = mock_enc
    mock_engine.decide_with_encoding.return_value = ("fold", False, None, mock_enc)
    mock_engine.decide.return_value = "fold"

    mock_writer = MagicMock()

    # Track ordered calls on adapter._env (only seed/reset/step are interesting;
    # has_more/get_player_id/get_state are not part of the Pitfall-1 invariant).
    call_order: list[tuple[str, tuple, dict]] = []

    def _track(name):
        def _t(*a, **k):
            call_order.append((name, a, k))
            return MagicMock()

        return _t

    mock_adapter._env.seed.side_effect = _track("seed")
    mock_adapter._env.reset.side_effect = _track("reset")
    mock_adapter._env.step.side_effect = _track("step")

    run_record_session(
        mock_adapter,
        mock_engine,
        mock_writer,
        session_seed=42,
        n_hands=2,
    )

    # Pitfall 1: every seed() MUST be IMMEDIATELY followed by reset()
    # (no other adapter._env.* call in between).
    names = [c[0] for c in call_order]
    # Expect at least one (seed, reset) pair per hand.
    assert names.count("seed") == 2, f"expected 2 seed() calls, got {names.count('seed')}: {names}"
    assert names.count("reset") == 2, f"expected 2 reset() calls, got {names.count('reset')}: {names}"

    # For every seed at index i, the next adapter._env call (index i+1) MUST be reset.
    for i, name in enumerate(names):
        if name == "seed":
            assert i + 1 < len(names), f"seed at end of trace with no reset: {names}"
            assert names[i + 1] == "reset", (
                f"Pitfall 1 violation: seed at index {i} not followed by reset; "
                f"got {names[i + 1]!r} instead. Full sequence: {names}"
            )
