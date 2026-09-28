"""Phase 10 Plan 02: harness hand_id/decision_id minting + felt pass-through.

Tests verify D-05: harness mints hand_id={session_id}_h{hand_idx} and
decision_id={hand_id}_dp{decisions_this_hand}, and passes msgspec-decoded
GameState as felt_snapshot to writer.record().

Uses a spy writer (captured kwargs) and a mock adapter + engine to avoid
requiring live Milvus or TimescaleDB connections.
"""

from __future__ import annotations

from unittest.mock import MagicMock

_GAMESTATE_FIELDS = {
    "street",
    "hero_position",
    "hero_hole_cards",
    "board_cards",
    "pot_size_bb",
    "effective_stack_bb",
    "hero_facing_bet_bb",
    "hero_bet_size_bb",
    "action_sequence",
    "opponents_remaining",
    "prior_street_aggressor",
}


def _make_mock_game_state():
    """Build a minimal mock GameState with all 11 required fields."""
    from src.protocols.game_state import GameState

    return GameState(
        street="flop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=("2c", "7s", "Jh"),
        pot_size_bb=4.5,
        effective_stack_bb=97.5,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("fold", "call"),
        opponents_remaining=1,
        prior_street_aggressor="CO",
    )


def _make_mock_encode_result(embedding_dim: int = 4):
    """Build a mock EncodeResult with hard_filter and embedding."""
    enc = MagicMock()
    enc.hard_filter = {"street_class": "postflop", "pot_type": "srp"}
    enc.embedding = [0.1] * embedding_dim
    return enc


def _make_mock_adapter(gs, n_decisions: int = 2):
    """Build a mock SimAdapter producing n_decisions per hand then stopping.

    The counter resets each time adapter._env.reset() is called (mirroring
    the real harness which calls env.reset() at the start of each hand).
    """
    adapter = MagicMock()
    adapter._env = MagicMock()

    call_count = {"n": 0}

    def env_reset():
        call_count["n"] = 0

    def has_more():
        return call_count["n"] < n_decisions

    def next_game_state():
        call_count["n"] += 1
        return gs

    def current_legal_actions():
        return ["call", "fold"]

    def map_to_rlcard_action(action, legal):
        return 0

    adapter._env.reset.side_effect = env_reset
    adapter.has_more.side_effect = has_more
    adapter.next_game_state.side_effect = next_game_state
    adapter.current_legal_actions.side_effect = current_legal_actions
    adapter.map_to_rlcard_action.side_effect = map_to_rlcard_action
    return adapter


def _make_mock_engine(enc):
    """Build a mock engine that returns (action, flagged_sparse, max_dist, enc)."""
    engine = MagicMock()
    engine.decide_with_encoding.return_value = ("call", False, 0.1, enc)
    return engine


class SpyWriter:
    """Minimal writer spy that captures record() calls as list of kwargs dicts."""

    def __init__(self):
        self.calls: list[dict] = []

    def record(self, cluster_key, embedding, action_taken, **kwargs):
        self.calls.append(
            {
                "cluster_key": cluster_key,
                "embedding": embedding,
                "action_taken": action_taken,
                **kwargs,
            }
        )

    def close(self):
        pass


def test_hand_id_format():
    """Harness mints hand_id = f'{session_id}_h{hand_idx}' once per hand."""
    from src.sim.harness import run_record_session

    gs = _make_mock_game_state()
    enc = _make_mock_encode_result()
    adapter = _make_mock_adapter(gs, n_decisions=1)
    engine = _make_mock_engine(enc)
    writer = SpyWriter()

    session_id = "test_sess_001"
    run_record_session(adapter, engine, writer, session_seed=42, n_hands=2, session_id=session_id)

    assert len(writer.calls) == 2, f"expected 2 calls (1 decision/hand * 2 hands), got {len(writer.calls)}"
    assert writer.calls[0]["hand_id"] == f"{session_id}_h0"
    assert writer.calls[1]["hand_id"] == f"{session_id}_h1"


def test_decision_id_format():
    """Harness mints decision_id = f'{hand_id}_dp{decisions_this_hand}'."""
    from src.sim.harness import run_record_session

    gs = _make_mock_game_state()
    enc = _make_mock_encode_result()
    adapter = _make_mock_adapter(gs, n_decisions=3)
    engine = _make_mock_engine(enc)
    writer = SpyWriter()

    session_id = "test_sess_002"
    run_record_session(adapter, engine, writer, session_seed=42, n_hands=1, session_id=session_id)

    assert len(writer.calls) == 3
    hand_id = f"{session_id}_h0"
    assert writer.calls[0]["decision_id"] == f"{hand_id}_dp0"
    assert writer.calls[1]["decision_id"] == f"{hand_id}_dp1"
    assert writer.calls[2]["decision_id"] == f"{hand_id}_dp2"


def test_felt_snapshot_contains_all_11_fields():
    """felt_snapshot passed to writer is a dict with all 11 GameState field names."""
    from src.sim.harness import run_record_session

    gs = _make_mock_game_state()
    enc = _make_mock_encode_result()
    adapter = _make_mock_adapter(gs, n_decisions=1)
    engine = _make_mock_engine(enc)
    writer = SpyWriter()

    run_record_session(adapter, engine, writer, session_seed=42, n_hands=1, session_id="test_sess_003")

    assert len(writer.calls) == 1
    felt = writer.calls[0]["felt_snapshot"]
    assert isinstance(felt, dict), f"felt_snapshot should be dict, got {type(felt)}"
    missing = _GAMESTATE_FIELDS - felt.keys()
    assert not missing, f"felt_snapshot missing fields: {missing}"


def test_felt_snapshot_values_match_game_state():
    """felt_snapshot values are the verbatim GameState field values (tuples -> lists)."""
    from src.sim.harness import run_record_session

    gs = _make_mock_game_state()
    enc = _make_mock_encode_result()
    adapter = _make_mock_adapter(gs, n_decisions=1)
    engine = _make_mock_engine(enc)
    writer = SpyWriter()

    run_record_session(adapter, engine, writer, session_seed=42, n_hands=1, session_id="test_sess_004")

    felt = writer.calls[0]["felt_snapshot"]
    assert felt["street"] == "flop"
    assert felt["hero_position"] == "BTN"
    assert felt["hero_hole_cards"] == ["Ah", "Kd"], "tuples should be lists in felt"
    assert felt["board_cards"] == ["2c", "7s", "Jh"], "tuples should be lists in felt"
    assert felt["pot_size_bb"] == 4.5
    assert felt["prior_street_aggressor"] == "CO"


def test_decisions_this_hand_increments_after_record():
    """decision_id counter increments across multiple decisions in one hand."""
    from src.sim.harness import run_record_session

    gs = _make_mock_game_state()
    enc = _make_mock_encode_result()
    adapter = _make_mock_adapter(gs, n_decisions=2)
    engine = _make_mock_engine(enc)
    writer = SpyWriter()

    session_id = "test_sess_005"
    run_record_session(adapter, engine, writer, session_seed=42, n_hands=1, session_id=session_id)

    hand_id = f"{session_id}_h0"
    assert writer.calls[0]["decision_id"] == f"{hand_id}_dp0"
    assert writer.calls[1]["decision_id"] == f"{hand_id}_dp1"
