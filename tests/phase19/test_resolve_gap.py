from __future__ import annotations

import uuid
from unittest.mock import MagicMock


def _make_mock_conn(obs_row, active_node_row):
    """Build a MagicMock conn whose cursor returns obs_row then active_node_row."""
    mock_conn = MagicMock()
    cursor_ctx = MagicMock()

    fetchone_side_effects = [obs_row, active_node_row]
    cursor_ctx.__enter__.return_value.fetchone.side_effect = fetchone_side_effects

    mock_conn.cursor.return_value = cursor_ctx
    mock_conn.transaction.return_value.__enter__ = lambda s: s
    mock_conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


def _make_mock_engine():
    mock_engine = MagicMock()
    mock_engine.apply.return_value = MagicMock(patch_id=uuid.uuid4(), new_node_id=uuid.uuid4())
    return mock_engine


def test_write_provenance():
    from src.study.gaps import resolve_gap

    obs_row = (
        "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        [0.1] * 128,
        "hand123",
        {"hero_facing_bet_bb": 0, "effective_stack_bb": 50.0, "street": "flop"},
    )
    active_node_row = None

    mock_conn = _make_mock_conn(obs_row, active_node_row)
    mock_engine = _make_mock_engine()

    resolve_gap(
        "hand123_dp0",
        {"check": 1.0},
        _tsdb_conn=mock_conn,
        _milvus=MagicMock(),
        _patch_engine=mock_engine,
    )

    spec = mock_engine.apply.call_args.args[0]
    assert spec.gto_score == 0.911
    assert spec.confidence == 1.0
    assert spec.source == "manual"


def test_sentinel_validation():
    from src.study.gaps import resolve_gap

    obs_row = (
        "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        [0.1] * 128,
        "hand123",
        {"hero_facing_bet_bb": 0, "effective_stack_bb": 50.0, "street": "flop"},
    )
    active_node_row = None

    mock_conn = _make_mock_conn(obs_row, active_node_row)
    mock_engine = _make_mock_engine()

    resolve_gap(
        "hand123_dp0",
        {"check": 1.0},
        _tsdb_conn=mock_conn,
        _milvus=MagicMock(),
        _patch_engine=mock_engine,
    )

    validation = mock_engine.apply.call_args.kwargs["validation"]
    assert validation.zero_hits is True
    assert validation.n_hands == 0


def test_first_ever_node():
    from src.study.gaps import resolve_gap

    obs_row = (
        "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        [0.1] * 128,
        "hand123",
        {"hero_facing_bet_bb": 0, "effective_stack_bb": 50.0, "street": "flop"},
    )
    active_node_row = None

    mock_conn = _make_mock_conn(obs_row, active_node_row)
    mock_engine = _make_mock_engine()

    resolve_gap(
        "hand123_dp0",
        {"check": 1.0},
        _tsdb_conn=mock_conn,
        _milvus=MagicMock(),
        _patch_engine=mock_engine,
    )

    spec = mock_engine.apply.call_args.args[0]
    assert spec.prev_node_id is None
    mock_engine.apply.assert_called_once()


def test_missing_embedding_does_not_upsert_to_milvus():
    """When normalization returns None (absent/wrong-dim embedding), the gap is
    written to TSDB but the un-z-scored vector must NEVER reach the COSINE corpus.
    The raw embedding must not be passed as the spec embedding either."""
    from src.study.gaps import resolve_gap

    obs_row = (
        "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        [],  # absent embedding → _normalize_for_corpus returns None
        "hand123",
        {"hero_facing_bet_bb": 0, "effective_stack_bb": 50.0, "street": "flop"},
    )
    active_node_row = None

    mock_conn = _make_mock_conn(obs_row, active_node_row)
    mock_engine = _make_mock_engine()

    resolve_gap(
        "hand123_dp0",
        {"check": 1.0},
        _tsdb_conn=mock_conn,
        _milvus=MagicMock(),
        _patch_engine=mock_engine,
    )

    mock_engine.apply.assert_called_once()
    spec = mock_engine.apply.call_args.args[0]
    assert spec.embedding == [], "raw un-z-scored embedding must not be carried as spec embedding"
    assert mock_engine.apply.call_args.kwargs["skip_milvus"] is True


def test_missing_dim_embedding_skips_milvus():
    """A wrong-dimension embedding (dim mismatch) also normalizes to None and must
    skip the Milvus write."""
    from src.study.gaps import resolve_gap

    obs_row = (
        "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        [0.1] * 7,  # wrong dim → _normalize_for_corpus returns None
        "hand123",
        {"hero_facing_bet_bb": 0, "effective_stack_bb": 50.0, "street": "flop"},
    )
    active_node_row = None

    mock_conn = _make_mock_conn(obs_row, active_node_row)
    mock_engine = _make_mock_engine()

    resolve_gap(
        "hand123_dp0",
        {"check": 1.0},
        _tsdb_conn=mock_conn,
        _milvus=MagicMock(),
        _patch_engine=mock_engine,
    )

    assert mock_engine.apply.call_args.kwargs["skip_milvus"] is True


def test_flag_clear_and_dedup_marker_share_one_transaction():
    """The flagged_sparse UPDATE and the human_resolved INSERT must commit together,
    so a partial failure cannot leave the flag cleared without a dedup marker."""
    from src.study.gaps import resolve_gap

    events: list[str] = []

    obs_row = (
        "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        [0.1] * 128,
        "hand123",
        {"hero_facing_bet_bb": 0, "effective_stack_bb": 50.0, "street": "flop"},
    )

    mock_conn = MagicMock()

    cursor_ctx = MagicMock()
    fetch = cursor_ctx.__enter__.return_value
    fetch.fetchone.side_effect = [obs_row, None]

    def _record_execute(sql, *a, **k):
        if "flagged_sparse" in sql:
            events.append("update")
        elif "INSERT INTO solver_cache" in sql:
            events.append("insert")

    fetch.execute.side_effect = _record_execute
    mock_conn.cursor.return_value = cursor_ctx

    txn = mock_conn.transaction.return_value
    txn.__enter__ = lambda s: (events.append("txn_enter"), s)[1]
    txn.__exit__ = lambda s, *e: (events.append("txn_exit"), False)[1]

    mock_engine = _make_mock_engine()

    resolve_gap(
        "hand123_dp0",
        {"check": 1.0},
        _tsdb_conn=mock_conn,
        _milvus=MagicMock(),
        _patch_engine=mock_engine,
    )

    # Both writes commit in ONE transaction: no txn_exit may occur between them,
    # else a partial failure could clear the flag without writing the dedup marker.
    assert "update" in events and "insert" in events
    lo, hi = sorted((events.index("update"), events.index("insert")))
    assert events.index("txn_enter") < lo
    assert "txn_exit" not in events[lo:hi], "writes split across separate transactions"
