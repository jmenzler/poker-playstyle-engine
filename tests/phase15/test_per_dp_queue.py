# rot-allow-file
"""TDD RED: per-DP queue dedup, preflop exclusion, placeholder skip, per-DP Milvus inject.

Tests for the solver queue refactor (15-07):
  - _PRIORITY_SQL dedups per decision_id (NOT cluster_key)
  - Preflop observations excluded
  - Placeholder spots skipped + recorded in solver_cache
  - _inject writes a per-DP Milvus row keyed by decision_id + obs embedding
  - persist_solve writes decision_id
  - SolverCacheEntry accepts decision_id field
  - migration 018 pattern (unit-tested separately)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_PLACEHOLDER_RANGE = "AA-22,AKs-A2s"

# ---------------------------------------------------------------------------
# Real felt_snapshot shapes (no synthetic keys like pot_type/villain_pos)
# ---------------------------------------------------------------------------

_REAL_3BET_POSTFLOP_FELT: dict = {
    "street": "flop",
    "board_cards": ["Jc", "7s", "2h"],
    "pot_size_bb": 27.0,
    "hero_position": "BB",
    "action_sequence": [
        "UTG:fold",
        "MP:fold",
        "CO:fold",
        "BTN:open_2.5",
        "SB:fold",
        "BB:3bet_9",
        "BTN:call",
    ],
    "hero_hole_cards": ["Ah", "Kd"],
    "hero_bet_size_bb": 0.0,
    "effective_stack_bb": 91.0,
    "hero_facing_bet_bb": 0.0,
    "opponents_remaining": 1,
    "prior_street_aggressor": "BB",
}

_REAL_PREFLOP_FELT: dict = {
    "street": "preflop",
    "board_cards": [],
    "pot_size_bb": 3.0,
    "hero_position": "BTN",
    "action_sequence": ["UTG:fold", "MP:fold", "CO:fold", "BTN:open_2.5"],
    "hero_hole_cards": ["Ah", "Kd"],
    "hero_bet_size_bb": 2.5,
    "effective_stack_bb": 97.5,
    "hero_facing_bet_bb": 0.0,
    "opponents_remaining": 1,
    "prior_street_aggressor": None,
}


def _make_obs_row(
    obs_id: str = "obs-dp-001",
    decision_id: str = "dp-uuid-001",
    cluster_key: str = "hero_pos_rel=BB|n_players_active=2|pot_type=3bet|street_class=postflop",
    embedding: list[float] | None = None,
    felt: dict | None = None,
) -> dict:
    return {
        "obs_id": obs_id,
        "decision_id": decision_id,
        "cluster_key": cluster_key,
        "embedding": embedding or [0.1] * 80,
        "max_neighbor_distance": 0.82,
        "felt_snapshot": felt or _REAL_3BET_POSTFLOP_FELT,
        "cluster_freq": 300,
        "ev_loss": 0.9,
        "priority_score": 300 * 0.9 * 0.82,
    }


# ---------------------------------------------------------------------------
# 1. _PRIORITY_SQL shape — dedups per decision_id, not cluster_key
# ---------------------------------------------------------------------------


def test_priority_sql_contains_decision_id_column() -> None:
    """_PRIORITY_SQL must SELECT o.decision_id — needed for per-DP keying."""
    from src.solver.queue_driver import _PRIORITY_SQL

    sql_lower = _PRIORITY_SQL.lower()
    assert "decision_id" in sql_lower, (
        "_PRIORITY_SQL must include decision_id column. Got: " + _PRIORITY_SQL[:200]
    )


def test_priority_sql_contains_embedding_column() -> None:
    """_PRIORITY_SQL must SELECT o.embedding — obs embedding travels into _inject."""
    from src.solver.queue_driver import _PRIORITY_SQL

    sql_lower = _PRIORITY_SQL.lower()
    assert "embedding" in sql_lower, (
        "_PRIORITY_SQL must include embedding column. Got: " + _PRIORITY_SQL[:200]
    )


def test_priority_sql_dedup_is_per_decision_id() -> None:
    """The NOT EXISTS dedup in _PRIORITY_SQL must reference sc.decision_id, not sc.cluster_key."""
    from src.solver.queue_driver import _PRIORITY_SQL

    assert "sc.decision_id" in _PRIORITY_SQL, (
        "NOT EXISTS dedup must be per decision_id (sc.decision_id = o.decision_id), "
        "not per cluster_key. Got: " + _PRIORITY_SQL
    )
    # Old per-cluster dedup should NOT be present
    assert "sc.cluster_key = o.cluster_key" not in _PRIORITY_SQL, (
        "_PRIORITY_SQL still has old per-cluster_key dedup — must switch to per decision_id"
    )


def test_priority_sql_excludes_preflop() -> None:
    """_PRIORITY_SQL must exclude preflop obs (board=[] causes postflop-cli panic)."""
    from src.solver.queue_driver import _PRIORITY_SQL

    sql_upper = _PRIORITY_SQL.upper()
    assert "PREFLOP" in sql_upper or "preflop" in _PRIORITY_SQL, (
        "_PRIORITY_SQL must filter out preflop observations. "
        "Add: AND o.cluster_key NOT LIKE '%street_class=preflop%'"
    )


# ---------------------------------------------------------------------------
# 2. _fetch_batch — deduplicates per decision_id within batch, threads both fields
# ---------------------------------------------------------------------------


def _make_fetch_row(
    obs_id: str,
    decision_id: str,
    cluster_key: str,
    distance: float = 0.8,
    freq: int = 10,
    ev_loss: float = 0.5,
    embedding: list[float] | None = None,
    felt: dict | None = None,
    hand_id: str = "hand-001",
) -> tuple:
    """Build a raw DB cursor row matching the new _PRIORITY_SQL column order."""
    emb = embedding or [0.1] * 80
    priority_score = freq * ev_loss * distance
    return (
        obs_id,
        decision_id,
        cluster_key,
        distance,
        felt or _REAL_3BET_POSTFLOP_FELT,
        freq,
        ev_loss,
        priority_score,
        emb,
        hand_id,
    )


def _make_cursor_returning(rows: list[tuple]) -> MagicMock:
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchall.return_value = rows
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


def test_fetch_batch_includes_decision_id_in_result() -> None:
    """_fetch_batch must include decision_id in each result row dict."""
    from src.solver.queue_driver import QueueDriver

    row = _make_fetch_row(
        "obs-1", "dp-uuid-1", "hero_pos_rel=BB|n_players_active=2|pot_type=3bet|street_class=postflop"
    )
    mock_conn = _make_cursor_returning([row])

    driver = QueueDriver.__new__(QueueDriver)
    batch = driver._fetch_batch(mock_conn, limit=100)

    assert len(batch) == 1
    assert "decision_id" in batch[0], f"decision_id missing from batch row: {list(batch[0].keys())}"
    assert batch[0]["decision_id"] == "dp-uuid-1"


def test_fetch_batch_includes_embedding_in_result() -> None:
    """_fetch_batch must include embedding in each result row dict."""
    from src.solver.queue_driver import QueueDriver

    emb = [0.5] * 80
    row = _make_fetch_row(
        "obs-1",
        "dp-uuid-1",
        "hero_pos_rel=BB|n_players_active=2|pot_type=3bet|street_class=postflop",
        embedding=emb,
    )
    mock_conn = _make_cursor_returning([row])

    driver = QueueDriver.__new__(QueueDriver)
    batch = driver._fetch_batch(mock_conn, limit=100)

    assert "embedding" in batch[0], f"embedding missing from batch row: {list(batch[0].keys())}"
    assert batch[0]["embedding"] == emb


def test_fetch_batch_deduplicates_per_decision_id() -> None:
    """Two rows with same decision_id but different cluster_key: only first is kept."""
    from src.solver.queue_driver import QueueDriver

    row_a = _make_fetch_row(
        "obs-1", "dp-uuid-1", "hero_pos_rel=BB|n_players_active=2|pot_type=3bet|street_class=postflop"
    )
    row_b = _make_fetch_row(
        "obs-2", "dp-uuid-1", "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
    )
    mock_conn = _make_cursor_returning([row_a, row_b])

    driver = QueueDriver.__new__(QueueDriver)
    batch = driver._fetch_batch(mock_conn, limit=100)

    assert len(batch) == 1, f"Expected 1 result (dedup by decision_id), got {len(batch)}"
    assert batch[0]["obs_id"] == "obs-1"


def test_fetch_batch_allows_multiple_decision_ids_per_cluster() -> None:
    """Two rows with same cluster_key but different decision_id: both kept (old dedup dropped)."""
    from src.solver.queue_driver import QueueDriver

    ck = "hero_pos_rel=BB|n_players_active=2|pot_type=3bet|street_class=postflop"
    row_a = _make_fetch_row("obs-1", "dp-uuid-1", ck)
    row_b = _make_fetch_row("obs-2", "dp-uuid-2", ck)
    mock_conn = _make_cursor_returning([row_a, row_b])

    driver = QueueDriver.__new__(QueueDriver)
    batch = driver._fetch_batch(mock_conn, limit=100)

    assert len(batch) == 2, (
        f"Expected 2 results (different decision_ids in same cluster allowed), got {len(batch)}. "
        "Old per-cluster dedup incorrectly collapsed them."
    )


# ---------------------------------------------------------------------------
# 3. Placeholder detection + skip — _is_placeholder_range / _solve_one_spot
# ---------------------------------------------------------------------------


def test_is_placeholder_range_detects_placeholder() -> None:
    """_is_placeholder_range returns True for the canonical placeholder string."""
    from src.solver.queue_driver import _is_placeholder_range

    assert _is_placeholder_range(_PLACEHOLDER_RANGE), (
        f"_is_placeholder_range must return True for {_PLACEHOLDER_RANGE!r}"
    )


def test_is_placeholder_range_returns_false_for_real_range() -> None:
    """_is_placeholder_range returns False for a real poker range."""
    from src.solver.queue_driver import _is_placeholder_range

    assert not _is_placeholder_range("AA,KK,QQ,AKs"), (
        "_is_placeholder_range must return False for a real range"
    )


def test_placeholder_spot_skipped_not_solved() -> None:
    """When range_ip or range_oop is placeholder, _inject is NOT called and a skip record is written."""

    obs_row = _make_obs_row()
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    inject_calls: list[dict] = []
    persist_calls: list[dict] = []

    def mock_inject(*args, **kwargs):
        inject_calls.append(kwargs)

    def mock_persist(entry, *, _tsdb_conn=None):
        persist_calls.append({"entry": entry})

    placeholder_context = {
        "range_ip": _PLACEHOLDER_RANGE,
        "range_oop": "AA,KK",
        "action_line_full": ["BTN:raise", "BB:3bet", "BTN:call"],
        "board": ["Jc", "7s", "2h"],
        "hero_hole": ["Ah", "Kd"],
        "pot": 2700,
        "effective_stack": 9100,
        "prev_bet": None,
        "street": "flop",
        "hero_pos": "BB",
        "villain_pos": "BTN",
        "pot_type": "3bet",
        "n_players_at_street": 2,
        "preflop_action_seq": ["BTN:raise", "BB:3bet", "BTN:call"],
        "solver_settings": {},
    }

    mock_spot = MagicMock()
    mock_spot.range_ip = _PLACEHOLDER_RANGE
    mock_spot.range_oop = "AA,KK"

    mock_solver_result = MagicMock()
    mock_solver_result.action_dist = {"check": 1.0}
    mock_solver_result.exploitability_pct = 0.0

    with (
        patch("src.solver.queue_driver._build_spot", return_value=(mock_spot, placeholder_context)),
        patch("src.solver.queue_driver._inject", mock_inject),
        patch("src.solver.queue_driver.persist_solve", mock_persist),
    ):
        from pathlib import Path

        from src._config import AutoLoopConfig, load_toml_config
        from src.solver.queue_driver import _solve_one_spot

        cfg = load_toml_config(Path("config/autoloop.toml"), AutoLoopConfig).solver_queue
        result = _solve_one_spot(obs_row, MagicMock(), cfg)

    assert result.get("skipped_placeholder"), (
        "_solve_one_spot must return {'skipped_placeholder': True} for placeholder spots"
    )


def test_placeholder_skip_writes_solver_cache_record() -> None:
    """Placeholder skip writes a solver_cache row for decision_id so it won't re-loop."""
    from src.solver.queue_driver import _handle_placeholder_skip

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    persist_calls: list = []

    def mock_persist(entry, *, _tsdb_conn=None):
        persist_calls.append(entry)

    obs_row = _make_obs_row(decision_id="dp-skip-001")

    with patch("src.solver.queue_driver.persist_solve", mock_persist):
        _handle_placeholder_skip(obs_row, conn=mock_conn)

    assert len(persist_calls) == 1, "Must call persist_solve once to record the skip"
    entry = persist_calls[0]
    assert entry.decision_id == "dp-skip-001", (
        f"skip record must use decision_id={obs_row['decision_id']!r}, got {entry.decision_id!r}"
    )
    assert entry.solver_version == "skipped_placeholder", (
        f"skip record solver_version must be 'skipped_placeholder', got {entry.solver_version!r}"
    )
    assert entry.action_dist == {}, f"skip record action_dist must be empty dict, got {entry.action_dist!r}"


def test_timeout_skip_writes_solver_cache_record() -> None:
    """A timed-out solve writes a 'timed_out' solver_cache row so resumes dedup it out."""
    from src.solver.queue_driver import _handle_timeout_skip

    persist_calls: list = []

    def mock_persist(entry, *, _tsdb_conn=None):
        persist_calls.append(entry)

    obs_row = _make_obs_row(decision_id="dp-timeout-001")

    with patch("src.solver.queue_driver.persist_solve", mock_persist):
        _handle_timeout_skip(obs_row, conn=MagicMock())

    assert len(persist_calls) == 1, "Must call persist_solve once to record the timeout"
    entry = persist_calls[0]
    assert entry.decision_id == "dp-timeout-001", (
        f"timeout record must use decision_id, got {entry.decision_id!r}"
    )
    assert entry.solver_version == "timed_out", (
        f"timeout record solver_version must be 'timed_out', got {entry.solver_version!r}"
    )
    assert entry.action_dist == {}, f"timeout record action_dist must be empty, got {entry.action_dist!r}"


# ---------------------------------------------------------------------------
# 4. _inject — per-DP Milvus write (NOT PatchEngine)
# ---------------------------------------------------------------------------


def _make_postflop_obs_row(
    decision_id: str = "dp-inject-001",
    embedding: list[float] | None = None,
) -> dict:
    return {
        "obs_id": "obs-inject-dp-001",
        "decision_id": decision_id,
        "cluster_key": "hero_pos_rel=BB|n_players_active=2|pot_type=3bet|street_class=postflop",
        "embedding": embedding or [0.2] * 80,
        "max_neighbor_distance": 0.8,
        "felt_snapshot": _REAL_3BET_POSTFLOP_FELT,
        "cluster_freq": 50,
        "ev_loss": 0.8,
        "priority_score": 50 * 0.8 * 0.8,
    }


def test_inject_does_not_call_patch_engine() -> None:
    """_inject for the solver path must NOT call PatchEngine — write directly to Milvus."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_postflop_obs_row()
    spot = SolverSpot(pot=2700, effective_stack=9100, board=["Jc", "7s", "2h"], range_ip="AA", range_oop="KK")

    mock_result = MagicMock()
    mock_result.action_dist = {"bet_33": 0.7, "check": 0.3}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_milvus = MagicMock()
    mock_patch_engine = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
            _patch_engine=mock_patch_engine,
        )

    mock_patch_engine.apply.assert_not_called(), "PatchEngine.apply must NOT be called on solver path"


def test_inject_calls_milvus_upsert_with_per_dp_row() -> None:
    """_inject calls milvus.upsert with a row keyed by obs decision_id + obs embedding."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_embedding = [0.33] * 80
    obs_row = _make_postflop_obs_row(decision_id="dp-inject-milvus-001", embedding=obs_embedding)

    spot = SolverSpot(pot=2700, effective_stack=9100, board=["Jc", "7s", "2h"], range_ip="AA", range_oop="KK")

    mock_result = MagicMock()
    mock_result.action_dist = {"bet_33": 0.7, "check": 0.3}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
        )

    mock_milvus.upsert.assert_called_once()
    upsert_kwargs = mock_milvus.upsert.call_args.kwargs
    data = upsert_kwargs["data"]
    assert len(data) == 1, f"Expected 1 row in upsert data, got {len(data)}"

    row = data[0]
    assert row["decision_id"] == "dp-inject-milvus-001", (
        f"Milvus row decision_id must be obs decision_id, got {row.get('decision_id')!r}"
    )
    import numpy as np

    from src.solver.queue_driver import _load_zscore_transform, normalize

    mean, std, weights = _load_zscore_transform("postflop_decisions")
    expected = normalize(np.asarray(obs_embedding, dtype=np.float64), mean, std, weights).tolist()
    assert row["embedding"] == pytest.approx(expected), (
        "Milvus row embedding must be the obs's own vector run through the shared "
        "z-score+weights transform (same space the corpus + query use)"
    )


def test_inject_milvus_row_has_full_schema_fields() -> None:
    """_inject row carries every collection field: enable_dynamic_field=False
    and no defaults, so a row missing any field fails the (swallowed) upsert.
    """
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    _SCHEMA_FIELDS = frozenset(
        {
            "decision_id",
            "embedding",
            "street_class",
            "street",
            "pot_type",
            "hero_pos_rel",
            "n_players_active",
            "spr_x100",
            "hero_action_type",
            "hero_action_size_pot_frac",
            "raise_ratio",
            "hero_action_allin",
            "active",
            "confidence",
            "gto_score",
            "feature_spec_version",
            "action_dist",
        }
    )

    obs_row = _make_postflop_obs_row()
    spot = SolverSpot(pot=2700, effective_stack=9100, board=["Jc", "7s", "2h"], range_ip="AA", range_oop="KK")

    mock_result = MagicMock()
    mock_result.action_dist = {"bet_33": 0.7, "check": 0.3}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
        )

    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert set(row.keys()) == _SCHEMA_FIELDS, (
        f"Milvus row must have exactly the collection schema fields.\n"
        f"  Extra  : {set(row.keys()) - _SCHEMA_FIELDS}\n"
        f"  Missing: {_SCHEMA_FIELDS - set(row.keys())}"
    )


def test_inject_milvus_row_active_true() -> None:
    """_inject sets active=True on the upserted Milvus row."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_postflop_obs_row()
    spot = SolverSpot(pot=2700, effective_stack=9100, board=["Jc", "7s", "2h"], range_ip="AA", range_oop="KK")

    mock_result = MagicMock()
    mock_result.action_dist = {"bet_33": 0.7, "check": 0.3}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(obs_row, spot, mock_result, conn=mock_conn, milvus=mock_milvus, palette_lookup={})

    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert row["active"] is True, f"Milvus row active must be True, got {row.get('active')!r}"


def test_inject_hero_action_type_is_dominant_action() -> None:
    """_inject sets hero_action_type = dominant action from solver action_dist."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_postflop_obs_row()
    spot = SolverSpot(pot=2700, effective_stack=9100, board=["Jc", "7s", "2h"], range_ip="AA", range_oop="KK")

    mock_result = MagicMock()
    mock_result.action_dist = {"bet_33": 0.1, "check": 0.05, "fold": 0.85}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(obs_row, spot, mock_result, conn=mock_conn, milvus=mock_milvus, palette_lookup={})

    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert row["hero_action_type"] == "fold", (
        f"hero_action_type must be dominant action 'fold', got {row.get('hero_action_type')!r}"
    )


def test_inject_persist_before_milvus_upsert() -> None:
    """persist_solve must be called BEFORE milvus.upsert (D-16 irreversible write-order)."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_postflop_obs_row()
    spot = SolverSpot(pot=2700, effective_stack=9100, board=["Jc", "7s", "2h"], range_ip="AA", range_oop="KK")

    mock_result = MagicMock()
    mock_result.action_dist = {"bet_33": 0.7, "check": 0.3}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_milvus = MagicMock()

    call_order: list[str] = []

    def track_persist(entry, *, _tsdb_conn=None):
        call_order.append("persist_solve")

    mock_milvus.upsert.side_effect = lambda **kwargs: call_order.append("milvus_upsert")

    with patch("src.solver.queue_driver.persist_solve", track_persist):
        _inject(obs_row, spot, mock_result, conn=mock_conn, milvus=mock_milvus, palette_lookup={})

    assert "persist_solve" in call_order, "persist_solve was not called"
    assert "milvus_upsert" in call_order, "milvus.upsert was not called"
    assert call_order.index("persist_solve") < call_order.index("milvus_upsert"), (
        f"persist_solve must come BEFORE milvus.upsert. Order was: {call_order}"
    )


# ---------------------------------------------------------------------------
# 5. persist_solve writes decision_id
# ---------------------------------------------------------------------------


def test_solver_cache_entry_accepts_decision_id() -> None:
    """SolverCacheEntry must accept a decision_id field."""
    from src.eval.solver_cache import SolverCacheEntry

    entry = SolverCacheEntry(
        cluster_key="hero_pos_rel=BB|n_players_active=2|pot_type=3bet|street_class=postflop",
        decision_id="dp-persist-001",
        action_dist={"bet_33": 0.7, "check": 0.3},
        exploitability_pct=0.4,
        solver_version="postflop-cli",
    )
    assert entry.decision_id == "dp-persist-001"


def test_persist_solve_writes_decision_id() -> None:
    """persist_solve must write decision_id to the solver_cache row."""
    from src.eval.solver_cache import SolverCacheEntry, persist_solve

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur
    mock_conn.transaction.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.transaction.return_value.__exit__ = MagicMock(return_value=False)

    entry = SolverCacheEntry(
        cluster_key="hero_pos_rel=BB|n_players_active=2|pot_type=3bet|street_class=postflop",
        decision_id="dp-persist-sql-001",
        action_dist={"bet_33": 0.7},
        exploitability_pct=0.4,
        solver_version="postflop-cli",
    )

    persist_solve(entry, _tsdb_conn=mock_conn)

    mock_cur.execute.assert_called_once()
    sql, params = mock_cur.execute.call_args[0]
    # decision_id must appear in the SQL
    assert "decision_id" in sql, f"persist_solve SQL must include decision_id column. Got: {sql}"
    # decision_id value must be in the params
    assert "dp-persist-sql-001" in params, (
        f"persist_solve params must include decision_id value. Params: {params}"
    )


# ---------------------------------------------------------------------------
# 6. inject writes decision_id to persist_solve
# ---------------------------------------------------------------------------


def test_inject_passes_decision_id_to_persist_solve() -> None:
    """_inject passes obs_row decision_id to SolverCacheEntry for persist_solve."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_postflop_obs_row(decision_id="dp-verify-persist-001")
    spot = SolverSpot(pot=2700, effective_stack=9100, board=["Jc", "7s", "2h"], range_ip="AA", range_oop="KK")

    mock_result = MagicMock()
    mock_result.action_dist = {"bet_33": 0.7, "check": 0.3}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    persisted_entries: list = []

    def capture_persist(entry, *, _tsdb_conn=None):
        persisted_entries.append(entry)

    with patch("src.solver.queue_driver.persist_solve", capture_persist):
        _inject(obs_row, spot, mock_result, conn=mock_conn, milvus=MagicMock(), palette_lookup={})

    assert len(persisted_entries) >= 1, "persist_solve was not called"
    entry = persisted_entries[0]
    assert entry.decision_id == "dp-verify-persist-001", (
        f"persist_solve entry.decision_id must be obs decision_id 'dp-verify-persist-001', "
        f"got {entry.decision_id!r}"
    )


# ---------------------------------------------------------------------------
# 7. _count_sparse dedup also uses decision_id
# ---------------------------------------------------------------------------


def test_count_sparse_uses_decision_id_dedup() -> None:
    """_count_sparse query must count per decision_id, not per cluster_key."""
    from src.solver.queue_driver import QueueDriver

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = (42,)
    mock_conn.cursor.return_value = mock_cur

    driver = QueueDriver.__new__(QueueDriver)
    count = driver._count_sparse(mock_conn)

    assert count == 42
    sql = mock_cur.execute.call_args[0][0]
    assert "decision_id" in sql, f"_count_sparse SQL must dedup by decision_id. Got: {sql}"


def test_priority_sql_escapes_like_percent():
    """_PRIORITY_SQL LIKE patterns must use %% (psycopg treats bare % as a placeholder).

    A bare % alongside the %s LIMIT param raises psycopg.ProgrammingError at execute time.
    """
    from src.solver.queue_driver import _PRIORITY_SQL

    assert "LIKE '%%" in _PRIORITY_SQL and "%%'" in _PRIORITY_SQL, (
        "LIKE pattern must escape % as %% for psycopg parameterized execute"
    )
    assert "LIKE '%street_class" not in _PRIORITY_SQL, "bare single-% LIKE will break execute()"
