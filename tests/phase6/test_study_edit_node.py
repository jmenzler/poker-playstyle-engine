"""Unit tests for src/study/edit_node.py (CLI-03 / D-11).

Covers:
- _validate_action_dist: sum tolerance, unknown keys, negative frequencies
- edit_node: happy path, missing active node, sentinel ValidationResult
- set_lock_from_autoloop: direct UPDATE (documented PatchEngine bypass)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src._errors import ValidationError
from src.study.edit_node import (
    _validate_action_dist,
    edit_node,
    set_lock_from_autoloop,
)


def _mock_conn(strategy_node_row=("node1", [0.1] * 80, 0.5, 0.5), decision_row=("dp_real",)):
    """Build a MagicMock psycopg conn. fetchone yields the active-node row, then the
    representative decision_id row (the two SELECTs edit_node makes)."""
    conn = MagicMock()

    cm = MagicMock()
    cm.__enter__ = lambda s: cm
    cm.__exit__ = lambda *a: False
    cm.fetchone.side_effect = [strategy_node_row, decision_row]
    cm.rowcount = 1
    conn.cursor.return_value = cm

    tx = MagicMock()
    tx.__enter__ = lambda s: s
    tx.__exit__ = lambda *a: False
    conn.transaction.return_value = tx
    return conn


# --- _validate_action_dist ---------------------------------------------------


def test_validate_sum_passes_exactly_one():
    _validate_action_dist({"fold": 0.5, "call": 0.5})


def test_validate_sum_passes_within_tolerance():
    # within 0.001 tolerance
    _validate_action_dist({"fold": 0.5005, "call": 0.4995})


def test_validate_sum_fails_outside_tolerance():
    with pytest.raises(ValidationError, match="sum"):
        _validate_action_dist({"fold": 0.5, "call": 0.4})


def test_validate_unknown_key():
    with pytest.raises(ValidationError, match="unknown"):
        _validate_action_dist({"fold": 0.5, "garbage_action": 0.5})


def test_validate_negative_frequency():
    with pytest.raises(ValidationError, match="negative"):
        _validate_action_dist({"fold": 1.2, "call": -0.2})


# --- edit_node --------------------------------------------------------------


def test_edit_node_missing_active_node_raises():
    conn = _mock_conn(strategy_node_row=None)
    with pytest.raises(ValidationError, match="no active"):
        edit_node(
            "ck1",
            {"fold": 0.5, "call": 0.5},
            _tsdb_conn=conn,
            _patch_engine=MagicMock(),
        )


def test_edit_node_happy_path_calls_patch_engine_with_manual_source():
    conn = _mock_conn()
    engine = MagicMock()
    record = MagicMock()
    record.patch_id = "p123"
    record.new_node_id = "n456"
    engine.apply.return_value = record

    result = edit_node(
        "ck1",
        {"fold": 0.3, "call": 0.5, "bet_50": 0.2},
        reason="manual fix",
        _tsdb_conn=conn,
        _patch_engine=engine,
    )

    engine.apply.assert_called_once()
    args, kwargs = engine.apply.call_args
    spec = args[0]
    assert spec.source == "manual"
    assert spec.cluster_key == "ck1"
    assert spec.decision_id == "dp_real"
    assert spec.action_dist == {"fold": 0.3, "call": 0.5, "bet_50": 0.2}
    # Sentinel ValidationResult is passed via kwargs
    val = kwargs["validation"]
    assert val.zero_hits is True
    assert val.n_hands == 0
    assert val.confidence_interval == (0.0, 0.0)
    assert val.seed == 0
    assert val.ev_loss_delta == 0.0
    assert val.n_cluster_hits == 0
    # Result is JSON-serializable dict
    assert result["patch_id"] == "p123"
    assert result["new_node_id"] == "n456"
    assert result["source"] == "manual"
    assert result["reason"] == "manual fix"


def test_edit_node_no_observation_decision_id_raises():
    """Active node exists but observations carry no decision_id → fail loud."""
    conn = _mock_conn(decision_row=None)
    with pytest.raises(ValidationError, match="decision_id"):
        edit_node(
            "ck1",
            {"fold": 0.5, "call": 0.5},
            _tsdb_conn=conn,
            _patch_engine=MagicMock(),
        )


def test_edit_node_validates_before_db_lookup():
    """Validation runs BEFORE any DB call — so a bad action_dist never reaches the cursor."""
    conn = _mock_conn()
    engine = MagicMock()
    with pytest.raises(ValidationError, match="unknown"):
        edit_node(
            "ck1",
            {"fold": 0.5, "totally_made_up": 0.5},
            _tsdb_conn=conn,
            _patch_engine=engine,
        )
    # Engine never called because validation failed first
    engine.apply.assert_not_called()


# --- set_lock_from_autoloop -------------------------------------------------


def test_set_lock_from_autoloop_runs_update():
    conn = _mock_conn()
    n = set_lock_from_autoloop("ck1", True, _tsdb_conn=conn)
    assert n == 1
    call_args = conn.cursor.return_value.execute.call_args
    sql = call_args[0][0]
    assert "UPDATE strategy_nodes" in sql
    assert "locked_from_autoloop" in sql
    # Confirm parameters: (locked, cluster_key)
    params = call_args[0][1]
    assert params == (True, "ck1")


def test_set_lock_from_autoloop_unlock_path():
    conn = _mock_conn()
    n = set_lock_from_autoloop("ck1", False, _tsdb_conn=conn)
    assert n == 1
    params = conn.cursor.return_value.execute.call_args[0][1]
    assert params == (False, "ck1")
