"""Unit tests for src/study/ab.py (CLI-05).

Asserts run_ab calls src/validation/sim_ab.validate with the actual Phase 5
signature: validate(patch_spec, config, *, baseline_engine, sim_adapter, seed,
                    _tsdb_conn=None, _milvus=None).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.study.ab import run_ab


def _mock_conn(patch_row):
    """Build a MagicMock psycopg connection returning a single fetchone() row."""
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = lambda s: cm
    cm.__exit__ = lambda *a: False
    cm.fetchone.return_value = patch_row
    conn.cursor.return_value = cm
    return conn


def _patch_row(cluster_key="ck1"):
    """Construct a representative joined row from the (patches JOIN strategy_nodes) SELECT."""
    # columns: patch_id, cluster_key, source, prev_node_id, new_node_id,
    #          action_dist, embedding, gto_score, confidence, decision_id
    return (
        "p1",
        cluster_key,
        "manual",
        "prev_node",
        "new_node",
        {"fold": 0.5, "call": 0.5},
        [0.1] * 80,
        0.5,
        0.5,
        "dp_orig",
    )


def test_run_ab_calls_validate_with_phase5_signature():
    """validate(patch_spec, config, *, baseline_engine, sim_adapter, seed, ...)."""
    conn = _mock_conn(_patch_row())
    fake_result = MagicMock()
    fake_result.ev_loss_delta = 0.05
    fake_result.confidence_interval = (0.02, 0.08)
    fake_result.n_hands = 10000
    fake_result.seed = 42
    fake_result.n_cluster_hits = 100
    fake_result.zero_hits = False
    validate_mock = MagicMock(return_value=fake_result)

    out = run_ab(
        "ck1",
        "p1",
        _tsdb_conn=conn,
        _validate=validate_mock,
        _baseline_engine=MagicMock(),
        _sim_adapter=MagicMock(),
    )

    validate_mock.assert_called_once()
    args, kwargs = validate_mock.call_args
    assert len(args) == 2, f"expected (patch_spec, config) positional; got {args!r}"
    patch_spec, _config = args
    assert patch_spec.cluster_key == "ck1"
    assert "baseline_engine" in kwargs
    assert "sim_adapter" in kwargs
    assert "seed" in kwargs
    assert kwargs["seed"] == 42
    # Result unwrapped to dict
    assert out["ev_loss_delta"] == 0.05
    assert out["ci_low"] == 0.02
    assert out["ci_high"] == 0.08
    assert out["n_hands"] == 10000
    assert out["seed"] == 42
    assert out["cluster_key"] == "ck1"
    assert out["patch_id"] == "p1"


def test_run_ab_missing_patch_raises_value_error():
    conn = _mock_conn(None)
    with pytest.raises(ValueError, match="not found"):
        run_ab(
            "ck1",
            "missing-id",
            _tsdb_conn=conn,
            _validate=MagicMock(),
            _baseline_engine=MagicMock(),
            _sim_adapter=MagicMock(),
        )


def test_run_ab_cluster_key_mismatch_raises_value_error():
    """Sanity check: patches.cluster_key must match the URL/arg cluster_key."""
    conn = _mock_conn(_patch_row(cluster_key="DIFFERENT_CK"))
    with pytest.raises(ValueError, match="mismatch"):
        run_ab(
            "ck1",
            "p1",
            _tsdb_conn=conn,
            _validate=MagicMock(),
            _baseline_engine=MagicMock(),
            _sim_adapter=MagicMock(),
        )


def test_run_ab_passes_n_hands_through_config():
    """n_hands kwarg flows into AutoLoopConfig.n_hands_validation."""
    conn = _mock_conn(_patch_row())
    fake_result = MagicMock()
    fake_result.ev_loss_delta = 0.0
    fake_result.confidence_interval = (0.0, 0.0)
    fake_result.n_hands = 500
    fake_result.seed = 7
    fake_result.n_cluster_hits = 0
    fake_result.zero_hits = True
    validate_mock = MagicMock(return_value=fake_result)

    run_ab(
        "ck1",
        "p1",
        n_hands=500,
        seed=7,
        _tsdb_conn=conn,
        _validate=validate_mock,
        _baseline_engine=MagicMock(),
        _sim_adapter=MagicMock(),
    )
    _patch_spec, config = validate_mock.call_args[0]
    assert config.n_hands_validation == 500
    assert validate_mock.call_args[1]["seed"] == 7


def test_run_ab_raises_on_null_decision_id():
    """A pre-migration patch (decision_id NULL) cannot be re-keyed — run_ab fails loud."""
    row = list(_patch_row())
    row[-1] = None  # decision_id column is NULL
    conn = _mock_conn(tuple(row))

    with pytest.raises(ValueError, match="decision_id"):
        run_ab(
            "ck1",
            "p1",
            _tsdb_conn=conn,
            _validate=MagicMock(),
            _baseline_engine=MagicMock(),
            _sim_adapter=MagicMock(),
        )
