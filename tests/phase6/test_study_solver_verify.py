"""Unit tests for src/study/solver_verify.py (Stage B, D-05 + D-07).

verify_cluster is a PURE wrapper — it does NOT contain concurrency primitives.
Per RESEARCH OQ-2 (RESOLVED): the asyncio.Semaphore(1) lives in JobRegistry
(Plan 07). This module's sole responsibilities are:

1. Cache short-circuit on existing source='solver' rows
2. Resolve SolverSpot from observations
3. Call PostflopCliBackend.solve() (or injected solver)
4. Cache result via PatchEngine.apply with sentinel ValidationResult
5. Push events to job.events queue when job provided
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src._errors import SolverParseError
from src.study.solver_verify import verify_cluster


def _mock_conn(
    *,
    cached_row=None,
    spot_features=None,
    active_row=("prev_node", [0.1] * 80),
    decision_row=("dp_rep",),
):
    """MagicMock psycopg conn; fetchone yields the SELECTs in order: cache check,
    spot_features, active embedding, representative decision_id."""
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = lambda s: cm
    cm.__exit__ = lambda *a: False
    cm.fetchone.side_effect = [
        cached_row,
        (spot_features,) if spot_features is not None else None,
        active_row,
        decision_row,
    ]
    conn.cursor.return_value = cm
    return conn


def test_verify_cluster_cache_hit_skips_solver():
    conn = _mock_conn(cached_row=("existing_node",))
    solver = MagicMock()
    out = verify_cluster("ck1", _tsdb_conn=conn, _solver=solver, _patch_engine=MagicMock())
    assert out["cached"] is True
    assert out["source"] == "solver"
    solver.solve.assert_not_called()


def test_verify_cluster_no_decision_id_raises():
    """Solver runs but observations carry no decision_id → NoStrategyError (fail loud)."""
    from src._errors import NoStrategyError

    spot = {
        "board": ["Ah", "Kh", "Qh"],
        "pot": 100,
        "effective_stack": 1000,
        "prev_bet": 50,
        "range_ip": "AA",
        "range_oop": "AA",
    }
    conn = _mock_conn(spot_features=spot, active_row=("prev_node", [0.1] * 80), decision_row=None)
    solver = MagicMock()
    solver_result = MagicMock()
    solver_result.action_dist = {"fold": 1.0}
    solver_result.exploitability_pct = 1.0
    solver_result.solve_time_ms = 5
    solver.solve.return_value = solver_result

    with pytest.raises(NoStrategyError, match="decision_id"):
        verify_cluster("ck1", _tsdb_conn=conn, _solver=solver, _patch_engine=MagicMock())


def test_verify_cluster_happy_path_calls_solver_and_engine():
    spot = {
        "board": ["Ah", "Kh", "Qh"],
        "pot": 100,
        "effective_stack": 1000,
        "prev_bet": 50,
        "range_ip": "AA",
        "range_oop": "AA",
    }
    conn = _mock_conn(
        cached_row=None,
        spot_features=spot,
        active_row=("prev_node", [0.1] * 80),
    )
    solver = MagicMock()
    solver_result = MagicMock()
    solver_result.action_dist = {"fold": 0.5, "call": 0.5}
    solver_result.exploitability_pct = 2.0
    solver_result.solve_time_ms = 1500
    solver.solve.return_value = solver_result

    engine = MagicMock()
    record = MagicMock()
    record.patch_id = "p1"
    record.new_node_id = "n1"
    engine.apply.return_value = record

    out = verify_cluster("ck1", _tsdb_conn=conn, _solver=solver, _patch_engine=engine)
    assert out["cached"] is False
    assert out["source"] == "solver"
    solver.solve.assert_called_once()
    engine.apply.assert_called_once()
    spec = engine.apply.call_args[0][0]
    assert spec.source == "solver"
    assert spec.cluster_key == "ck1"
    assert spec.decision_id == "dp_rep"
    # Sentinel ValidationResult is passed in kwargs
    val = engine.apply.call_args[1]["validation"]
    assert val.zero_hits is True
    assert val.n_hands == 0


def test_verify_cluster_pushes_events():
    spot = {
        "board": [],
        "pot": 100,
        "effective_stack": 1000,
        "prev_bet": 0,
        "range_ip": "AA",
        "range_oop": "AA",
    }
    conn = _mock_conn(
        cached_row=None,
        spot_features=spot,
        active_row=("prev_node", [0.1] * 80),
    )
    solver = MagicMock()
    sr = MagicMock(action_dist={"fold": 1.0}, exploitability_pct=1.0, solve_time_ms=10)
    solver.solve.return_value = sr
    engine = MagicMock()
    engine.apply.return_value = MagicMock(patch_id="p1", new_node_id="n1")

    job = MagicMock()
    job.events = MagicMock()
    pushed: list[dict] = []
    job.events.put_nowait = lambda d: pushed.append(d)

    verify_cluster("ck1", job=job, _tsdb_conn=conn, _solver=solver, _patch_engine=engine)
    event_types = [e["event"] for e in pushed]
    assert "progress" in event_types
    assert "done" in event_types
    # At least 2 progress events (init + solving) per the plan
    assert event_types.count("progress") >= 2


def test_verify_cluster_no_observations_raises():
    """If observations have no row for cluster_key, SolverParseError fires."""
    # cache miss, then no felt_snapshot row
    conn = _mock_conn(cached_row=None, spot_features=None)
    with pytest.raises(SolverParseError, match="no backing observation"):
        verify_cluster(
            "missing_ck",
            _tsdb_conn=conn,
            _solver=MagicMock(),
            _patch_engine=MagicMock(),
        )


def test_verify_cluster_solver_error_propagates_and_pushes_event():
    spot = {
        "board": [],
        "pot": 100,
        "effective_stack": 1000,
        "prev_bet": 0,
        "range_ip": "AA",
        "range_oop": "AA",
    }
    conn = _mock_conn(cached_row=None, spot_features=spot)
    solver = MagicMock()
    solver.solve.side_effect = SolverParseError("malformed stdout")

    job = MagicMock()
    job.events = MagicMock()
    pushed: list[dict] = []
    job.events.put_nowait = lambda d: pushed.append(d)

    with pytest.raises(SolverParseError):
        verify_cluster(
            "ck1",
            job=job,
            _tsdb_conn=conn,
            _solver=solver,
            _patch_engine=MagicMock(),
        )
    event_types = [e["event"] for e in pushed]
    assert "error" in event_types


def test_verify_cluster_force_bypasses_cache():
    """force=True must skip the cache short-circuit even if a cached row exists."""
    spot = {
        "board": [],
        "pot": 100,
        "effective_stack": 1000,
        "prev_bet": 0,
        "range_ip": "AA",
        "range_oop": "AA",
    }
    # cached_row IS present, but force=True must skip it. Side-effect order
    # changes when force=True (cache SELECT is skipped, so only spot + active).
    conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = lambda s: cm
    cm.__exit__ = lambda *a: False
    cm.fetchone.side_effect = [(spot,), ("prev_node", [0.1] * 80), ("dp_rep",)]
    conn.cursor.return_value = cm

    solver = MagicMock()
    solver.solve.return_value = MagicMock(action_dist={"fold": 1.0}, exploitability_pct=1.0, solve_time_ms=5)
    engine = MagicMock()
    engine.apply.return_value = MagicMock(patch_id="p9", new_node_id="n9")

    out = verify_cluster(
        "ck1",
        force=True,
        _tsdb_conn=conn,
        _solver=solver,
        _patch_engine=engine,
    )
    assert out["cached"] is False
    solver.solve.assert_called_once()


def test_no_semaphore_in_module():
    """Concurrency control lives in JobRegistry (Plan 07), NOT in solver_verify.

    Structural test — if a Semaphore/Lock appears, we've duplicated
    concurrency control and the JobRegistry queue would be bypassable.
    """
    import inspect

    from src.study import solver_verify

    src = inspect.getsource(solver_verify)
    src_stripped = "\n".join(line for line in src.split("\n") if not line.lstrip().startswith("#"))
    assert "Semaphore" not in src_stripped, "concurrency control must live in JobRegistry, not solver_verify"
    assert "asyncio.Lock" not in src_stripped, (
        "concurrency control must live in JobRegistry, not solver_verify"
    )
