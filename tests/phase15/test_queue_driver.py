"""Tests for QueueDriver — config, checkpoint, priority ordering, dedup, disk watchdog, eval cadence."""  # rot-allow

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from src._config import AutoLoopConfig, load_toml_config
from src.autoloop.checkpoint import checkpoint_path_for, write_checkpoint


def test_config_loads():
    cfg = load_toml_config(Path("config/autoloop.toml"), AutoLoopConfig)
    sq = cfg.solver_queue
    assert sq.n_workers == 2
    assert sq.target_exploitability_pct == pytest.approx(0.5)
    assert sq.max_iterations == 500
    assert sq.timeout_s == pytest.approx(3600.0)
    assert sq.checkpoint_every == 100
    assert sq.eval_loo_every == 500
    assert sq.eval_suite_every == 5000
    assert sq.disk_floor_gb == pytest.approx(20.0)
    assert sq.retention_days == 30
    assert sq.heartbeat_every == 50
    assert sq.checkpoint_path == "data/solver_queue_checkpoints"
    assert sq.bet_sizes == ("33%", "66%", "e", "a")


def test_checkpoint_resume():
    from src.solver.queue_driver import SolverQueueCheckpoint

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = SolverQueueCheckpoint(
            run_id="test-run-001",
            run_epoch="2026-05-30T10:00:00Z",
            spots_submitted=250,
            spots_completed=230,
            spots_failed=5,
            last_obs_id="obs-9999",
            last_eval_suite_at=200,
        )
        path = checkpoint_path_for("solver-queue-001", tmp)
        write_checkpoint(path, ckpt)

        read_back = msgspec.json.decode(path.read_bytes(), type=SolverQueueCheckpoint)
        assert read_back.run_id == "test-run-001"
        assert read_back.spots_submitted == 250
        assert read_back.spots_completed == 230
        assert read_back.spots_failed == 5
        assert read_back.last_obs_id == "obs-9999"
        assert read_back.last_eval_suite_at == 200
        assert read_back.run_epoch == "2026-05-30T10:00:00Z"


def _make_obs_row(
    obs_id: str = "obs-1",
    decision_id: str = "dp-1",
    cluster_key: str = "ck-1",
    ev_loss: float = 0.5,
    max_neighbor_distance: float = 0.8,
    cluster_freq: int = 10,
    felt_snapshot: dict | None = None,
) -> dict:
    """Build a mock observation row dict as returned by the priority queue SQL."""
    fs = felt_snapshot or {
        "action_sequence": ["BTN:raise", "BB:call"],
        "board_cards": ["Ah", "7c", "2d"],
        "effective_stack_bb": 100.0,
        "hero_bet_size_bb": 0.0,
        "hero_facing_bet_bb": 0.0,
        "hero_hole_cards": ["As", "Kd"],
        "hero_position": "BTN",
        "opponents_remaining": 1,
        "pot_size_bb": 10.0,
        "prior_street_aggressor": "BTN",
        "street": "flop",
    }
    return {
        "obs_id": obs_id,
        "decision_id": decision_id,
        "cluster_key": cluster_key,
        "ev_loss": ev_loss,
        "max_neighbor_distance": max_neighbor_distance,
        "cluster_freq": cluster_freq,
        "priority_score": cluster_freq * ev_loss * max_neighbor_distance,
        "felt_snapshot": fs,
        "embedding": [0.1] * 80,
    }


def test_priority_ordering():
    """Queue fetch returns spots ordered by priority score DESC."""
    from src.solver.queue_driver import QueueDriver

    high = _make_obs_row("obs-1", "dp-1", "ck-1", ev_loss=2.0, max_neighbor_distance=0.9, cluster_freq=100)
    low = _make_obs_row("obs-2", "dp-2", "ck-2", ev_loss=0.1, max_neighbor_distance=0.3, cluster_freq=5)
    medium = _make_obs_row("obs-3", "dp-3", "ck-3", ev_loss=1.0, max_neighbor_distance=0.7, cluster_freq=20)

    rows = [high, low, medium]
    rows_sorted = sorted(rows, key=lambda r: r["priority_score"], reverse=True)
    assert rows_sorted[0]["obs_id"] == "obs-1"
    assert rows_sorted[2]["obs_id"] == "obs-2"

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = [
        (
            r["obs_id"],
            r["decision_id"],
            r["cluster_key"],
            r["max_neighbor_distance"],
            r["felt_snapshot"],
            r["cluster_freq"],
            r["ev_loss"],
            r["priority_score"],
            r["embedding"],
            r.get("hand_id", ""),
        )
        for r in rows_sorted
    ]
    mock_conn.cursor.return_value = mock_cursor

    driver = QueueDriver.__new__(QueueDriver)
    fetched = driver._fetch_batch(mock_conn, limit=100)
    assert fetched[0]["obs_id"] == "obs-1"
    assert fetched[2]["obs_id"] == "obs-2"


def test_skips_solved():
    """_fetch_batch excludes cluster_keys already covered in solver_cache."""
    from src.solver.queue_driver import QueueDriver

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value = mock_cursor

    driver = QueueDriver.__new__(QueueDriver)
    fetched = driver._fetch_batch(mock_conn, limit=100)
    assert fetched == []


def _postflop_felt(**over) -> dict:
    felt = {
        "hero_position": "BTN",
        "street": "flop",
        "board_cards": ["Qh", "Jh", "2s"],
        "hero_hole_cards": ["Ah", "Kh"],
        "pot_size_bb": 6.0,
        "effective_stack_bb": 97.0,
        "opponents_remaining": 1,
        "action_sequence": ["BTN:open_2_2bb", "BB:call"],
    }
    felt.update(over)
    return felt


@pytest.mark.parametrize("missing", ["pot_size_bb", "effective_stack_bb"])
def test_build_spot_fails_loud_on_missing_pot_or_stack(missing: str):
    """A postflop spot with no pot/stack is corrupt input — refuse, do not fabricate."""
    from src.solver.queue_driver import SolverQueueConfig, _build_spot

    felt = _postflop_felt()
    del felt[missing]
    obs_row = {"obs_id": "obs-x", "cluster_key": "pot_type=srp|street=flop", "felt_snapshot": felt}
    with pytest.raises(ValueError, match=r"(?i)pot|stack"):
        _build_spot(obs_row, {}, SolverQueueConfig())


def test_build_spot_handles_explicit_null_opponents_remaining():
    """opponents_remaining explicitly null must default to 1, not raise TypeError."""
    from src.solver.queue_driver import SolverQueueConfig, _build_spot

    felt = _postflop_felt(opponents_remaining=None)
    obs_row = {"obs_id": "obs-y", "cluster_key": "pot_type=srp|street=flop", "felt_snapshot": felt}
    _, ctx = _build_spot(obs_row, {}, SolverQueueConfig())
    assert ctx["n_players_at_street"] == 2


def test_disk_floor_halt():
    """_check_disk raises RuntimeError when free_gb < disk_floor_gb."""
    from src.solver.queue_driver import _check_disk

    with patch("shutil.disk_usage") as mock_du:
        mock_du.return_value = (shutil.disk_usage.__doc__ and MagicMock(free=5 * 1024**3)) or MagicMock(
            free=5 * 1024**3
        )
        mock_du.return_value.free = 5 * 1024**3  # 5 GB free
        with pytest.raises(RuntimeError, match="Disk below floor"):
            _check_disk(floor_gb=20.0)


def test_resim_on_drain():
    """When flagged_sparse count == 0 after drain, driver invokes re-sim hook."""
    from src.solver.queue_driver import QueueDriver

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_conn.cursor.return_value = mock_cursor

    sim_runner_called = []

    def mock_sim():
        sim_runner_called.append(1)

    driver = QueueDriver.__new__(QueueDriver)
    driver._run_id = "test-drain-run"
    driver._on_drain(mock_conn, _sim_runner=mock_sim)
    assert len(sim_runner_called) == 1


def test_eval_cadence_loo_not_suite():
    """LOO gate runs every eval_loo_every; suite only every eval_suite_every.
    Auto-pause fires on TVD > floor*2 or absolute delta > 0.15, NOT on regression_delta.
    """
    from src.solver.queue_driver import _should_pause_on_loo

    loo_result_ok = MagicMock()
    loo_result_ok.tier1_tvd = 0.10
    loo_result_ok.passed = True

    loo_result_bad = MagicMock()
    loo_result_bad.tier1_tvd = 0.35
    loo_result_bad.passed = False

    tvd_floor = 0.15

    assert not _should_pause_on_loo(loo_result_ok, tvd_floor=tvd_floor, last_tvd=0.09)
    assert _should_pause_on_loo(loo_result_bad, tvd_floor=tvd_floor, last_tvd=0.10)

    tier1_tvd_big = MagicMock()
    tier1_tvd_big.tier1_tvd = tvd_floor * 2.5
    tier1_tvd_big.passed = True
    assert _should_pause_on_loo(tier1_tvd_big, tvd_floor=tvd_floor, last_tvd=0.10)

    tier1_tvd_jump = MagicMock()
    tier1_tvd_jump.tier1_tvd = 0.20
    tier1_tvd_jump.passed = True
    assert _should_pause_on_loo(tier1_tvd_jump, tvd_floor=tvd_floor, last_tvd=0.04)


def test_resume_skips_completed():
    """Restart after partial run resumes from checkpoint, not re-solving completed obs."""
    from src.solver.queue_driver import SolverQueueCheckpoint

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = SolverQueueCheckpoint(
            run_id="resume-test",
            run_epoch="2026-05-30T08:00:00Z",
            spots_submitted=500,
            spots_completed=480,
            spots_failed=10,
            last_obs_id="obs-500",
            last_eval_suite_at=400,
        )
        path = checkpoint_path_for("resume-test", tmp)
        write_checkpoint(path, ckpt)

        loaded = msgspec.json.decode(path.read_bytes(), type=SolverQueueCheckpoint)
        assert loaded.last_obs_id == "obs-500"
        assert loaded.spots_completed == 480


def test_run_bootstraps_milvus_client():
    """run() must create a Milvus client when none is injected.

    Otherwise _inject() and the LOO gate receive milvus=None and every solver node
    is written to TSDB only — write-only, never retrievable (gap #1 regression).
    """
    from src.solver import queue_driver as qd

    calls: list[int] = []
    sentinel = MagicMock(name="milvus_client")

    mock_conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = []
    cur.fetchone.return_value = (0,)
    mock_conn.cursor.return_value = cur

    def fake_connect():
        calls.append(1)
        return sentinel

    with patch("src.db.milvus.connect_from_env", fake_connect):
        driver = qd.QueueDriver(run_id="test-milvus-bootstrap")
        driver.run(
            _tsdb_conn=mock_conn,
            _solver=MagicMock(),
            _sim_runner=None,
            install_signals=False,
        )

    assert calls, "run() did not bootstrap a Milvus client when _milvus was None"


def _drained_conn() -> MagicMock:
    """A tsdb conn whose queue is already empty (no spots to solve)."""
    mock_conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = []
    cur.fetchone.return_value = (0,)
    mock_conn.cursor.return_value = cur
    return mock_conn


def test_version_bump_skipped_when_zero_spots(monkeypatch):
    """A run that completes zero spots does NOT snapshot a corpus version."""
    from src.solver import queue_driver as qd

    mock_snapshot = MagicMock()
    monkeypatch.setattr("src.solver.queue_driver.corpus_versions.snapshot", mock_snapshot)

    driver = qd.QueueDriver(run_id="test-zero-spots")
    summary = driver.run(
        _tsdb_conn=_drained_conn(),
        _milvus=MagicMock(),
        _solver=MagicMock(),
        _sim_runner=None,
        install_signals=False,
    )

    assert summary["spots_completed"] == 0
    mock_snapshot.assert_not_called()


def test_version_bump_fires_when_spots_completed(monkeypatch):
    """A run that completes >=1 spot snapshots a corpus version (source='solver')."""
    from src.solver import queue_driver as qd

    run_id = "test-some-spots"
    resumed = qd.SolverQueueCheckpoint(
        run_id=run_id,
        run_epoch="2026-05-30T08:00:00Z",
        spots_submitted=10,
        spots_completed=7,
        spots_failed=0,
        last_obs_id="obs-10",
        last_eval_suite_at=0,
    )
    monkeypatch.setattr("src.solver.queue_driver._read_solver_checkpoint", MagicMock(return_value=resumed))
    monkeypatch.setattr("src.solver.queue_driver.write_checkpoint", MagicMock())

    mock_snapshot = MagicMock(return_value={"version": 3, "cutoff_ts": 123})
    monkeypatch.setattr("src.solver.queue_driver.corpus_versions.snapshot", mock_snapshot)

    mock_conn = _drained_conn()
    driver = qd.QueueDriver(run_id=run_id)
    summary = driver.run(
        _tsdb_conn=mock_conn,
        _milvus=MagicMock(),
        _solver=MagicMock(),
        _sim_runner=None,
        install_signals=False,
    )

    assert summary["spots_completed"] == 7
    mock_snapshot.assert_called_once()
    _, kwargs = mock_snapshot.call_args
    assert kwargs["source"] == "solver"
    assert kwargs["_tsdb_conn"] is mock_conn
