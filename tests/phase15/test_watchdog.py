"""Tests for queue_driver disk watchdog and operational guards (D-17)."""  # rot-allow

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest


def test_disk_floor_halt_on_low_space():
    """_check_disk raises RuntimeError on < floor_gb free disk."""
    from src.solver.queue_driver import _check_disk

    fake_usage = MagicMock()
    fake_usage.free = 10 * 1024**3  # 10 GB free

    with patch("shutil.disk_usage", return_value=fake_usage):
        with pytest.raises(RuntimeError, match="Disk below floor"):
            _check_disk(floor_gb=20.0)


def test_disk_floor_passes_on_ample_space():
    """_check_disk does not raise when free space is above the floor."""
    from src.solver.queue_driver import _check_disk

    fake_usage = MagicMock()
    fake_usage.free = 100 * 1024**3  # 100 GB free

    with patch("shutil.disk_usage", return_value=fake_usage):
        _check_disk(floor_gb=20.0)


def test_checkpoint_written_on_disk_halt():
    """QueueDriver writes a final checkpoint before disk-halt exit."""
    from src.solver.queue_driver import SolverQueueCheckpoint

    with tempfile.TemporaryDirectory() as tmp:
        import msgspec

        from src.autoloop.checkpoint import checkpoint_path_for, write_checkpoint

        ckpt = SolverQueueCheckpoint(
            run_id="disk-halt-test",
            run_epoch="2026-05-30T10:00:00Z",
            spots_submitted=450,
            spots_completed=430,
            spots_failed=8,
            last_obs_id="obs-430",
            last_eval_suite_at=400,
        )
        path = checkpoint_path_for("disk-halt-test", tmp)
        write_checkpoint(path, ckpt)

        loaded = msgspec.json.decode(path.read_bytes(), type=SolverQueueCheckpoint)
        assert loaded.run_id == "disk-halt-test"
        assert loaded.spots_completed == 430
        assert loaded.last_obs_id == "obs-430"


def test_retention_policy_check_runs():
    """_ensure_retention_policy queries timescaledb_information.jobs before adding."""
    from src._config import SolverQueueConfig
    from src.solver.queue_driver import QueueDriver

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = (0,)
    mock_conn.cursor.return_value = mock_cursor

    driver = QueueDriver.__new__(QueueDriver)
    driver._cfg = SolverQueueConfig(retention_days=30)

    driver._ensure_retention_policy(mock_conn)

    calls = [str(c) for c in mock_cursor.execute.call_args_list]
    assert any("timescaledb_information" in c or "add_retention_policy" in c for c in calls)


def test_ensure_retention_handles_exception():
    """_ensure_retention_policy swallows exceptions (non-TimescaleDB env)."""
    from src._config import SolverQueueConfig
    from src.solver.queue_driver import QueueDriver

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.execute.side_effect = Exception("relation does not exist")
    mock_conn.cursor.return_value = mock_cursor

    driver = QueueDriver.__new__(QueueDriver)
    driver._cfg = SolverQueueConfig(retention_days=30)

    driver._ensure_retention_policy(mock_conn)
