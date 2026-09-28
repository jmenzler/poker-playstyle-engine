from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

autoloop = pytest.importorskip("src.api.autoloop")

_start_run = autoloop._start_run
_get_status = autoloop._get_status
_PID_FILE_DIR = autoloop._PID_FILE_DIR


def test_run_rejects_duplicate_if_pid_alive(tmp_path, monkeypatch):
    """_start_run raises FileExistsError when a live PID file already exists."""
    monkeypatch.setattr("src.api.autoloop._PID_FILE_DIR", tmp_path)
    (tmp_path / "test-run.pid").write_text(str(os.getpid()))
    with pytest.raises(FileExistsError):
        _start_run("test-run", base_seed=0, max_cycles=1)


def test_status_idle_when_no_pid_file(tmp_path, monkeypatch):
    """_get_status returns state=idle when no PID file exists for the run_id."""
    monkeypatch.setattr("src.api.autoloop._PID_FILE_DIR", tmp_path)
    result = _get_status("nonexistent-run", conn=MagicMock())
    assert result["state"] == "idle"


def test_status_running_when_pid_alive(tmp_path, monkeypatch):
    """_get_status returns state=running when PID file contains an alive PID."""
    monkeypatch.setattr("src.api.autoloop._PID_FILE_DIR", tmp_path)
    (tmp_path / "active-run.pid").write_text(str(os.getpid()))
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    result = _get_status("active-run", conn=mock_conn)
    assert result["state"] == "running"


def test_invalid_run_id_rejected(tmp_path, monkeypatch):
    """_start_run raises ValueError for run_id containing path-traversal characters."""
    monkeypatch.setattr("src.api.autoloop._PID_FILE_DIR", tmp_path)
    with pytest.raises(ValueError):
        _start_run("../etc/passwd", base_seed=0, max_cycles=1)


def test_status_surfaces_trend_query_error(tmp_path, monkeypatch):
    """A real DB error on the ev_loss trend query is surfaced, not masked as empty."""
    monkeypatch.setattr("src.api.autoloop._PID_FILE_DIR", tmp_path)
    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = RuntimeError("connection reset by peer")
    result = _get_status("some-run", conn=mock_conn)
    assert result["ev_loss_trend"] == []
    assert "connection reset by peer" in result["ev_loss_trend_error"]


def test_status_no_error_field_on_success(tmp_path, monkeypatch):
    """The happy path omits the error field entirely."""
    monkeypatch.setattr("src.api.autoloop._PID_FILE_DIR", tmp_path)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value.fetchall.return_value = []
    result = _get_status("some-run", conn=mock_conn)
    assert "ev_loss_trend_error" not in result
