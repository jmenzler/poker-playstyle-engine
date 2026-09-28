"""Dashboard autoloop control: spawn/stop/resume + polled status.

Spawns the CLI as a detached child, tracks its PID atomically, delivers SIGTERM for
graceful stop, and reads heartbeat metrics + checkpoint JSON for status.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src._config import AutoLoopConfig, load_toml_config
from src._log import get_logger
from src.api.deps import get_tsdb
from src.autoloop.checkpoint import checkpoint_path_for, read_checkpoint

log = get_logger("api.autoloop")

router = APIRouter(prefix="/api", tags=["autoloop"])
TsdbDep = Annotated[Any, Depends(get_tsdb)]

_PID_FILE_DIR = Path("data/autoloop/pids")

_DEFAULT_CONFIG_PATH = Path("config/autoloop.toml")


# --- Pydantic request models -------------------------------------------------


class AutoloopRunRequest(BaseModel):
    run_id: str
    base_seed: int = 0
    max_cycles: int | None = None


class AutoloopResumeRequest(BaseModel):
    run_id: str


class AutoloopStopRequest(BaseModel):
    run_id: str


# --- Config helpers ----------------------------------------------------------


def _load_autoloop_config() -> AutoLoopConfig:
    """Load AutoLoopConfig; fall back to defaults when TOML is absent (dev/test).

    Parse errors propagate — a hand-edited config with a typo must surface as an
    actionable error, not silently revert to defaults.
    """
    if not _DEFAULT_CONFIG_PATH.exists():
        return AutoLoopConfig()
    return load_toml_config(_DEFAULT_CONFIG_PATH, AutoLoopConfig)


def _resolve_checkpoint_dir() -> Path:
    return Path(_load_autoloop_config().autonomous.checkpoint_path)


def _resolve_sentinel_path() -> Path:
    return Path(_load_autoloop_config().autonomous.sentinel_path)


# --- PID-file helpers --------------------------------------------------------


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"^[A-Za-z0-9._-]+$", run_id):
        raise ValueError(f"invalid run_id {run_id!r}: only [A-Za-z0-9._-] allowed")


def _pid_path_for(run_id: str) -> Path:
    _validate_run_id(run_id)
    return _PID_FILE_DIR / f"{run_id}.pid"


def _write_pid_file(run_id: str, pid: int) -> None:
    p = _pid_path_for(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(str(pid))
    os.replace(tmp, p)


def _read_pid_file(run_id: str) -> int | None:
    p = _pid_path_for(run_id)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# --- Core helpers ------------------------------------------------------------


def _start_run(run_id: str, base_seed: int, max_cycles: int | None) -> dict:
    _validate_run_id(run_id)
    existing_pid = _read_pid_file(run_id)
    if existing_pid is not None and _pid_alive(existing_pid):
        raise FileExistsError(f"run {run_id!r} already active (pid={existing_pid})")

    base_argv = [
        "poker-engine",
        "autoloop",
        "run",
        "--autonomous",
        "--run-id",
        run_id,
        "--seed",
        str(base_seed),
    ]
    argv = base_argv + (["--max-cycles", str(max_cycles)] if max_cycles is not None else [])

    proc = subprocess.Popen(argv, start_new_session=True)
    _write_pid_file(run_id, proc.pid)
    log.info("api.autoloop.started", run_id=run_id, pid=proc.pid)
    return {"run_id": run_id, "pid": proc.pid, "state": "running"}


def _resume_run(run_id: str) -> dict:
    _validate_run_id(run_id)
    existing_pid = _read_pid_file(run_id)
    if existing_pid is not None and _pid_alive(existing_pid):
        raise FileExistsError(f"run {run_id!r} already active (pid={existing_pid})")

    argv = [
        "poker-engine",
        "autoloop",
        "run",
        "--autonomous",
        "--resume",
        run_id,
    ]
    proc = subprocess.Popen(argv, start_new_session=True)
    _write_pid_file(run_id, proc.pid)
    log.info("api.autoloop.resumed", run_id=run_id, pid=proc.pid)
    return {"run_id": run_id, "pid": proc.pid, "state": "running"}


def _stop_run(run_id: str) -> dict:
    pid = _read_pid_file(run_id)
    if pid is None or not _pid_alive(pid):
        return {"run_id": run_id, "state": "stopped"}

    os.kill(pid, signal.SIGTERM)

    sentinel = _resolve_sentinel_path()
    with contextlib.suppress(OSError):
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()

    log.info("api.autoloop.stopped", run_id=run_id, pid=pid)
    return {"run_id": run_id, "state": "stopping"}


def _get_status(run_id: str, conn: Any) -> dict:
    pid = _read_pid_file(run_id)
    running = pid is not None and _pid_alive(pid)

    if pid is not None and not running:
        with contextlib.suppress(OSError):
            _pid_path_for(run_id).unlink(missing_ok=True)

    ckpt = read_checkpoint(checkpoint_path_for(run_id, _resolve_checkpoint_dir()))

    trend: list[float] = []
    trend_error: str | None = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM metrics WHERE metric_name='autoloop_heartbeat'"
                " AND session_id=%s ORDER BY ts DESC LIMIT 10",
                (run_id,),
            )
            rows = cur.fetchall()
            trend = [float(r[0]) for r in rows]
    except Exception as e:
        trend_error = str(e)
        log.error("api.autoloop.trend_query_failed", run_id=run_id, error=trend_error)

    state = "running" if running else ("stopped" if ckpt else "idle")
    status = {
        "run_id": run_id,
        "state": state,
        "cycles_completed": (ckpt.last_cycle + 1) if ckpt else 0,
        "patches_accepted": ckpt.patches_accepted if ckpt else 0,
        "patches_rejected": ckpt.patches_rejected if ckpt else 0,
        "ev_loss_trend": trend,
    }
    if trend_error is not None:
        status["ev_loss_trend_error"] = trend_error
    return status


# --- Endpoints ---------------------------------------------------------------


@router.post("/autoloop/run")
async def autoloop_run(body: AutoloopRunRequest, conn: TsdbDep) -> dict:
    """Start a new autonomous loop run."""
    try:
        action = _start_run(body.run_id, body.base_seed, body.max_cycles)
        return {**_get_status(body.run_id, conn), **action}
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/autoloop/resume")
async def autoloop_resume(body: AutoloopResumeRequest, conn: TsdbDep) -> dict:
    """Resume an existing run from its checkpoint."""
    try:
        action = _resume_run(body.run_id)
        return {**_get_status(body.run_id, conn), **action}
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/autoloop/stop")
async def autoloop_stop(body: AutoloopStopRequest, conn: TsdbDep) -> dict:
    """Deliver SIGTERM to a running loop. The in-flight cycle finishes before exit."""
    try:
        action = _stop_run(body.run_id)
        return {**_get_status(body.run_id, conn), **action}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/autoloop/status")
async def autoloop_status(run_id: str, conn: TsdbDep) -> dict:
    """Return current run state, progress, and ev_loss trend without attaching to the process."""
    try:
        return _get_status(run_id, conn)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
