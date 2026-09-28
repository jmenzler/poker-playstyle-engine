"""POST /api/hands/by-hand/{hand_id}/solve-ranges (SSE) + GET ranges."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse

from src._log import get_logger
from src.api import jobs as api_jobs
from src.api.deps import get_tsdb
from src.db import timescale
from src.study.hand_ranges import load_hand_ranges_contract, solve_and_persist_hand_ranges

log = get_logger("api.hand_ranges")
router = APIRouter(prefix="/api", tags=["hand-ranges"])

_solve_semaphore = asyncio.Semaphore(1)
_HAND_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

TsdbDep = Annotated[Any, Depends(get_tsdb)]


def _tsdb_dsn_from_env() -> str:
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


@router.post("/hands/by-hand/{hand_id}/solve-ranges")
async def start_solve_ranges(
    hand_id: str,
    force: bool = False,
) -> dict[str, Any]:
    """Spawn an async solve task; return job_id immediately."""
    if not _HAND_ID_RE.match(hand_id):
        raise HTTPException(status_code=400, detail="hand_id must be alphanumeric, underscore, or hyphen")
    job = api_jobs.registry.create(
        kind="solve_ranges",
        metadata={"hand_id": hand_id, "force": force},
    )
    job.task = asyncio.create_task(_run_solve(job, hand_id, force))
    log.info("api.hand_ranges.queued", job_id=job.job_id, hand_id=hand_id)
    return {"job_id": job.job_id, "status": job.status, "hand_id": hand_id}


@router.get("/hands/by-hand/{hand_id}/sse/{job_id}")
async def solve_sse(hand_id: str, job_id: str, request: Request) -> EventSourceResponse:
    """Stream solve-ranges job events as SSE."""
    job = api_jobs.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")

    async def gen() -> Any:
        while True:
            if await request.is_disconnected():
                break
            try:
                ev = await asyncio.wait_for(job.events.get(), timeout=20.0)
            except TimeoutError:
                continue
            yield {"event": ev["event"], "data": json.dumps(ev.get("data", {}), default=str)}
            if ev["event"] in ("done", "error"):
                break

    return EventSourceResponse(gen(), ping=15)


@router.get("/hands/by-hand/{hand_id}/ranges")
async def get_ranges(hand_id: str, conn: TsdbDep) -> list[dict[str, Any]]:
    """Return the frozen per-DP dual-seat contract. Empty list if not yet solved."""
    if not _HAND_ID_RE.match(hand_id):
        raise HTTPException(status_code=400, detail="hand_id must be alphanumeric, underscore, or hyphen")
    return load_hand_ranges_contract(hand_id, conn=conn)


async def _run_solve(job: api_jobs.Job, hand_id: str, force: bool) -> None:
    """Run solve_and_persist_hand_ranges in a thread pool under the semaphore."""
    async with _solve_semaphore:
        api_jobs.registry.mark_running(job)
        try:
            loop = asyncio.get_event_loop()

            def _solve() -> dict[str, Any]:
                # autocommit: reads at the top of solve_and_persist would otherwise open a
                # transaction, turning each persist_harvest_range conn.transaction() into a
                # nested savepoint that never commits the outer tx (rows lost on close).
                conn = timescale.connect(_tsdb_dsn_from_env(), autocommit=True)
                try:
                    return solve_and_persist_hand_ranges(hand_id, force=force, conn=conn, job=job)
                finally:
                    conn.close()

            result = await loop.run_in_executor(None, _solve)
            api_jobs.registry.mark_done(job, result=result)
            with contextlib.suppress(Exception):
                if job.events.empty():
                    await job.events.put({"event": "done", "data": result})
        except asyncio.CancelledError:
            api_jobs.registry.mark_cancelled(job)
            raise
        except Exception as exc:
            log.error("api.hand_ranges._run_solve.failed", error=str(exc), job_id=job.job_id)
            api_jobs.registry.mark_failed(job, error=str(exc))
            with contextlib.suppress(Exception):
                await job.events.put({"event": "error", "data": {"error": str(exc)}})


def _build_solver_path() -> Path:
    return Path(os.environ.get("POSTFLOP_CLI_BIN", "~/postflop-cli/target/release/postflop-cli")).expanduser()
