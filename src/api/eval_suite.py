"""POST /api/eval/suite — full eval suite endpoint with SSE progress.

Security mitigations (T-9-18, T-9-19):
    SuiteRequest: pydantic rejects non-int hands/seed (input validation).
    asyncio.Semaphore(1): prevents concurrent suite runs (DoS / CPU exhaustion).
    loop.run_in_executor: keeps blocking suite off the FastAPI event loop (T-9-19).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from src._log import get_logger
from src.api import jobs as api_jobs
from src.api.deps import get_tsdb

log = get_logger("api.eval_suite")
router = APIRouter(prefix="/api/eval", tags=["eval"])

_eval_suite_semaphore = asyncio.Semaphore(1)

# Last completed suite result, kept outside the TTL-swept job registry so
# /suite/status survives the sweep (the registry prunes done jobs).
_last_suite_result: Any = None

TsdbDep = Annotated[Any, Depends(get_tsdb)]


class SuiteRequest(BaseModel):
    hands: int = 10000
    seed: int = 42


@router.post("/suite")
async def run_suite_endpoint(body: SuiteRequest) -> dict:
    """Start an async eval suite run. Returns job_id for SSE tracking."""
    job = api_jobs.registry.create(
        kind="eval-suite",
        metadata={"hands": body.hands, "seed": body.seed},
    )
    log.info("api.eval_suite.queued", job_id=job.job_id, hands=body.hands, seed=body.seed)
    job.task = asyncio.create_task(_run_suite(job, body.hands, body.seed))
    return {
        "job_id": job.job_id,
        "status": job.status,
        "hands": body.hands,
    }


@router.get("/suite/sse/{job_id}")
async def suite_sse(job_id: str, request: Request) -> EventSourceResponse:
    """Stream suite progress events as SSE. Closes on done|error or disconnect."""
    job = api_jobs.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")

    async def gen():
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


@router.get("/suite/status")
async def suite_status() -> dict:
    """Return the most recent completed suite result (survives the TTL sweep)."""
    if _last_suite_result is None:
        return {"status": "no_runs"}
    return {"status": "done", "result": _last_suite_result}


async def _run_suite(job: api_jobs.Job, hands: int, seed: int) -> None:
    """Run eval suite in a thread executor. Pushes events to job.events.

    Uses Semaphore(1) to serialize suite runs (solver + matches are blocking
    and CPU-intensive; T-9-18/T-9-19 mitigations).
    """
    async with _eval_suite_semaphore:
        api_jobs.registry.mark_running(job)
        try:
            from src.eval.suite import run_suite

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_suite(hands=hands, seed=seed, job=job),
            )
            import msgspec

            payload = msgspec.to_builtins(result)
            global _last_suite_result
            _last_suite_result = payload
            api_jobs.registry.mark_done(job, result=payload)
            with contextlib.suppress(Exception):
                if job.events.empty():
                    await job.events.put({"event": "done", "data": payload})
        except asyncio.CancelledError:
            api_jobs.registry.mark_cancelled(job)
            raise
        except Exception as exc:
            log.error("api.eval_suite._run_suite.failed", error=str(exc), job_id=job.job_id)
            api_jobs.registry.mark_failed(job, error=str(exc))
            with contextlib.suppress(Exception):
                await job.events.put({"event": "error", "data": {"error": str(exc)}})
