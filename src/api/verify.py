"""POST /api/verify — Stage B solver verification (async + SSE).

D-05 + D-07 + OQ-2: sequential globally. A module-level
``asyncio.Semaphore(1)`` enforces ``StudyConfig.solver_max_concurrent``
across all in-flight verify jobs (the StudyConfig field is informational here
— the actual concurrency limit is the Semaphore's initial value, kept at 1
for v1). Excess requests still create their Job in the registry and start
their task, but the task awaits the semaphore — manifest behaviour:
``status='queued'`` from outside (the asyncio.Task hasn't run mark_running
yet), then ``status='running'`` once the slot frees.

Endpoints:
    POST /api/verify                    → start a verify job.
    GET  /api/verify/sse/{job_id}       → SSE stream of progress/done/error events.
    POST /api/verify/cancel/{job_id}    → cancel an in-flight task.

The actual solver call lives in src/study/solver_verify.verify_cluster (a pure
wrapper per OQ-2 RESOLVED — Plan 05 ships it without concurrency primitives).
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
from src.api.deps import get_config, get_milvus, get_tsdb

log = get_logger("api.verify")
router = APIRouter(prefix="/api", tags=["verify"])

# Module-level Semaphore enforces sequential Stage B across all callers
# (D-05 + OQ-2). solver_verify.verify_cluster is a pure wrapper; this is the
# only concurrency primitive that gates it from the API surface.
_verify_semaphore = asyncio.Semaphore(1)


TsdbDep = Annotated[Any, Depends(get_tsdb)]
MilvusDep = Annotated[Any, Depends(get_milvus)]
ConfigDep = Annotated[Any, Depends(get_config)]


class VerifyRequest(BaseModel):
    cluster_key: str
    force: bool = False


@router.post("/verify")
async def start_verify(body: VerifyRequest, cfg: ConfigDep) -> dict:
    """Spawn an async task to run verify_cluster; return job_id immediately.

    Concurrency: a Semaphore inside ``_run_verify`` enforces sequential Stage
    B globally — but we still create the Job and start the task immediately
    so the operator gets a job_id to subscribe to via SSE.
    """
    job = api_jobs.registry.create(
        kind="verify",
        metadata={"cluster_key": body.cluster_key, "force": body.force},
    )
    # in_flight tracking is updated when the task acquires the semaphore.
    in_flight = api_jobs.registry.in_flight_count("verify")
    log.info(
        "api.verify.queued",
        job_id=job.job_id,
        cluster_key=body.cluster_key,
        in_flight=in_flight,
        max_concurrent=cfg.solver_max_concurrent,
    )
    job.task = asyncio.create_task(_run_verify(job, body.cluster_key, body.force))
    return {
        "job_id": job.job_id,
        "status": job.status,
        "cluster_key": body.cluster_key,
    }


@router.get("/verify/sse/{job_id}")
async def verify_sse(job_id: str, request: Request) -> EventSourceResponse:
    """Stream job events as Server-Sent Events. Closes on done|error or client disconnect."""
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
                # ping=15 below handles keepalive; loop and re-check disconnect.
                continue
            yield {"event": ev["event"], "data": json.dumps(ev.get("data", {}), default=str)}
            if ev["event"] in ("done", "error"):
                break

    return EventSourceResponse(gen(), ping=15)


@router.post("/verify/cancel/{job_id}")
async def cancel(job_id: str) -> dict:
    """Request cancellation of an in-flight verify job (best-effort).

    The solver runs in a thread-pool executor that cannot be interrupted, so
    cancel only injects CancelledError at the awaiting task and flags intent.
    The job's terminal status (and the semaphore release) happen in
    ``_run_verify`` once the executor thread actually exits — we never report
    'cancelled' while work is still running.
    """
    job = api_jobs.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    job.metadata["cancel_requested"] = True
    if job.task is not None and not job.task.done():
        job.task.cancel()
    return {"job_id": job_id, "status": job.status, "cancel_requested": True}


async def _run_verify(job: api_jobs.Job, cluster_key: str, force: bool) -> None:
    """Run Stage B in a thread pool (verify_cluster is sync).

    Pushes events to ``job.events`` queue:
        - progress: {stage: ...}
        - done:     {cluster_key, patch_id, ...}
        - error:    {error}

    Cancellation is best-effort: the executor thread is not interruptible, so a
    cancel waits for the in-flight solver to finish before releasing the
    semaphore — otherwise the next queued job would start while this one still
    holds the solver.
    """
    await _verify_semaphore.acquire()
    try:
        api_jobs.registry.mark_running(job)
        from src.study.solver_verify import verify_cluster

        loop = asyncio.get_event_loop()
        # Pass job through so verify_cluster can push progress events. ensure_future +
        # shield keeps the executor work running when the outer task is cancelled,
        # so we can await its true completion before releasing the semaphore.
        inner = asyncio.ensure_future(
            loop.run_in_executor(
                None,
                lambda: verify_cluster(cluster_key, force=force, job=job),
            )
        )
        try:
            result = await asyncio.shield(inner)
        except asyncio.CancelledError:
            # The executor thread can't be interrupted; wait for it to truly exit
            # before the semaphore is released so the next queued job doesn't run
            # concurrently. shield absorbs the pending cancel without abandoning inner.
            while not inner.done():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(inner)
            api_jobs.registry.mark_cancelled(job)
            raise
        api_jobs.registry.mark_done(job, result=result)
        # verify_cluster already pushes a 'done' event when given a job —
        # avoid duplicate. If it didn't (e.g. exception path), push here.
        with contextlib.suppress(Exception):
            if job.events.empty():
                await job.events.put({"event": "done", "data": result})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.error("api.verify._run_verify.failed", error=str(exc), job_id=job.job_id)
        api_jobs.registry.mark_failed(job, error=str(exc))
        with contextlib.suppress(Exception):
            await job.events.put({"event": "error", "data": {"error": str(exc)}})
    finally:
        _verify_semaphore.release()
