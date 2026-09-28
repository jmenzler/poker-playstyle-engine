"""Eval routes — POST /api/eval/run + SSE + GET /api/eval/leaderboard + GET /api/eval/matches.

D-NEW-30 (Phase 6 addendum #3 — Eval tab closes the trust loop):
    * POST /api/eval/run            → start an engine-vs-baseline match.
    * GET  /api/eval/sse/{job_id}   → stream per-hand progress events.
    * GET  /api/eval/leaderboard    → per-opponent aggregate stats.
    * GET  /api/eval/matches        → reverse-chrono match history with filters.

Single uvicorn worker (D-02 + Pitfall 6) → module-level Semaphore safe.
Sequential per OQ-2 — Eval matches share the solver hardware so we never
run two in parallel even though they're cheaper than Stage B.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from src._log import get_logger
from src.api import jobs as api_jobs
from src.api.deps import get_tsdb
from src.eval.baselines import REGISTRY as BASELINE_REGISTRY

log = get_logger("api.eval")
router = APIRouter(prefix="/api/eval", tags=["eval"])

# Sequential per OQ-2. Same pattern as src/api/verify.py.
_eval_semaphore = asyncio.Semaphore(1)


TsdbDep = Annotated[Any, Depends(get_tsdb)]


_MAX_TABLE_SIZE = 6


class RunMatchRequest(BaseModel):
    opponent: str | None = None  # homogeneous mode: cloned to every non-hero seat
    opponents: list[str] | None = None  # per-seat mode: one name per non-hero seat
    hands: int = 10000
    seed: int = 42
    table_size: int = 6  # 6-max by default; the corpus is 6-max (HU is out-of-distribution)


class CompareRequest(BaseModel):
    v_old: int
    v_new: int
    hands: int = 2000
    seed: int = 42


@router.post("/run")
async def run_match_endpoint(body: RunMatchRequest) -> dict:
    """Spawn an async task to run an engine-vs-baseline match.

    Rejects unknown opponents with HTTP 400 (T-06-22 mitigation: every opponent
    must be in REGISTRY — no arbitrary strategy injection from API input).
    """
    if not (2 <= body.table_size <= _MAX_TABLE_SIZE):
        raise HTTPException(status_code=400, detail=f"table_size must be 2..{_MAX_TABLE_SIZE}")

    label = body.opponent
    if body.opponents is not None:
        if len(body.opponents) != body.table_size - 1:
            raise HTTPException(
                status_code=400,
                detail=f"opponents needs {body.table_size - 1} entries for table_size={body.table_size}",
            )
        unknown = [o for o in body.opponents if o not in BASELINE_REGISTRY]
        if unknown:
            raise HTTPException(
                status_code=400, detail=f"unknown opponent(s) {unknown}; valid: {sorted(BASELINE_REGISTRY)}"
            )
        label = "+".join(body.opponents)
    elif body.opponent not in BASELINE_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(f"unknown opponent {body.opponent!r}; valid: {sorted(BASELINE_REGISTRY)}"),
        )

    job = api_jobs.registry.create(
        kind="eval",
        metadata={"opponent": label, "hands": body.hands, "seed": body.seed, "table_size": body.table_size},
    )
    log.info(
        "api.eval.queued",
        job_id=job.job_id,
        opponent=label,
        table_size=body.table_size,
        hands=body.hands,
        seed=body.seed,
    )
    job.task = asyncio.create_task(
        _run_match(job, body.opponent, body.hands, body.seed, body.table_size, body.opponents)
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "opponent": label,
        "table_size": body.table_size,
        "hands": body.hands,
    }


@router.get("/sse/{job_id}")
async def eval_sse(job_id: str, request: Request) -> EventSourceResponse:
    """Stream match progress events as SSE. Closes on done|error or disconnect."""
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


@router.post("/compare")
async def compare_endpoint(body: CompareRequest) -> dict:
    """Spawn a head-to-head match: the v_new engine vs the v_old engine."""
    if body.v_old == body.v_new:
        raise HTTPException(status_code=400, detail="v_old and v_new must differ")
    job = api_jobs.registry.create(
        kind="eval",
        metadata={"v_old": body.v_old, "v_new": body.v_new, "hands": body.hands, "seed": body.seed},
    )
    log.info(
        "api.eval.compare_queued", job_id=job.job_id, v_old=body.v_old, v_new=body.v_new, hands=body.hands
    )
    job.task = asyncio.create_task(_run_compare(job, body.v_old, body.v_new, body.hands, body.seed))
    return {
        "job_id": job.job_id,
        "status": job.status,
        "v_old": body.v_old,
        "v_new": body.v_new,
        "hands": body.hands,
    }


@router.get("/versions")
async def list_versions_endpoint(conn: TsdbDep) -> list[dict]:
    """Registered corpus versions, newest first — for the compare dropdowns."""
    from src.study import corpus_versions

    return corpus_versions.list_versions(_tsdb_conn=conn)


@router.get("/leaderboard")
async def leaderboard(conn: TsdbDep) -> list[dict]:
    """Per-opponent aggregated stats from the matches hypertable.

    Excludes non-terminal lifecycle states (queued/running/failed/cancelled) —
    only verdict-bearing rows contribute to the leaderboard. The avg over
    bb_per_100 + CI is a coarse rollup; the React panel surfaces per-match
    detail via the matches endpoint.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT opponent,
                   COUNT(*)                                       AS n_matches,
                   SUM(hands)                                     AS total_hands,
                   AVG(bb_per_100)                                AS avg_bb_per_100,
                   AVG(ci_low)                                    AS avg_ci_low,
                   AVG(ci_high)                                   AS avg_ci_high,
                   COUNT(*) FILTER (WHERE status = 'won')         AS wins,
                   COUNT(*) FILTER (WHERE status = 'lost')        AS losses,
                   COUNT(*) FILTER (WHERE status = 'inconclusive') AS ties,
                   COUNT(*) FILTER (WHERE status = 'regression')  AS regressions
            FROM matches
            WHERE status IN ('won', 'lost', 'inconclusive', 'regression')
            GROUP BY opponent
            ORDER BY avg_bb_per_100 DESC NULLS LAST
            """
        )
        rows = cur.fetchall()
        col = [d[0] for d in cur.description]
    return [dict(zip(col, r, strict=True)) for r in rows]


@router.get("/matches")
async def matches(
    conn: TsdbDep,
    opponent: str | None = None,
    status: Literal[
        "won",
        "lost",
        "inconclusive",
        "regression",
        "running",
        "queued",
        "failed",
        "cancelled",
        "any",
    ] = "any",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[dict]:
    """Reverse-chronological match history with optional opponent + status filters."""
    sql = (
        "SELECT match_id, opponent, hands, seed, bb_per_100, ci_low, ci_high, "
        "       status, engine_version, started_at, finished_at "
        "FROM matches WHERE 1=1"
    )
    params: list[Any] = []
    if opponent is not None:
        sql += " AND opponent = %s"
        params.append(opponent)
    if status != "any":
        sql += " AND status = %s"
        params.append(status)
    sql += " ORDER BY started_at DESC LIMIT %s"
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        col = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return [dict(zip(col, r, strict=True)) for r in rows]


@router.delete("/matches/{match_id}")
async def delete_match(match_id: str, conn: TsdbDep) -> dict:
    """Delete a match by id. 400 on malformed UUID, 404 when no row matched."""
    try:
        uuid.UUID(match_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid match_id: {match_id!r}") from exc
    with conn.cursor() as cur:
        cur.execute("DELETE FROM matches WHERE match_id = %s", (match_id,))
        deleted = cur.rowcount
    conn.commit()
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found")
    return {"deleted": deleted}


async def _run_match(
    job: api_jobs.Job,
    opponent: str | None,
    hands: int,
    seed: int,
    table_size: int = 6,
    opponents: list[str] | None = None,
) -> None:
    """Run an engine-vs-baseline match in a thread pool. Pushes events to job.events.

    Uses the same Semaphore(1) gate as verify (different semaphore instance,
    same idiom) so two concurrent eval requests serialize cleanly.
    """
    async with _eval_semaphore:
        api_jobs.registry.mark_running(job)
        try:
            from src.eval.run_match import run_match

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_match(
                    opponent,
                    opponents=opponents,
                    hands=hands,
                    seed=seed,
                    table_size=table_size,
                    persist=True,
                    job=job,
                ),
            )
            # MatchResult is a frozen msgspec.Struct — convert to JSON-friendly dict.
            import msgspec

            payload = msgspec.to_builtins(result)
            api_jobs.registry.mark_done(job, result=payload)
            with contextlib.suppress(Exception):
                if job.events.empty():
                    await job.events.put({"event": "done", "data": payload})
        except asyncio.CancelledError:
            api_jobs.registry.mark_cancelled(job)
            raise
        except Exception as exc:
            log.error("api.eval._run_match.failed", error=str(exc), job_id=job.job_id)
            api_jobs.registry.mark_failed(job, error=str(exc))
            with contextlib.suppress(Exception):
                await job.events.put({"event": "error", "data": {"error": str(exc)}})


async def _run_compare(job: api_jobs.Job, v_old: int, v_new: int, hands: int, seed: int) -> None:
    """Run a head-to-head version match (v_new vs v_old) as a first-class eval match.

    Goes through run_match (persist=True + job) so it lands in match history +
    leaderboard with the same progress/MatchDetail surface as a baseline match.
    """
    async with _eval_semaphore:
        api_jobs.registry.mark_running(job)
        try:
            import msgspec

            from src.decision_engine.engine import engine_from_env
            from src.eval.run_match import run_match

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_match(
                    opponent_engine=engine_from_env(as_of_version=v_old),
                    opponent_label=f"v{v_new}_vs_v{v_old}",
                    hands=hands,
                    seed=seed,
                    table_size=2,
                    persist=True,
                    _engine=engine_from_env(as_of_version=v_new),
                    job=job,
                ),
            )
            payload = msgspec.to_builtins(result)
            api_jobs.registry.mark_done(job, result=payload)
            with contextlib.suppress(Exception):
                if job.events.empty():
                    await job.events.put({"event": "done", "data": payload})
        except asyncio.CancelledError:
            api_jobs.registry.mark_cancelled(job)
            raise
        except Exception as exc:
            log.error("api.eval._run_compare.failed", error=str(exc), job_id=job.job_id)
            api_jobs.registry.mark_failed(job, error=str(exc))
            with contextlib.suppress(Exception):
                await job.events.put({"event": "error", "data": {"error": str(exc)}})
