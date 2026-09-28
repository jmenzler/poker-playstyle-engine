"""In-memory job registry for long-running Stage B (verify) + Eval (run_match) tasks.

Single uvicorn worker (D-02 + Pitfall 6) → module-level singleton safe. A TTL
sweeper task prunes completed (done/failed/cancelled) jobs older than
``StudyConfig.job_ttl_s`` to prevent the registry from growing unbounded
(Pitfall 1).

Concurrency control:
    Each router that spawns a long-running task takes an ``asyncio.Semaphore(1)``
    keyed by ``kind`` (see verify.py / eval.py). The registry itself tracks
    in-flight job_ids per kind so the SSE / status endpoints can report queue
    depth (``in_flight_count(kind)``).

Public surface:
    Job              — dataclass holding job_id, status, events queue, task ref.
    JobRegistry      — create/get/lifecycle/sweep helpers.
    registry         — module-level JobRegistry singleton.
    sweeper_loop     — coroutine started by main.py's lifespan handler.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src._log import get_logger

log = get_logger("api.jobs")


@dataclass
class Job:
    """Single long-running task descriptor (verify / eval / ingest).

    ``events`` is an asyncio.Queue of ``{"event": str, "data": dict}`` payloads
    that the SSE endpoint reads with ``await events.get()``. Producers push via
    ``put_nowait`` (best-effort; failures swallowed by ``_maybe_push`` patterns
    in src/study/solver_verify.py + src/eval/run_match.py).
    """

    job_id: str
    kind: str = "generic"  # "verify" | "eval" | "ingest" | "generic"
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    result: Any = None
    error: str | None = None
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    metadata: dict = field(default_factory=dict)


class JobRegistry:
    """In-memory job dict + per-kind in-flight tracker + sweep.

    Not thread-safe; safe under a single asyncio event loop (D-02 single worker).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._in_flight: dict[str, set[str]] = {}  # kind → set of in-flight job_ids

    def create(self, kind: str = "generic", metadata: dict | None = None) -> Job:
        """Create a new queued job. Caller schedules the asyncio task and
        assigns it to ``job.task`` after construction.
        """
        job = Job(job_id=str(uuid.uuid4()), kind=kind, metadata=metadata or {})
        self._jobs[job.job_id] = job
        self._in_flight.setdefault(kind, set())
        log.info("api.jobs.created", job_id=job.job_id, kind=kind)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def mark_running(self, job: Job) -> None:
        job.status = "running"
        self._in_flight.setdefault(job.kind, set()).add(job.job_id)

    def mark_done(self, job: Job, result: Any = None) -> None:
        job.status = "done"
        job.result = result
        job.completed_at = time.monotonic()
        self._in_flight.get(job.kind, set()).discard(job.job_id)

    def mark_failed(self, job: Job, error: str) -> None:
        job.status = "failed"
        job.error = error
        job.completed_at = time.monotonic()
        self._in_flight.get(job.kind, set()).discard(job.job_id)

    def mark_cancelled(self, job: Job) -> None:
        job.status = "cancelled"
        job.completed_at = time.monotonic()
        self._in_flight.get(job.kind, set()).discard(job.job_id)

    def in_flight_count(self, kind: str) -> int:
        return len(self._in_flight.get(kind, set()))

    def sweep(self, ttl_s: int) -> int:
        """Prune jobs that completed (done|failed|cancelled) AND are older than ttl_s.

        Running and queued jobs are never swept; they have no ``completed_at``.
        """
        now = time.monotonic()
        to_remove = [
            jid
            for jid, j in self._jobs.items()
            if j.status in ("done", "failed", "cancelled")
            and j.completed_at is not None
            and (now - j.completed_at) > ttl_s
        ]
        for jid in to_remove:
            del self._jobs[jid]
        if to_remove:
            log.info("api.jobs.swept", n_removed=len(to_remove))
        return len(to_remove)

    def __len__(self) -> int:
        return len(self._jobs)


# Module-level singleton — safe because D-02 mandates a single uvicorn worker.
registry = JobRegistry()


async def sweeper_loop(ttl_s: int, interval_s: float = 300.0) -> None:
    """Background coroutine: run ``registry.sweep`` every ``interval_s``.

    Started by ``src/api/main.py`` lifespan handler; cancelled on shutdown.
    Failures inside ``sweep`` are logged and swallowed so the loop keeps
    running even if a stray Job has malformed state.
    """
    while True:
        try:
            registry.sweep(ttl_s)
        except Exception as exc:
            log.error("api.jobs.sweeper_error", error=str(exc))
        await asyncio.sleep(interval_s)
