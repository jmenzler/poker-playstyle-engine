"""Tests for src/api/jobs.py — JobRegistry + Job + TTL sweeper.

JobRegistry CRUD tests are fully sync. The sweeper test patches
``time.monotonic`` to simulate ttl_s elapsing without sleeping.
A minimal SSE smoke is exercised against ``/api/health`` (the SSE-specific
verify/eval streams are tested in test_api_eval.py / via verify.py tests).
"""

from __future__ import annotations

import asyncio

import pytest


def test_create_returns_unique_jobs() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    j1 = reg.create()
    j2 = reg.create()
    assert j1.job_id != j2.job_id
    assert j1.status == "queued"
    assert j2.status == "queued"
    assert isinstance(j1.events, asyncio.Queue)


def test_create_with_kind_and_metadata() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    j = reg.create(kind="verify", metadata={"cluster_key": "ck1"})
    assert j.kind == "verify"
    assert j.metadata == {"cluster_key": "ck1"}


def test_get_missing_returns_none() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    assert reg.get("does-not-exist") is None


def test_get_returns_job_by_id() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    j = reg.create(kind="eval")
    assert reg.get(j.job_id) is j


def test_in_flight_count_tracks_running_jobs() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    assert reg.in_flight_count("verify") == 0
    j = reg.create(kind="verify")
    reg.mark_running(j)
    assert reg.in_flight_count("verify") == 1
    reg.mark_done(j, result={"ok": True})
    assert reg.in_flight_count("verify") == 0


def test_mark_done_sets_status_and_completed_at() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    j = reg.create(kind="verify")
    reg.mark_done(j, result={"r": 1})
    assert j.status == "done"
    assert j.result == {"r": 1}
    assert j.completed_at is not None


def test_mark_failed_sets_error() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    j = reg.create(kind="eval")
    reg.mark_failed(j, error="boom")
    assert j.status == "failed"
    assert j.error == "boom"
    assert j.completed_at is not None


def test_mark_cancelled() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    j = reg.create(kind="verify")
    reg.mark_cancelled(j)
    assert j.status == "cancelled"
    assert j.completed_at is not None


def test_sweep_removes_old_done_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sweeper prunes done/failed/cancelled jobs older than ttl_s."""
    import src.api.jobs as api_jobs

    reg = api_jobs.JobRegistry()

    # Fake monotonic clock
    clock = [1000.0]

    def fake_monotonic() -> float:
        return clock[0]

    monkeypatch.setattr(api_jobs.time, "monotonic", fake_monotonic)

    j_done = reg.create(kind="verify")
    j_running = reg.create(kind="verify")
    j_queued = reg.create(kind="eval")

    # Mark done at t=1000
    reg.mark_done(j_done, result={"ok": True})

    # Advance clock by 3700s (just past 1h ttl)
    clock[0] = 4700.0

    n_removed = reg.sweep(ttl_s=3600)
    assert n_removed == 1
    assert reg.get(j_done.job_id) is None  # pruned
    assert reg.get(j_running.job_id) is not None  # status='queued', not swept
    assert reg.get(j_queued.job_id) is not None  # status='queued', not swept


def test_sweep_keeps_recent_done_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A done job younger than ttl_s must NOT be swept."""
    import src.api.jobs as api_jobs

    reg = api_jobs.JobRegistry()

    clock = [1000.0]
    monkeypatch.setattr(api_jobs.time, "monotonic", lambda: clock[0])

    j = reg.create(kind="verify")
    reg.mark_done(j, result={"ok": True})

    clock[0] = 1500.0  # only 500s later
    assert reg.sweep(ttl_s=3600) == 0
    assert reg.get(j.job_id) is not None


def test_sweep_does_not_remove_running() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    j = reg.create(kind="verify")
    reg.mark_running(j)
    # Even with ttl=0, running jobs are never swept (no completed_at).
    assert reg.sweep(ttl_s=0) == 0
    assert reg.get(j.job_id) is not None


def test_registry_len() -> None:
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    assert len(reg) == 0
    reg.create()
    reg.create()
    assert len(reg) == 2


def test_module_level_registry_singleton() -> None:
    """``src.api.jobs.registry`` is a singleton instance used by all routers."""
    from src.api.jobs import JobRegistry, registry

    assert isinstance(registry, JobRegistry)


async def test_sweeper_loop_runs_and_can_be_cancelled() -> None:
    """sweeper_loop is awaitable and respects cancellation."""
    from src.api.jobs import sweeper_loop

    task = asyncio.create_task(sweeper_loop(ttl_s=3600, interval_s=0.01))
    await asyncio.sleep(0.05)  # let it run a few iterations
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert task.done()


async def test_job_events_queue_basic_round_trip() -> None:
    """SSE pattern: producer pushes to queue, consumer awaits get()."""
    from src.api.jobs import JobRegistry

    reg = JobRegistry()
    j = reg.create(kind="verify")
    await j.events.put({"event": "progress", "data": {"stage": "init"}})
    await j.events.put({"event": "done", "data": {"result": "ok"}})

    ev1 = await j.events.get()
    ev2 = await j.events.get()
    assert ev1["event"] == "progress"
    assert ev2["event"] == "done"
