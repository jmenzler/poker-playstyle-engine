"""Tests for src/api/verify.py cancel semantics.

The solver runs in a thread-pool executor that cannot be interrupted. Cancel is
best-effort: it must NOT release the module semaphore or report 'cancelled'
while the executor thread is still running, otherwise the next queued job runs
concurrently with the one being "cancelled".
"""

from __future__ import annotations

import asyncio
import threading

import pytest


@pytest.fixture
def block_solver(monkeypatch):
    """Replace verify_cluster with a thread-blocking stub gated by Events.

    Returns (started, release): wait on `started` to know the executor thread
    is running; set `release` to let it finish.
    """
    import src.study.solver_verify as sv

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def fake_verify_cluster(cluster_key, *, force=False, job=None):
        started.set()
        release.wait(timeout=5.0)
        finished.set()
        return {"cluster_key": cluster_key, "cached": False}

    monkeypatch.setattr(sv, "verify_cluster", fake_verify_cluster)
    return started, release, finished


async def test_cancel_does_not_release_semaphore_until_thread_exits(block_solver):
    import src.api.jobs as api_jobs
    import src.api.verify as verify

    started, release, finished = block_solver

    job = api_jobs.registry.create(kind="verify", metadata={"cluster_key": "ck1"})
    job.task = asyncio.create_task(verify._run_verify(job, "ck1", False))

    # Wait for the executor thread to actually be running the (blocked) solver.
    await asyncio.to_thread(started.wait, 5.0)
    assert started.is_set()
    assert verify._verify_semaphore.locked()  # held while solver runs

    # Cancel: best-effort. Status must NOT flip to 'cancelled' yet, and the
    # semaphore must remain held while the thread is still blocked.
    await verify.cancel(job.job_id)
    await asyncio.sleep(0.05)
    assert verify._verify_semaphore.locked(), "semaphore released before thread exited"
    assert not finished.is_set()
    assert job.status != "cancelled", "status lied: marked cancelled while work runs"

    # Let the thread finish; now the task resolves, semaphore frees, status flips.
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await job.task
    assert finished.is_set()
    assert job.status == "cancelled"
    assert not verify._verify_semaphore.locked()


async def test_normal_completion_releases_semaphore(block_solver):
    import src.api.jobs as api_jobs
    import src.api.verify as verify

    _started, release, _finished = block_solver
    release.set()  # don't block

    job = api_jobs.registry.create(kind="verify", metadata={"cluster_key": "ck2"})
    job.task = asyncio.create_task(verify._run_verify(job, "ck2", False))
    await job.task
    assert job.status == "done"
    assert not verify._verify_semaphore.locked()


async def test_cancel_sets_best_effort_flag(block_solver):
    import src.api.jobs as api_jobs
    import src.api.verify as verify

    started, release, _finished = block_solver
    job = api_jobs.registry.create(kind="verify", metadata={"cluster_key": "ck3"})
    job.task = asyncio.create_task(verify._run_verify(job, "ck3", False))
    await asyncio.to_thread(started.wait, 5.0)

    resp = await verify.cancel(job.job_id)
    assert resp["cancel_requested"] is True
    assert job.metadata.get("cancel_requested") is True

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await job.task


async def test_cancel_does_not_overwrite_done_status():
    """A cancel racing a completed job must keep the existing terminal verdict."""
    import src.api.jobs as api_jobs
    import src.api.verify as verify

    job = api_jobs.registry.create(kind="verify", metadata={"cluster_key": "ck-done"})

    async def _noop():
        return None

    job.task = asyncio.create_task(_noop())
    await job.task
    api_jobs.registry.mark_done(job, result={"cluster_key": "ck-done"})

    resp = await verify.cancel(job.job_id)

    assert job.status == "done", "cancel clobbered a terminal status"
    assert resp["status"] == "done"
