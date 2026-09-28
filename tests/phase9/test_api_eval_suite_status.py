"""src/api/eval_suite.py suite_status — survives the TTL sweep, no private _jobs scan."""

from __future__ import annotations

import src.api.eval_suite as eval_suite


async def test_suite_status_no_runs_when_unset(monkeypatch):
    monkeypatch.setattr(eval_suite, "_last_suite_result", None)
    assert await eval_suite.suite_status() == {"status": "no_runs"}


async def test_suite_status_returns_last_result_without_registry(monkeypatch):
    """The result is read from the module slot, so a swept registry still reports it."""
    payload = {"suite_pass": True, "tvd_loo": 0.012}
    monkeypatch.setattr(eval_suite, "_last_suite_result", payload)

    import src.api.jobs as api_jobs

    monkeypatch.setattr(api_jobs.registry, "_jobs", {})

    out = await eval_suite.suite_status()
    assert out == {"status": "done", "result": payload}
