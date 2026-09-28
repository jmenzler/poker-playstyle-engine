"""Shared fixtures for orchestrator tests — fakes that let the loop run with no DB/Milvus/sim."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


class CallLog:
    """Shared ordered record of sim/step calls so tests can assert sim-before-step."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.sim_calls: list[dict] = []
        self.step_calls: list[int] = []
        self.heartbeats: list[dict] = []


class FakeSimRunner:
    """Records each per-cycle sim invocation; returns a dummy summary.

    Optionally raises a configured exception on specific cycle indices to drive
    retry / skip / circuit-breaker paths.
    """

    def __init__(
        self,
        call_log: CallLog,
        *,
        raise_map: dict[int, BaseException] | None = None,
        raise_once: dict[int, BaseException] | None = None,
    ) -> None:
        self._log = call_log
        self._raise_map = raise_map or {}
        self._raise_once = dict(raise_once or {})

    def __call__(
        self,
        *,
        run_id: str,
        cycle_idx: int,
        cycle_seed: int,
        session_id: str,
        session_started_at: datetime,
        n_hands: int,
    ) -> dict:
        if cycle_idx in self._raise_once:
            exc = self._raise_once.pop(cycle_idx)
            raise exc
        if cycle_idx in self._raise_map:
            raise self._raise_map[cycle_idx]
        self._log.events.append(("sim", cycle_idx))
        self._log.sim_calls.append(
            {
                "run_id": run_id,
                "cycle_idx": cycle_idx,
                "cycle_seed": cycle_seed,
                "session_id": session_id,
                "session_started_at": session_started_at,
                "n_hands": n_hands,
            }
        )
        return {"session_id": session_id}


class FakeDriver:
    """step() returns a configurable patches-applied count and records call order.

    ``raise_once`` lets a test make the Nth step() call raise (counting from the
    total number of step invocations so far) to drive the step-only retry path.
    """

    def __init__(
        self,
        call_log: CallLog,
        *,
        applied: int = 1,
        raise_once: dict[int, BaseException] | None = None,
    ) -> None:
        self._log = call_log
        self._applied = applied
        self._raise_once = dict(raise_once or {})
        self._step_attempts = 0

    def step(self) -> int:
        attempt = self._step_attempts
        self._step_attempts += 1
        if attempt in self._raise_once:
            raise self._raise_once.pop(attempt)
        # cycle_idx is the count of step calls so far (loop calls sim then step)
        self._log.events.append(("step", len(self._log.step_calls)))
        self._log.step_calls.append(self._applied)
        return self._applied


class FakeMetricsWriter:
    """Records each heartbeat row instead of writing to TSDB."""

    def __init__(self, call_log: CallLog) -> None:
        self._log = call_log

    def __call__(self, *, run_id: str, value: float, ts: datetime) -> None:
        self._log.heartbeats.append({"run_id": run_id, "value": value, "ts": ts})


@pytest.fixture
def call_log() -> CallLog:
    return CallLog()


@pytest.fixture
def fake_sim_runner(call_log: CallLog) -> FakeSimRunner:
    return FakeSimRunner(call_log)


@pytest.fixture
def fake_driver(call_log: CallLog) -> FakeDriver:
    return FakeDriver(call_log, applied=1)


@pytest.fixture
def capture_metrics(call_log: CallLog) -> FakeMetricsWriter:
    return FakeMetricsWriter(call_log)


@pytest.fixture
def frozen_epoch() -> datetime:
    return datetime(2026, 5, 21, 0, 0, 0, tzinfo=UTC)
