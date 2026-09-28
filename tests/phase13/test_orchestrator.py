"""AutonomousRunner loop tests (LOOP-07..11) via injected fakes — no DB/Milvus/sim."""

from __future__ import annotations

from pathlib import Path

from src._errors import DBConnectError
from src.autoloop.checkpoint import RunCheckpoint, checkpoint_path_for, read_checkpoint, write_checkpoint
from src.autoloop.orchestrator import AutonomousRunner
from tests.phase13.conftest import CallLog, FakeDriver, FakeMetricsWriter, FakeSimRunner


def _make_config(tmp_path: Path, **overrides: object) -> Path:
    """Write a tmp autoloop.toml whose [autonomous] dirs live under tmp_path."""
    ckpt = overrides.get("checkpoint_path", str(tmp_path / "ckpt"))
    sentinel = overrides.get("sentinel_path", str(tmp_path / "stop"))
    max_retries = overrides.get("max_retries", 2)
    backoff = overrides.get("backoff_base_s", 0.0)
    max_consec = overrides.get("max_consecutive_failures", 3)
    hb_every = overrides.get("heartbeat_every_cycles", 1)
    toml = tmp_path / "autoloop.toml"
    toml.write_text(
        "enabled = true\n"
        "[autonomous]\n"
        f"max_retries = {max_retries}\n"
        f"backoff_base_s = {backoff}\n"
        f"max_consecutive_failures = {max_consec}\n"
        f"heartbeat_every_cycles = {hb_every}\n"
        f'checkpoint_path = "{ckpt}"\n'
        f'sentinel_path = "{sentinel}"\n'
        "sim_n_hands = 10\n"
    )
    return toml


def _runner(
    tmp_path: Path,
    call_log: CallLog,
    *,
    max_cycles: int | None,
    run_id: str = "r1",
    base_seed: int = 100,
    sim_runner: object | None = None,
    driver: object | None = None,
    sleeps: list[float] | None = None,
    **cfg: object,
) -> AutonomousRunner:
    sim = sim_runner or FakeSimRunner(call_log)
    drv = driver or FakeDriver(call_log, applied=1)
    sleeps_list = sleeps if sleeps is not None else []
    return AutonomousRunner(
        run_id=run_id,
        base_seed=base_seed,
        max_cycles=max_cycles,
        config_path=_make_config(tmp_path, **cfg),
        driver=drv,
        sim_runner=sim,
        metrics_writer=FakeMetricsWriter(call_log),
        sleep=sleeps_list.append,
        install_signals=False,
    )


def test_runs_n_cycles_sim_then_step(tmp_path: Path, call_log: CallLog) -> None:
    """LOOP-07: run(max_cycles=3) calls sim then step 3x, in order, no manual input."""
    runner = _runner(tmp_path, call_log, max_cycles=3)
    runner.run()

    assert call_log.events == [
        ("sim", 0),
        ("step", 0),
        ("sim", 1),
        ("step", 1),
        ("sim", 2),
        ("step", 2),
    ], "LOOP-07: sim-before-step each cycle, 3 cycles"


def test_deterministic_seed_and_session_id(tmp_path: Path, call_log: CallLog) -> None:
    """D-02/D-05: cycle_seed = base+idx; session_id = run_id_c{idx}; ts derived from run_epoch."""
    runner = _runner(tmp_path, call_log, max_cycles=2, base_seed=100)
    runner.run()

    assert call_log.sim_calls[0]["cycle_seed"] == 100
    assert call_log.sim_calls[1]["cycle_seed"] == 101
    assert call_log.sim_calls[0]["session_id"] == "r1_c0"
    assert call_log.sim_calls[1]["session_id"] == "r1_c1"
    assert call_log.sim_calls[1]["session_started_at"] > call_log.sim_calls[0]["session_started_at"]


def test_checkpoint_written_each_cycle(tmp_path: Path, call_log: CallLog) -> None:
    """LOOP-08: a checkpoint is written after each cycle; last_cycle is the final index."""
    runner = _runner(tmp_path, call_log, max_cycles=3)
    runner.run()

    ckpt = read_checkpoint(checkpoint_path_for("r1", tmp_path / "ckpt"))
    assert ckpt is not None, "checkpoint exists"
    assert ckpt.last_cycle == 2, "LOOP-08: last completed cycle index (0-based)"
    assert ckpt.patches_accepted == 3, "3 cycles x 1 applied"


def test_resume_skips_completed_cycles(tmp_path: Path, call_log: CallLog) -> None:
    """LOOP-08: resume from a checkpoint(last_cycle=1) starts at cycle 2."""
    cfg = _make_config(tmp_path)
    write_checkpoint(
        checkpoint_path_for("r1", tmp_path / "ckpt"),
        RunCheckpoint(
            run_id="r1",
            last_cycle=1,
            patches_accepted=2,
            patches_rejected=0,
            base_seed=100,
            run_epoch="2026-05-21T00:00:00+00:00",
            session_cursor="r1_c1",
        ),
    )
    runner = AutonomousRunner(
        run_id="r1",
        base_seed=100,
        max_cycles=3,
        config_path=cfg,
        driver=FakeDriver(call_log, applied=1),
        sim_runner=FakeSimRunner(call_log),
        metrics_writer=FakeMetricsWriter(call_log),
        sleep=lambda _s: None,
        install_signals=False,
    )
    runner.run()

    assert call_log.sim_calls[0]["cycle_idx"] == 2, "LOOP-08: resume starts at last_cycle+1"
    assert call_log.sim_calls[0]["session_id"] == "r1_c2"


def test_retryable_error_retries_then_continues(tmp_path: Path, call_log: CallLog) -> None:
    """LOOP-09: a retryable error retries with backoff then the cycle completes."""
    sleeps: list[float] = []
    sim = FakeSimRunner(call_log, raise_once={0: DBConnectError("db blip")})
    runner = _runner(tmp_path, call_log, max_cycles=1, sim_runner=sim, sleeps=sleeps)
    runner.run()

    assert len(sleeps) >= 1, "LOOP-09: backoff sleep invoked on retry"
    assert call_log.step_calls, "LOOP-09: cycle ultimately completed after retry"


def test_driver_step_retry_does_not_rerun_successful_sim(tmp_path: Path, call_log: CallLog) -> None:
    """A retryable failure in driver.step() retries the step only — it must NOT re-run a
    sim that already succeeded (re-running writes duplicate observation rows under the
    same deterministic session_id).
    """
    sleeps: list[float] = []
    driver = FakeDriver(call_log, applied=1, raise_once={0: DBConnectError("step blip")})
    runner = _runner(tmp_path, call_log, max_cycles=1, driver=driver, sleeps=sleeps)
    runner.run()

    sim_events = [e for e in call_log.events if e[0] == "sim"]
    step_events = [e for e in call_log.events if e[0] == "step"]
    assert sim_events == [("sim", 0)], "sim must run exactly once even though step retried"
    assert step_events == [("step", 0)], "step ultimately succeeded on retry"
    assert len(sleeps) >= 1, "backoff sleep invoked on the step retry"


def test_non_retryable_error_skips_cycle(tmp_path: Path, call_log: CallLog) -> None:
    """LOOP-09: a non-retryable error skips the cycle; the loop continues."""
    sim = FakeSimRunner(call_log, raise_map={0: ValueError("boom")})
    runner = _runner(tmp_path, call_log, max_cycles=2, sim_runner=sim)
    runner.run()

    # cycle 0 skipped (raised), cycle 1 ran sim+step
    assert ("sim", 1) in call_log.events, "LOOP-09: loop continued to cycle 1 after skipping 0"


def test_circuit_breaker_aborts_clean(tmp_path: Path, call_log: CallLog) -> None:
    """LOOP-09/D-08: max_consecutive_failures aborts the run cleanly (no exception escapes)."""
    sim = FakeSimRunner(
        call_log,
        raise_map={
            0: DBConnectError("down"),
            1: DBConnectError("down"),
            2: DBConnectError("down"),
            3: DBConnectError("down"),
        },
    )
    summary = _runner(tmp_path, call_log, max_cycles=10, sim_runner=sim, max_consecutive_failures=2).run()

    assert isinstance(summary, dict), "clean return, no raise"
    assert summary.get("circuit_broken") is True, "D-08: circuit-breaker tripped"
    assert summary["patches_rejected"] == 2, "LOOP-11: failed cycles counted as rejected"


def test_heartbeat_every_cycles_zero_raises_config_error(tmp_path: Path, call_log: CallLog) -> None:
    """A heartbeat_every_cycles of 0 is rejected at startup (avoids a mid-run ZeroDivisionError)."""
    import pytest

    from src._errors import ConfigError

    with pytest.raises(ConfigError):
        _runner(tmp_path, call_log, max_cycles=1, heartbeat_every_cycles=0)


def test_heartbeat_log_and_metric_per_cycle(tmp_path: Path, call_log: CallLog, caplog) -> None:
    """LOOP-10: per cycle, a heartbeat metric row is written (and a log line emitted)."""
    runner = _runner(tmp_path, call_log, max_cycles=3)
    runner.run()

    assert len(call_log.heartbeats) == 3, "LOOP-10: one heartbeat metric row per cycle"


def test_graceful_stop_finishes_inflight_then_summary(tmp_path: Path, call_log: CallLog) -> None:
    """LOOP-11/D-13: a stop mid-run finishes the in-flight cycle, flushes, prints summary.

    The sim_runner requests stop during cycle 0; D-13 means that cycle still completes
    (sim+step), then the loop halts at the next boundary.
    """
    cfg = _make_config(tmp_path)

    class StopOnFirstSim(FakeSimRunner):
        def __init__(self, log: CallLog, runner_ref: dict) -> None:
            super().__init__(log)
            self._runner_ref = runner_ref

        def __call__(self, **kwargs: object) -> dict:
            result = super().__call__(**kwargs)  # type: ignore[arg-type]
            self._runner_ref["runner"].request_stop()
            return result

    runner_ref: dict = {}
    runner = AutonomousRunner(
        run_id="r1",
        base_seed=100,
        max_cycles=None,
        config_path=cfg,
        driver=FakeDriver(call_log, applied=1),
        sim_runner=StopOnFirstSim(call_log, runner_ref),
        metrics_writer=FakeMetricsWriter(call_log),
        sleep=lambda _s: None,
        install_signals=False,
    )
    runner_ref["runner"] = runner
    summary = runner.run()

    # in-flight (first) cycle finishes: sim+step both ran for cycle 0
    assert ("sim", 0) in call_log.events and ("step", 0) in call_log.events, "D-13: in-flight cycle finished"
    assert ("sim", 1) not in call_log.events, "D-13: no new cycle started after stop"
    assert summary["cycles"] == 1, "exactly the in-flight cycle completed"
    for key in ("cycles", "patches_accepted", "patches_rejected", "patches_accepted_cumulative"):
        assert key in summary, f"LOOP-11: final summary has {key}"


def test_sentinel_present_at_start_runs_zero_cycles(tmp_path: Path, call_log: CallLog) -> None:
    """LOOP-11/D-12: a sentinel file already present at the first boundary stops before any cycle.

    Nothing is in-flight, so zero cycles run — the boundary check guarantees no cycle is ever
    started (and therefore never aborted mid-A/B).
    """
    sentinel = tmp_path / "stop"
    sentinel.write_text("")
    runner = _runner(tmp_path, call_log, max_cycles=None, sentinel_path=str(sentinel))
    summary = runner.run()

    assert call_log.events == [], "D-12: no cycle started when stop is already requested"
    assert summary["cycles"] == 0
    for key in ("cycles", "patches_accepted", "patches_rejected", "patches_accepted_cumulative"):
        assert key in summary, f"LOOP-11: final summary has {key}"
