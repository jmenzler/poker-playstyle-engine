"""AutonomousRunner: wraps AutoLoopDriver.step() into an unattended multi-cycle loop.
Each cycle runs a fresh sim then step(), with deterministic per-cycle seed/session-id,
bounded retry + circuit-breaker, per-cycle heartbeat + atomic checkpoint, graceful stop,
and a final summary. It never reimplements step() and adds no strategy_nodes write path.
"""

from __future__ import annotations

import json
import os
import signal
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src._config import AutoLoopConfig, load_toml_config
from src._errors import ConfigError, DBConnectError
from src._log import get_logger
from src.autoloop.checkpoint import (
    RunCheckpoint,
    checkpoint_path_for,
    read_checkpoint,
    write_checkpoint,
)

DEFAULT_CONFIG_PATH = Path("config/autoloop.toml")

_STREETS_ORDER = ("preflop", "flop", "turn", "river")


def _retryable_types() -> tuple[type[BaseException], ...]:
    """Transient exception types that warrant a bounded retry (D-06).

    psycopg + pymilvus are imported defensively so the predicate never crashes
    when an optional dep is absent (e.g. a pure-unit test environment).
    """
    types: list[type[BaseException]] = [DBConnectError, TimeoutError, ConnectionError]
    try:
        import psycopg

        types.append(psycopg.OperationalError)
    except Exception:
        pass
    try:
        from pymilvus.exceptions import MilvusException

        types.append(MilvusException)
    except Exception:
        pass
    return tuple(types)


class AutonomousRunner:
    """Drive repeated sim->step cycles unattended, crash-safe and gracefully stoppable."""

    def __init__(
        self,
        *,
        run_id: str,
        base_seed: int,
        max_cycles: int | None = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
        driver: Any = None,
        sim_runner: Callable[..., dict[str, Any]] | None = None,
        metrics_writer: Callable[..., None] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        install_signals: bool = True,
    ) -> None:
        self._run_id = run_id
        self._base_seed = base_seed
        self._max_cycles = max_cycles
        self._config_path = config_path
        self._config: AutoLoopConfig = load_toml_config(config_path, AutoLoopConfig)
        self._auto = self._config.autonomous
        if self._auto.heartbeat_every_cycles < 1:
            raise ConfigError("autonomous.heartbeat_every_cycles must be >= 1")

        self._driver = driver if driver is not None else self._build_driver()
        self._sim_runner = sim_runner if sim_runner is not None else self._default_sim_runner
        self._metrics_writer = (
            metrics_writer if metrics_writer is not None else self._default_heartbeat_writer
        )
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._sleep = sleep if sleep is not None else time.sleep
        self._retryable = _retryable_types()

        self._log = get_logger("autoloop.orchestrator")
        self._stop_requested = False
        if install_signals:
            self._install_signal_handlers()

    # -- public control ---------------------------------------------------

    def request_stop(self) -> None:
        """Request a graceful stop (also set by SIGTERM/SIGINT)."""
        self._stop_requested = True

    def run(self) -> dict[str, Any]:
        """Run the loop until max_cycles, graceful stop, or circuit-breaker. Returns the summary."""
        ckpt_path = checkpoint_path_for(self._run_id, self._auto.checkpoint_path)
        existing = read_checkpoint(ckpt_path)
        if existing is not None:
            start_cycle = existing.last_cycle + 1
            run_epoch = datetime.fromisoformat(existing.run_epoch)
            patches_accepted = existing.patches_accepted
            patches_rejected = existing.patches_rejected
            cycles_completed = existing.last_cycle + 1
        else:
            start_cycle = 0
            run_epoch = self._clock()
            patches_accepted = 0
            patches_rejected = 0
            cycles_completed = 0

        consecutive_failures = 0
        circuit_broken = False
        cycle_idx = start_cycle

        while True:
            if self._max_cycles is not None and cycle_idx >= start_cycle + self._max_cycles:
                break
            if self._should_stop():
                self._log.info("autoloop.orchestrator.stop_requested", run_id=self._run_id, cycle=cycle_idx)
                break

            applied, failed = self._run_cycle_with_retry(cycle_idx, run_epoch)

            if failed:
                patches_rejected += 1
                consecutive_failures += 1
                if consecutive_failures >= self._auto.max_consecutive_failures:
                    circuit_broken = True
                    self._log.error(
                        "autoloop.orchestrator.circuit_broken",
                        run_id=self._run_id,
                        cycle=cycle_idx,
                        consecutive_failures=consecutive_failures,
                    )
                    break
                cycle_idx += 1
                continue

            consecutive_failures = 0
            patches_accepted += applied
            cycles_completed = cycle_idx + 1

            self._flush_checkpoint(ckpt_path, run_epoch, cycle_idx, patches_accepted, patches_rejected)
            self._emit_heartbeat(cycle_idx, cycles_completed, applied, patches_accepted)
            cycle_idx += 1

        summary = {
            "run_id": self._run_id,
            "cycles": cycles_completed - start_cycle,
            "cycles_completed": cycles_completed,
            "patches_accepted": patches_accepted,
            "patches_rejected": patches_rejected,
            "patches_accepted_cumulative": self._patches_accepted_cumulative(patches_accepted),
            "circuit_broken": circuit_broken,
        }
        print(json.dumps(summary))
        self._log.info("autoloop.orchestrator.summary", **summary)
        return summary

    # -- cycle body -------------------------------------------------------

    def _run_cycle_with_retry(self, cycle_idx: int, run_epoch: datetime) -> tuple[int, bool]:
        """Run one cycle with bounded retry. Returns (patches_applied, failed)."""
        cycle_seed = self._base_seed + cycle_idx
        session_id = f"{self._run_id}_c{cycle_idx}"
        # Deterministic ts so a re-run cycle re-pins the identical session_started_at,
        # making flush_session_metrics a no-op (idempotent resume).
        session_started_at = run_epoch + timedelta(seconds=cycle_idx)

        # Re-running a succeeded sim under the same deterministic session_id would
        # write duplicate observation rows (plain INSERT, uuid4 PK, no ON CONFLICT),
        # so on a step-only retry the sim is skipped, not replayed.
        sim_done = False
        for attempt in range(self._auto.max_retries + 1):
            try:
                if not sim_done:
                    self._sim_runner(
                        run_id=self._run_id,
                        cycle_idx=cycle_idx,
                        cycle_seed=cycle_seed,
                        session_id=session_id,
                        session_started_at=session_started_at,
                        n_hands=self._auto.sim_n_hands,
                    )
                    sim_done = True
                applied = self._driver.step()
                return applied, False
            except Exception as exc:
                self._log.error(
                    "autoloop.orchestrator.cycle_failed",
                    cycle_id=session_id,
                    cycle=cycle_idx,
                    attempt=attempt,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                if not isinstance(exc, self._retryable):
                    return 0, True
                if attempt >= self._auto.max_retries:
                    return 0, True
                self._sleep(self._auto.backoff_base_s * (2**attempt))
        return 0, True

    # -- stop handling ----------------------------------------------------

    def _should_stop(self) -> bool:
        if self._stop_requested:
            return True
        return Path(self._auto.sentinel_path).exists()

    def _install_signal_handlers(self) -> None:
        def _handler(_signum: int, _frame: Any) -> None:
            self._stop_requested = True  # async-signal-safe: flag only, no I/O

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except ValueError:
            # signal.signal only works on the main thread; ignore off-main-thread (e.g. tests)
            pass

    # -- side effects -----------------------------------------------------

    def _flush_checkpoint(
        self,
        ckpt_path: Path,
        run_epoch: datetime,
        cycle_idx: int,
        patches_accepted: int,
        patches_rejected: int,
    ) -> None:
        write_checkpoint(
            ckpt_path,
            RunCheckpoint(
                run_id=self._run_id,
                last_cycle=cycle_idx,
                patches_accepted=patches_accepted,
                patches_rejected=patches_rejected,
                base_seed=self._base_seed,
                run_epoch=run_epoch.isoformat(),
                session_cursor=f"{self._run_id}_c{cycle_idx}",
            ),
        )

    def _emit_heartbeat(
        self, cycle_idx: int, cycles_completed: int, applied: int, patches_accepted: int
    ) -> None:
        if cycle_idx % self._auto.heartbeat_every_cycles != 0:
            return
        self._log.info(
            "autoloop.orchestrator.heartbeat",
            run_id=self._run_id,
            cycle=cycle_idx,
            cycles_completed=cycles_completed,
            patches_applied=applied,
            patches_accepted=patches_accepted,
            patches_accepted_cumulative=self._patches_accepted_cumulative(patches_accepted),
        )
        self._metrics_writer(run_id=self._run_id, value=float(cycles_completed), ts=self._clock())

    def _patches_accepted_cumulative(self, patches_accepted: int) -> float:
        """Cumulative accepted-patch count for the heartbeat/summary.

        Each accepted patch passed the ci_low > 0 acceptance gate, so a rising
        count is a coarse positive-ev_loss-reducing-progress proxy — it is NOT a
        measured ev_loss delta.
        """
        return float(patches_accepted)

    # -- production defaults (overridden in tests via injection) ----------

    def _build_driver(self) -> Any:
        from src.autoloop.driver import AutoLoopDriver

        return AutoLoopDriver(self._config_path)

    def _default_sim_runner(
        self,
        *,
        run_id: str,
        cycle_idx: int,
        cycle_seed: int,
        session_id: str,
        session_started_at: datetime,
        n_hands: int,
    ) -> dict[str, Any]:
        """Run one fresh RECORD sim and flush its metrics (reuses run_sim_session wiring)."""
        from src.decision_engine.engine import engine_from_env
        from src.metrics.sink import flush_session_metrics
        from src.sim import ObservationWriter, SimAdapter, run_record_session

        dsn = _tsdb_dsn_from_env()
        engine = engine_from_env(rng_seed=cycle_seed)
        adapter = SimAdapter()
        with ObservationWriter(dsn, session_id) as writer:
            summary = run_record_session(
                adapter,
                engine,
                writer,
                session_seed=cycle_seed,
                n_hands=n_hands,
                session_id=session_id,
            )
        flush_session_metrics(
            summary["session_id"],
            summary.get("latency_buffer", []),
            session_started_at=session_started_at,
        )
        return summary

    def _default_heartbeat_writer(self, *, run_id: str, value: float, ts: datetime) -> None:
        """Insert one autoloop_heartbeat row into the metrics hypertable (no migration)."""
        from src.db import timescale

        conn = timescale.connect(_tsdb_dsn_from_env())
        try:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO metrics (metric_id, session_id, cluster_key, metric_name, value, ts) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (session_id, metric_name, cluster_key, ts) DO NOTHING",
                    (str(uuid.uuid4()), run_id, None, "autoloop_heartbeat", value, ts),
                )
        finally:
            conn.close()


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables (TSDB_PASSWORD required)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
