"""Phase 5 Plan 08: AutoLoopDriver smoke test on live PC state.

Integration tests for the auto-loop driver against a real (sparse) Milvus + TSDB setup.

Tests:
  - test_autoloop_step_no_crash_on_real_pc_state: driver.step() completes without
    crashing; n_applied >= 0 (likely 0 since no fresh metrics from a recent session).
  - test_autoloop_step_disabled_does_no_db_io: LOOP-05 runtime enforcement at
    the integration layer — enabled=False prevents any DB I/O.

Run on PC: pytest -m integration tests/integration/test_autoloop_e2e_integration.py -v
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_autoloop_step_no_crash_on_real_pc_state(
    tsdb_dsn: str,
    milvus_uri: str,
    milvus_token: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AutoLoopDriver.step() completes on real PC state without crashing.

    The goal is a smoke test: verify the driver can connect, query metrics, and
    exit cleanly. On a fresh PC with no metrics rows from a recent RECORD session,
    the expected result is n_applied == 0 (no leaks detected). The important thing
    is no crash and no exception.

    LOOP-05 verification (enabled=true path): the "autoloop.driver.step.disabled"
    log line must NOT appear (driver is enabled).

    WR-03 fix: use monkeypatch for env-var injection (handles cleanup atomically,
    correctly restores empty-string values) and parse DSN with key=value splitting
    that handles passwords containing spaces.
    """
    import structlog.testing

    from src._config import AutoLoopConfig
    from src.autoloop.driver import AutoLoopDriver

    # Parse tsdb_dsn using key=value token split (handles values containing spaces
    # correctly via split("=", 1); whitespace-split of the DSN string itself is fine
    # because the libpq keyword=value format uses spaces as token separators and
    # passwords with spaces must be quoted at the libpq level, not embedded raw).
    dsn_parts = dict(part.split("=", 1) for part in tsdb_dsn.split() if "=" in part)
    monkeypatch.setenv("TSDB_HOST", dsn_parts.get("host", "127.0.0.1"))
    monkeypatch.setenv("TSDB_PORT", dsn_parts.get("port", "55432"))
    monkeypatch.setenv("TSDB_DB", dsn_parts.get("dbname", "poker_engine"))
    monkeypatch.setenv("TSDB_USER", dsn_parts.get("user", "poker"))
    monkeypatch.setenv("TSDB_PASSWORD", dsn_parts.get("password", ""))

    host_port = milvus_uri.replace("http://", "").split(":")
    monkeypatch.setenv("MILVUS_HOST", host_port[0])
    if len(host_port) > 1:
        monkeypatch.setenv("MILVUS_PORT", host_port[1])

    # Use a config that mirrors production but with safer limits for smoke testing
    from unittest.mock import patch as mock_patch

    fast_config = AutoLoopConfig(
        enabled=True,
        tau_leak=0.3,
        min_observations=20,
        max_patches_per_run=1,
        n_hands_validation=10,
        bootstrap_resamples=10,
        ci_alpha=0.05,
    )

    with mock_patch("src.autoloop.driver.load_toml_config", return_value=fast_config):
        driver = AutoLoopDriver(validation_seed=42)

        with structlog.testing.capture_logs() as captured_logs:
            n_applied = driver.step()

    # Smoke assertion: no crash, return value is non-negative
    assert n_applied >= 0, f"driver.step() must return >= 0; got {n_applied}"

    # LOOP-05 check (enabled=true path): disabled log line must NOT appear
    disabled_events = [log for log in captured_logs if log.get("event") == "autoloop.driver.step.disabled"]
    assert len(disabled_events) == 0, (
        "LOOP-05: enabled=true driver must NOT log 'autoloop.driver.step.disabled'"
    )


def test_autoloop_step_disabled_does_no_db_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOOP-05 runtime enforcement: enabled=False prevents all DB I/O.

    Monkeypatches:
    - load_toml_config → returns AutoLoopConfig(enabled=False)
    - timescale.connect → raises RuntimeError (any call is a LOOP-05 violation)
    - milvus_db.connect → raises RuntimeError

    Verifies that driver.step() returns 0 without touching either DB.
    This is the integration-layer LOOP-05 enforcement test (complementary to the
    unit-level test in tests/phase5/test_autoloop_step.py).

    Note: This test does NOT require the external test services — it runs on Mac via monkeypatching.
    It is in the integration suite because it verifies the LOOP-05 contract at the
    system-interface boundary (config load → DB connect decision).
    """
    import structlog.testing

    from src._config import AutoLoopConfig
    from src.autoloop.driver import AutoLoopDriver

    disabled_config = AutoLoopConfig(
        enabled=False,
        tau_leak=0.3,
        min_observations=20,
        max_patches_per_run=5,
        n_hands_validation=5000,
        bootstrap_resamples=1000,
        ci_alpha=0.05,
    )

    monkeypatch.setattr(
        "src.autoloop.driver.load_toml_config",
        lambda path, schema: disabled_config,
    )

    # Any DB connection attempt is a LOOP-05 violation
    def _fail_connect(*args, **kwargs):
        raise RuntimeError("LOOP-05 violation: DB connect called despite enabled=False")

    monkeypatch.setattr("src.autoloop.driver.timescale.connect", _fail_connect)
    monkeypatch.setattr("src.autoloop.driver.milvus_db.connect", _fail_connect)
    monkeypatch.setattr("src.autoloop.driver._tsdb_dsn_from_env", lambda: "ignored")
    monkeypatch.setattr("src.autoloop.driver._milvus_uri_from_env", lambda: "ignored")

    with structlog.testing.capture_logs() as captured_logs:
        driver = AutoLoopDriver(validation_seed=0)
        n_applied = driver.step()

    # Must return 0 (no patches applied on disabled path)
    assert n_applied == 0, f"LOOP-05: disabled driver must return 0; got {n_applied}"

    # The disabled log line MUST appear
    disabled_events = [log for log in captured_logs if log.get("event") == "autoloop.driver.step.disabled"]
    assert len(disabled_events) == 1, (
        f"LOOP-05: expected exactly 1 'autoloop.driver.step.disabled' log; got {len(disabled_events)}"
    )
