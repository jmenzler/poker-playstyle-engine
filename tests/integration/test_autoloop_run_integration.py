"""DB-backed e2e for AutonomousRunner: multi-cycle run, heartbeat rows, resume, summary.

Runs only under `-m integration` against a real TimescaleDB + Milvus stack:
    pytest -m integration tests/integration/test_autoloop_run_integration.py -v
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from src.autoloop.checkpoint import checkpoint_path_for, read_checkpoint
from src.autoloop.orchestrator import AutonomousRunner
from src.db import timescale

pytestmark = pytest.mark.integration


def _inject_env(
    monkeypatch: pytest.MonkeyPatch, tsdb_dsn: str, milvus_uri: str, milvus_token: str | None
) -> None:
    """Mirror the env-injection pattern from test_autoloop_e2e_integration.py."""
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
    if milvus_token:
        monkeypatch.setenv("MILVUS_TOKEN", milvus_token)


def _write_fast_config(tmp_path: Path, *, sim_n_hands: int = 300) -> Path:
    """Write a tmp autoloop.toml with small sim + tmp checkpoint/sentinel dirs."""
    toml = tmp_path / "autoloop.toml"
    toml.write_text(
        "enabled = true\n"
        "tau_leak = 0.3\n"
        "min_observations = 20\n"
        "max_patches_per_run = 1\n"
        "n_hands_validation = 100\n"
        "bootstrap_resamples = 50\n"
        "ci_alpha = 0.05\n"
        '[algo]\nkind = "knn_rebalance"\n'
        "[autonomous]\n"
        "max_retries = 1\n"
        "backoff_base_s = 0.5\n"
        "max_consecutive_failures = 3\n"
        "heartbeat_every_cycles = 1\n"
        f'checkpoint_path = "{tmp_path / "ckpt"}"\n'
        f'sentinel_path = "{tmp_path / "stop"}"\n'
        f"sim_n_hands = {sim_n_hands}\n"
    )
    return toml


def _count_heartbeats(tsdb_dsn: str, run_id: str) -> int:
    conn = timescale.connect(tsdb_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM metrics WHERE metric_name = 'autoloop_heartbeat' AND session_id = %s",
                (run_id,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        conn.close()


def test_autonomous_run_multi_cycle(
    tmp_path: Path,
    tsdb_dsn: str,
    milvus_uri: str,
    milvus_token: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOOP-07/10/11: a 2-cycle run completes, writes one heartbeat row per cycle, summarizes."""
    _inject_env(monkeypatch, tsdb_dsn, milvus_uri, milvus_token)
    cfg = _write_fast_config(tmp_path)
    run_id = f"itest-{uuid.uuid4().hex[:8]}"

    runner = AutonomousRunner(
        run_id=run_id, base_seed=7, max_cycles=2, config_path=cfg, install_signals=False
    )
    summary = runner.run()

    assert summary["cycles"] == 2, "LOOP-07: two full cycles ran"
    for key in ("cycles", "patches_accepted", "patches_rejected", "patches_accepted_cumulative"):
        assert key in summary, f"LOOP-11: final summary has {key}"

    hb = _count_heartbeats(tsdb_dsn, run_id)
    assert hb == 2, f"LOOP-10: one autoloop_heartbeat row per cycle, got {hb}"

    ckpt = read_checkpoint(checkpoint_path_for(run_id, tmp_path / "ckpt"))
    assert ckpt is not None and ckpt.last_cycle == 1, "LOOP-08: checkpoint at final cycle (0-based)"


def test_autonomous_resume_skips_completed(
    tmp_path: Path,
    tsdb_dsn: str,
    milvus_uri: str,
    milvus_token: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOOP-08: resume continues from the checkpoint without re-running completed cycles."""
    _inject_env(monkeypatch, tsdb_dsn, milvus_uri, milvus_token)
    cfg = _write_fast_config(tmp_path)
    run_id = f"itest-{uuid.uuid4().hex[:8]}"

    # First run: one cycle (writes checkpoint last_cycle=0).
    AutonomousRunner(run_id=run_id, base_seed=7, max_cycles=1, config_path=cfg, install_signals=False).run()
    ckpt1 = read_checkpoint(checkpoint_path_for(run_id, tmp_path / "ckpt"))
    assert ckpt1 is not None and ckpt1.last_cycle == 0

    # Resume: another cycle continues at cycle 1 (not re-running cycle 0).
    summary = AutonomousRunner(
        run_id=run_id, base_seed=7, max_cycles=1, config_path=cfg, install_signals=False
    ).run()
    ckpt2 = read_checkpoint(checkpoint_path_for(run_id, tmp_path / "ckpt"))

    assert ckpt2 is not None and ckpt2.last_cycle == 1, "LOOP-08: resume advanced to cycle 1"
    assert summary["cycles_completed"] == 2, "two cycles completed across the two runs"
