"""Eval suite orchestrator — wires Tier-1 (LOO), Tier-2 (exploitability), match-play.

Aggregation rule (D-09-15):
    suite_pass = tier1_pass AND match_pass
    Tier-2 exploitability is REPORT-ONLY and NEVER affects suite_pass.

Persistence (D-09-17):
    tvd_loo + exploitability_pct rows written to metrics.
    Per-opponent MatchResults written via run_match(persist=True).
    JSON artifact written to data/eval_artifacts/suite_<iso>.json.

D-09-16 invariant: suite.py MUST NOT import patch_engine or autoloop.
    The suite returns an exit-code only; human/CI decides whether to gate.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec

from src._config import load_eval_config
from src._log import get_logger
from src.eval.baselines import REGISTRY
from src.eval.exploitability import aggregate_exploitability
from src.eval.loo_gate import run_loo_gate
from src.eval.result import MatchResult
from src.eval.run_match import run_match

log = get_logger("eval.suite")

_INSERT_METRIC_SQL = (
    "INSERT INTO metrics (metric_id, session_id, cluster_key, metric_name, value, ts) "
    "VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (session_id, metric_name, cluster_key, ts) DO NOTHING"
)


class EvalSuiteResult(msgspec.Struct, frozen=True, kw_only=True):
    """Full suite run result.

    suite_pass = tier1_pass AND match_pass (D-09-15).
    tier2_pass is always True (exploitability is report-only, D-09-7).
    """

    tier1_tvd: float
    tier1_top1: float
    tier1_threshold_floor: float
    tier1_regression_delta: float
    tier1_pass: bool
    tier2_exploitability: float
    tier2_pass: bool
    match_results: list[MatchResult]
    match_pass: bool
    suite_pass: bool
    # False when the tvd_loo/exploitability metric rows could not be written —
    # the regression baseline read by loo_gate may then be stale.
    metrics_persisted: bool = True


def run_suite(
    *,
    hands: int | None = None,
    seed: int = 42,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
    job: Any = None,
) -> EvalSuiteResult:
    """Run all three eval tiers and return an aggregated EvalSuiteResult.

    Args:
        hands: Override for match hands (both HU and 6-max); defaults from cfg.
        seed: RNG seed for determinism (SIM-02).
        _tsdb_conn: Injected psycopg connection for tests; None opens from env.
        _milvus: Injected Milvus client for tests; None opens from env.
        job: Optional JobRegistry Job for SSE progress events (best-effort).

    Returns:
        EvalSuiteResult with tier1_pass, tier2_pass (always True), match_pass,
        and suite_pass = tier1_pass AND match_pass.
    """
    log.info("eval.suite.started", seed=seed)
    run_start = datetime.now(UTC)
    suite_session_id = str(uuid.uuid4())

    try:
        cfg = load_eval_config()
        hu_hands = hands if hands is not None else cfg.match_hands_hu
        sixmax_hands = hands if hands is not None else cfg.match_hands_6max

        _maybe_push(job, {"event": "progress", "data": {"phase": "tier1", "pct": 0}})

        loo = run_loo_gate(
            _tsdb_conn=_tsdb_conn,
            _milvus=_milvus,
            _cfg=cfg,
        )
        tier1_pass = loo.passed
        log.info(
            "eval.suite.tier1_complete",
            tier1_tvd=loo.tier1_tvd,
            tier1_pass=tier1_pass,
        )

        _maybe_push(job, {"event": "progress", "data": {"phase": "tier2", "pct": 20}})

        expl = aggregate_exploitability(_tsdb_conn=_tsdb_conn)
        tier2_pass = True
        log.info(
            "eval.suite.tier2_complete",
            exploitability_mean=expl.exploitability_mean,
            report_only=True,
        )

        _maybe_push(job, {"event": "progress", "data": {"phase": "matches", "pct": 40}})

        match_results: list[MatchResult] = []
        for i, opponent_name in enumerate(REGISTRY):
            for table_size, match_hands in ((2, hu_hands), (6, sixmax_hands)):
                try:
                    mr = run_match(
                        opponent_name,
                        hands=match_hands,
                        seed=seed,
                        table_size=table_size,
                        persist=True,
                        _tsdb_conn=_tsdb_conn,
                        job=job,
                    )
                    match_results.append(mr)
                    log.info(
                        "eval.suite.match_complete",
                        opponent=opponent_name,
                        table_size=table_size,
                        status=mr.status,
                        ci_low=mr.ci_low,
                    )
                except NotImplementedError:
                    log.warning(
                        "eval.suite.match_skipped",
                        opponent=opponent_name,
                        table_size=table_size,
                        reason="NotImplementedError",
                    )

            pct = 40 + int(60 * (i + 1) / len(REGISTRY))
            _maybe_push(job, {"event": "progress", "data": {"phase": "matches", "pct": pct}})

        match_pass = all(mr.ci_low > 0 for mr in match_results) if match_results else True
        suite_pass = tier1_pass and match_pass

        log.info(
            "eval.suite.complete",
            suite_pass=suite_pass,
            tier1_pass=tier1_pass,
            match_pass=match_pass,
            n_matches=len(match_results),
        )

        metrics_persisted = bool(
            _persist_suite_metrics(
                suite_session_id=suite_session_id,
                run_start=run_start,
                tvd_loo=loo.tier1_tvd,
                exploitability_pct=expl.exploitability_mean,
                _tsdb_conn=_tsdb_conn,
            )
        )

        result = EvalSuiteResult(
            tier1_tvd=loo.tier1_tvd,
            tier1_top1=loo.tier1_top1,
            tier1_threshold_floor=cfg.tvd_floor,
            tier1_regression_delta=loo.regression_delta,
            tier1_pass=tier1_pass,
            tier2_exploitability=expl.exploitability_mean,
            tier2_pass=tier2_pass,
            match_results=match_results,
            match_pass=match_pass,
            suite_pass=suite_pass,
            metrics_persisted=metrics_persisted,
        )

        _write_artifact(result, run_start)

        return result

    except Exception as exc:
        log.error("eval.suite.failed", error=str(exc))
        raise


def _persist_suite_metrics(
    *,
    suite_session_id: str,
    run_start: datetime,
    tvd_loo: float,
    exploitability_pct: float,
    _tsdb_conn: Any = None,
) -> bool:
    """Write tvd_loo + exploitability_pct rows to the metrics hypertable.

    Returns True on success. A failure feeds the loo_gate regression baseline,
    so it is logged at error level and surfaced via the return (callers reflect
    it on EvalSuiteResult.metrics_persisted) rather than swallowed.
    """
    from src.db import timescale

    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                _INSERT_METRIC_SQL,
                (str(uuid.uuid4()), suite_session_id, None, "tvd_loo", tvd_loo, run_start),
            )
            cur.execute(
                _INSERT_METRIC_SQL,
                (
                    str(uuid.uuid4()),
                    suite_session_id,
                    None,
                    "exploitability_pct",
                    exploitability_pct,
                    run_start,
                ),
            )
        log.info("eval.suite.metrics_persisted", session_id=suite_session_id)
        return True
    except Exception as exc:
        log.error("eval.suite.metrics_persist_failed", error=str(exc))
        return False
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def _write_artifact(result: EvalSuiteResult, run_start: datetime) -> None:
    """Write full suite result as JSON to data/eval_artifacts/suite_<iso>.json."""
    try:
        artifact_dir = Path("data/eval_artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        iso = run_start.strftime("%Y%m%dT%H%M%SZ")
        path = artifact_dir / f"suite_{iso}.json"
        payload = msgspec.to_builtins(result)
        path.write_text(json.dumps(payload, indent=2, default=str))
        log.info("eval.suite.artifact_written", path=str(path))
    except Exception as exc:
        log.warning("eval.suite.artifact_write_failed", error=str(exc))


def _tsdb_dsn_from_env() -> str:
    import os

    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def _maybe_push(job: Any, event: dict) -> None:
    """Best-effort job event push; suppresses all failures."""
    if job is None:
        return
    with contextlib.suppress(Exception):
        job.events.put_nowait(event)
