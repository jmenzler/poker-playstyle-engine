"""Stage B solver verification (D-05, D-07, OQ-2 RESOLVED).

Given a cluster_key:
1. Resolve a SolverSpot (range_ip, range_oop, board, pot, prev_bet) from a
   representative observation row.
2. Call PostflopCliBackend.solve(spot) — Phase 6 Plan 02 wires the binary I/O.
3. Cache the result as a strategy_nodes row with source='solver' via
   PatchEngine.apply (Option A — sentinel ValidationResult). The solver IS the
   ground truth being cached, so A/B validation is bypassed.
4. Push progress events to ``job.events`` queue (Plan 07 SSE consumes these).

NO CONCURRENCY CONTROL HERE. Per RESEARCH OQ-2 (RESOLVED): the sequential
Stage-B serialization primitive lives in Plan 07's JobRegistry. This module
is a pure wrapper; the registry queues calls and limits parallelism.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src._errors import NoStrategyError, SolverParseError
from src._log import get_logger
from src.db import timescale
from src.patch_engine import PatchEngine, PatchSpec
from src.solver.postflop_cli import PostflopCliBackend, SolverSpot
from src.solver.range_resolver import _resolve_ranges, build_range_lookup
from src.study.edit_node import _sentinel_validation

log = get_logger("study.solver_verify")

PALETTE_DIR = Path("research/preflop-ranges/outputs")
_range_lookup: dict | None = None


def _get_range_lookup() -> dict:
    global _range_lookup
    if _range_lookup is None:
        _range_lookup = build_range_lookup(PALETTE_DIR) if PALETTE_DIR.exists() else {}
    return _range_lookup


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables (mirrors patch_engine)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def _default_solver() -> PostflopCliBackend:
    """Build a default PostflopCliBackend from POSTFLOP_CLI_BIN env var.

    Pattern mirrors tests/integration/test_postflop_cli_integration.py.
    """
    binary = Path(
        os.environ.get("POSTFLOP_CLI_BIN", "~/postflop-cli/target/release/postflop-cli")
    ).expanduser()
    return PostflopCliBackend(binary_path=binary)


def verify_cluster(
    cluster_key: str,
    *,
    force: bool = False,
    job: Any = None,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
    _solver: Any = None,
    _patch_engine: Any = None,
) -> dict:
    """Run Stage B for a single cluster. Returns dict with patch_id + cache status.

    Pure wrapper — concurrency control is the caller's responsibility (JobRegistry).

    Args:
        cluster_key: target cluster's canonical key string.
        force: when True, skip the cache short-circuit and always invoke the
            solver. Default False (cache hit returns the existing source='solver'
            row without re-solving).
        job: optional Plan-07 Job; when provided, push progress/done/error events
            to its asyncio.Queue (event_push is non-blocking and never raises).
        _tsdb_conn / _milvus / _solver / _patch_engine: test-injection hooks.

    Returns:
        JSON-serializable dict:
        - cache hit: {cached: True, cluster_key, node_id, source: 'solver'}
        - cache miss: {cached: False, cluster_key, patch_id, new_node_id,
                       source: 'solver', exploitability_pct, solve_time_ms}

    Raises:
        SolverParseError: from PostflopCliBackend.solve(), or when no
            observations row exists to derive a SolverSpot.
        NoStrategyError: when no active strategy_nodes row exists for the
            cluster_key after a solver run.
    """
    log.info("study.solver_verify.started", cluster_key=cluster_key, force=force)
    _maybe_push(job, "progress", {"stage": "init", "cluster_key": cluster_key})

    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        # 1. Cache short-circuit (unless force=True)
        if not force:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT node_id FROM strategy_nodes "
                    "WHERE cluster_key = %s AND source = 'solver' AND active = TRUE LIMIT 1",
                    (cluster_key,),
                )
                cached = cur.fetchone()
            if cached:
                log.info("study.solver_verify.cache_hit", cluster_key=cluster_key)
                _maybe_push(job, "done", {"cached": True, "cluster_key": cluster_key})
                return {
                    "cached": True,
                    "cluster_key": cluster_key,
                    "node_id": str(cached[0]),
                    "source": "solver",
                }

        # 2. Resolve SolverSpot from a representative observation
        _maybe_push(job, "progress", {"stage": "resolve_spot"})
        spot = _resolve_solver_spot(conn, cluster_key)

        # 3. Solver invocation
        _maybe_push(job, "progress", {"stage": "solving"})
        solver = _solver if _solver is not None else _default_solver()
        try:
            result = solver.solve(spot)
        except (SolverParseError, NoStrategyError) as exc:
            log.error(
                "study.solver_verify.solver_failed",
                cluster_key=cluster_key,
                error=str(exc),
            )
            _maybe_push(job, "error", {"error": str(exc), "cluster_key": cluster_key})
            raise

        # 4. Resolve the embedding: prefer the active node's, else bootstrap from a
        # representative observation (first-ever promotion on a freshly-rebuilt corpus).
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM strategy_nodes WHERE cluster_key = %s AND active = TRUE LIMIT 1",
                (cluster_key,),
            )
            row = cur.fetchone()
        if row is not None:
            embedding = row[0]
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT embedding FROM observations "
                    "WHERE cluster_key = %s AND embedding IS NOT NULL LIMIT 1",
                    (cluster_key,),
                )
                obs_row = cur.fetchone()
            if obs_row is None:
                raise NoStrategyError(
                    f"no active strategy_node and no observation embedding to bootstrap "
                    f"cluster_key={cluster_key!r}"
                )
            embedding = obs_row[0]

        # Tag the patch with a representative DP from observations (replayable
        # identity); the coexist model has no prior node to supersede.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT decision_id FROM observations "
                "WHERE cluster_key = %s AND decision_id IS NOT NULL LIMIT 1",
                (cluster_key,),
            )
            dp_row = cur.fetchone()
        if dp_row is None:
            raise NoStrategyError(
                f"no observation decision_id to tag solver patch for cluster_key={cluster_key!r}"
            )

        spec = PatchSpec(
            cluster_key=cluster_key,
            decision_id=dp_row[0],
            action_dist=result.action_dist,
            embedding=list(embedding) if not isinstance(embedding, list) else embedding,
            gto_score=max(0.0, 1.0 - (result.exploitability_pct / 100.0)),
            confidence=1.0,
            prev_node_id=None,
            source="solver",
        )
        engine = _patch_engine if _patch_engine is not None else PatchEngine()
        record = engine.apply(
            spec,
            validation=_sentinel_validation(),
            pre_ev_loss=None,
            post_ev_loss=None,
            _tsdb_conn=conn,
            _milvus=_milvus,
        )
        out = {
            "cached": False,
            "cluster_key": cluster_key,
            "patch_id": str(record.patch_id),
            "new_node_id": str(record.new_node_id),
            "source": "solver",
            "exploitability_pct": float(result.exploitability_pct),
            "solve_time_ms": int(result.solve_time_ms),
        }
        log.info(
            "study.solver_verify.complete",
            cluster_key=cluster_key,
            patch_id=out["patch_id"],
            solve_time_ms=result.solve_time_ms,
        )
        _maybe_push(job, "done", out)
        return out
    finally:
        if own_conn:
            conn.close()


def _maybe_push(job: Any, event: str, data: dict) -> None:
    """Push an event to job.events queue if job is provided. Non-blocking.

    Failures (queue full, queue missing, etc.) are swallowed and logged so a
    flaky event sink never breaks the solver path.
    """
    if job is None:
        return
    try:
        job.events.put_nowait({"event": event, "data": data})
    except Exception:
        log.warning("study.solver_verify.event_push_failed", event=event)


def _resolve_solver_spot(conn: Any, cluster_key: str) -> SolverSpot:
    """Build a SolverSpot from a representative observation for this cluster.

    Reads observations.felt_snapshot JSONB. Expected keys: board, pot,
    effective_stack, prev_bet, range_ip, range_oop. Missing fields fall back
    to project defaults (v1 placeholder; v2 fills default-range table from
    preflop solver charts).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT felt_snapshot FROM observations WHERE cluster_key = %s LIMIT 1",
            (cluster_key,),
        )
        row = cur.fetchone()
    if row is None:
        raise SolverParseError(
            "no solved data for this spot — on-demand spot solving is not available in this build "
            f"(cluster_key={cluster_key!r} has no backing observation)"
        )
    spot_features = row[0] or {}
    hero_pos = spot_features.get("hero_pos", "BTN")
    villain_pos = spot_features.get("villain_pos", "BB")
    pot_type = spot_features.get("pot_type", "srp")
    range_ip, range_oop = _resolve_ranges(
        obs_spot_features=spot_features,
        hero_pos=hero_pos,
        villain_pos=villain_pos,
        pot_type=pot_type,
        palette_lookup=_get_range_lookup(),
    )
    return SolverSpot(
        pot=int(spot_features.get("pot", 100)),
        effective_stack=int(spot_features.get("effective_stack", 1000)),
        board=list(spot_features.get("board", [])),
        prev_bet=(int(spot_features["prev_bet"]) if spot_features.get("prev_bet") is not None else None),
        range_ip=range_ip,
        range_oop=range_oop,
    )
