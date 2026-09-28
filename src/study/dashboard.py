"""CLI-06 / FastAPI ``/api/dashboard`` — multi-section snapshot.

Per D-NEW-27 (Dashboard verdict-led restructure):
    * ``loop_health``    — verdict bar (``improving``/``stalled``/``mixed``)
                           + one-sentence narrative.
    * ``ev_loss_trend``  — last 30 sessions, mean ev_loss per session
                           (for the hero chart).
    * ``recent_activity``— union of patches + match events, top 10 by ts DESC.
    * ``health``         — per-subsystem state (db/milvus/solver/fastapi).
    * ``session``        — in-flight session stats (n_obs, start_ts, last_ts).
    * ``kb_growth``      — KB node counts + last-7d deltas by source.

JSON shape is the single source of truth — the CLI text renderer parses this
dict.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from src._log import get_logger
from src.db import timescale

log = get_logger("study.dashboard")


def dashboard_snapshot(*, _tsdb_conn: Any = None) -> dict:
    """Compose all six dashboard sections from independent SQL queries.

    Each section is best-effort: a failure on one section MUST NOT take down
    the whole panel.  Today the failures bubble up (so test mocks catch them
    explicitly); production runs behind a 30s cache so partial failures
    surface clearly without page-level outages.
    """
    log.info("study.dashboard.started")
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        out = {
            "loop_health": _loop_health(conn),
            "ev_loss_trend": _ev_loss_trend(conn, n_sessions=30),
            "recent_activity": _recent_activity(conn, limit=10),
            "health": _health(conn),
            "session": _session_in_flight(conn),
            "kb_growth": _kb_growth(conn),
            "exploitability_trend": _exploitability_trend(conn, n_results=30),
            "tvd_loo_trend": _tvd_loo_trend(conn, n_results=30),
        }
        log.info("study.dashboard.complete", sections=list(out.keys()))
        return out
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def _ev_loss_trend(conn: Any, n_sessions: int = 30) -> list[dict]:
    """Mean ev_loss per session, last N sessions (newest first -> reverse for chart)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT session_id, MAX(ts) AS ts, AVG(value) AS ev_loss "
            "FROM metrics WHERE metric_name = 'ev_loss' "
            "GROUP BY session_id ORDER BY MAX(ts) DESC LIMIT %s",
            (n_sessions,),
        )
        rows = cur.fetchall()
    return [
        {
            "session_id": str(s),
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "ev_loss": float(ev),
        }
        for s, ts, ev in reversed(rows)
    ]


def _exploitability_trend(conn: Any, n_results: int = 30) -> list[dict]:
    """Last N exploitability_pct values from metrics, oldest-first for charting."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT session_id, MAX(ts) AS ts, AVG(value) AS exploitability "
            "FROM metrics WHERE metric_name = 'exploitability_pct' "
            "GROUP BY session_id ORDER BY MAX(ts) DESC LIMIT %s",
            (n_results,),
        )
        rows = cur.fetchall()
    return [
        {
            "session_id": str(s),
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "exploitability": float(ev),
        }
        for s, ts, ev in reversed(rows)
    ]


def _tvd_loo_trend(conn: Any, n_results: int = 30) -> list[dict]:
    """Last N tvd_loo values from metrics, oldest-first for charting."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT session_id, MAX(ts) AS ts, AVG(value) AS tvd_loo "
            "FROM metrics WHERE metric_name = 'tvd_loo' "
            "GROUP BY session_id ORDER BY MAX(ts) DESC LIMIT %s",
            (n_results,),
        )
        rows = cur.fetchall()
    return [
        {
            "session_id": str(s),
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "tvd_loo": float(v),
        }
        for s, ts, v in reversed(rows)
    ]


def _loop_health(conn: Any) -> dict:
    """Verdict from last-3-session ev_loss mean vs 7-day-prior mean.

    Buckets:
        * ``improving``: recent_mean < prior_mean * 0.95
        * ``stalled``:   recent_mean > prior_mean * 1.05
        * ``mixed``:     within ±5%, or insufficient history
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT AVG(value) FROM metrics WHERE metric_name = 'ev_loss' "
            "AND session_id IN (SELECT session_id FROM metrics "
            "                   WHERE metric_name='ev_loss' "
            "                   GROUP BY session_id "
            "                   ORDER BY MAX(ts) DESC LIMIT 3)"
        )
        recent_mean = cur.fetchone()[0]
        cur.execute(
            "SELECT AVG(value) FROM metrics WHERE metric_name = 'ev_loss' AND ts < now() - INTERVAL '7 days'"
        )
        prior_mean = cur.fetchone()[0]

    if recent_mean is None or prior_mean is None:
        verdict = "mixed"
        narrative = "insufficient session history for trend verdict"
    else:
        recent_f = float(recent_mean)
        prior_f = float(prior_mean)
        if recent_f < prior_f * 0.95:
            verdict = "improving"
            narrative = (
                f"ev_loss trending down · {(prior_f - recent_f) / prior_f * 100:.1f}% lower "
                f"than 7d-prior mean · no intervention needed"
            )
        elif recent_f > prior_f * 1.05:
            verdict = "stalled"
            narrative = (
                f"ev_loss trending up · {(recent_f - prior_f) / prior_f * 100:.1f}% higher "
                f"than 7d-prior mean · investigate Strategy Leaks"
            )
        else:
            verdict = "mixed"
            narrative = "ev_loss flat across recent sessions"

    return {
        "verdict": verdict,
        "narrative": narrative,
        "recent_mean": float(recent_mean) if recent_mean is not None else None,
        "prior_mean": float(prior_mean) if prior_mean is not None else None,
    }


def _recent_activity(conn: Any, limit: int = 10) -> list[dict]:
    """Union of recent patches + match completions, ordered by ts DESC."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, 'patch' AS event_type, "
            "       CONCAT('patch ', SUBSTRING(patch_id::text, 1, 8), ' on ', "
            "              SUBSTRING(cluster_key, 1, 30), ' source=', source) AS summary "
            "FROM patches "
            "UNION ALL "
            "SELECT started_at AS ts, 'eval_match' AS event_type, "
            "       CONCAT('match vs ', opponent, ' status=', status) AS summary "
            "FROM matches WHERE finished_at IS NOT NULL "
            "ORDER BY ts DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "event_type": ev,
            "summary": s,
        }
        for ts, ev, s in rows
    ]


def _health(conn: Any) -> dict:
    """Subsystem liveness.  ``db`` is checkable here; others reported ``unknown``.

    Milvus / Solver / FastAPI checks are best-effort and live in their own
    modules — the FastAPI router can call them and merge into this dict
    if/when desired.  Reporting ``unknown`` here is the safe default — it
    forces the UI to render a neutral indicator instead of falsely
    "healthy".
    """
    health = {"db": "unknown", "milvus": "unknown", "solver": "unknown", "fastapi": "unknown"}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        health["db"] = "healthy"
    except Exception:
        health["db"] = "down"
    return health


def _session_in_flight(conn: Any) -> dict | None:
    """Most-recent session in observations -> in-flight stats; ``None`` when DB empty."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT session_id, COUNT(*) AS n_obs, MIN(ts) AS start_ts, MAX(ts) AS last_ts "
            "FROM observations GROUP BY session_id ORDER BY MAX(ts) DESC LIMIT 1"
        )
        row = cur.fetchone()
    if row is None:
        return None
    session_id, n_obs, start_ts, last_ts = row
    return {
        "session_id": str(session_id),
        "n_obs": int(n_obs),
        "start_ts": start_ts.isoformat() if hasattr(start_ts, "isoformat") else str(start_ts),
        "last_ts": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
    }


def _kb_growth(conn: Any) -> dict:
    """KB node counts by source + last-7d deltas.

    Order matters: the by_source query MUST run before the last_7d_delta
    query so the mock (and prod queries) yield correct row sets.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT source, COUNT(*) FROM strategy_nodes WHERE active = TRUE GROUP BY source")
        by_source = {s: int(n) for s, n in cur.fetchall()}
        cur.execute(
            "SELECT source, COUNT(*) FROM strategy_nodes "
            "WHERE active = TRUE AND created_at > now() - INTERVAL '7 days' GROUP BY source"
        )
        last_7d_delta = {s: int(n) for s, n in cur.fetchall()}
    return {
        "total_nodes": sum(by_source.values()),
        "by_source": by_source,
        "last_7d_delta": last_7d_delta,
    }


def _tsdb_dsn_from_env() -> str:
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
