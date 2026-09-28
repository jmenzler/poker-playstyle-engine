"""MatchResult struct + persistence helpers for the ``matches`` hypertable (migration 014).

Per D-NEW-30: every Eval head-to-head run produces one ``MatchResult`` row
covering bb/100 + CI bounds + per-street + per-texture breakdown + top-5
profitable + top-5 leaky cluster_keys.

Schema reference: migrations/014_matches.sql (15 columns; composite PK
``(match_id, started_at)`` because the table is a TimescaleDB hypertable).

Threat-mitigation:
    T-06-22 (engine_version repudiation): the value is sourced from
    ``pyproject.toml`` inside ``run_match.py``, NEVER from user / API input.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import msgspec

from src._log import get_logger
from src.db import timescale

log = get_logger("eval.result")


class MatchResult(msgspec.Struct, frozen=True, kw_only=True):
    """One row of the ``matches`` hypertable.

    Attributes:
        match_id: UUID identifying the match.
        opponent: REGISTRY name (e.g. ``'random'``, ``'tight-passive'``).
        hands: Number of hands the match ran for.
        seed: Seed used to drive deterministic RLCard env.
        bb_per_100: Mean bb_delta per 100 hands.
        ci_low: Lower bound of 95% bootstrap CI on bb/100.
        ci_high: Upper bound of 95% bootstrap CI on bb/100.
        per_street: ``{preflop, flop, turn, river}`` -> bb_delta total.
        per_texture: ``{dry_rainbow, wet_two_tone, paired, monotone}`` -> bb_delta.
        top5_profitable: ``[{cluster_key, n_decisions, bb_total}]`` (positive bb).
        top5_leaky: same shape, negative bb (most-negative first).
        status: ``'won'|'lost'|'inconclusive'|'regression'|'failed'`` etc.
            (See migration 014 CHECK for full set.)
        engine_version: ``pyproject.toml [project].version`` at match start.
        started_at: UTC timestamp the match began.
        finished_at: UTC timestamp the match finished.
    """

    match_id: str
    opponent: str
    hands: int
    seed: int
    bb_per_100: float
    ci_low: float
    ci_high: float
    per_street: dict[str, float]
    per_texture: dict[str, float]
    top5_profitable: list[dict]
    top5_leaky: list[dict]
    status: str
    engine_version: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


def persist_match(result: MatchResult, *, _tsdb_conn: Any = None) -> None:
    """Insert a ``MatchResult`` into the ``matches`` hypertable.

    Args:
        result: The MatchResult to persist.
        _tsdb_conn: Optional injected psycopg connection (tests). When None,
            opens a fresh connection via ``src.db.timescale.connect`` using the
            env-var DSN (TSDB_HOST / TSDB_PORT / TSDB_DB / TSDB_USER /
            TSDB_PASSWORD) and closes it on exit.

    Raises:
        psycopg.Error: any underlying DB failure propagates; the surrounding
            transaction rolls back via ``conn.transaction()``.
    """
    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO matches (match_id, opponent, hands, seed, bb_per_100, "
                "                     ci_low, ci_high, per_street, per_texture, "
                "                     top5_profitable, top5_leaky, status, "
                "                     engine_version, started_at, finished_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, "
                "        %s::jsonb, %s::jsonb, %s, %s, %s, %s)",
                (
                    result.match_id,
                    result.opponent,
                    result.hands,
                    result.seed,
                    result.bb_per_100,
                    result.ci_low,
                    result.ci_high,
                    json.dumps(result.per_street),
                    json.dumps(result.per_texture),
                    json.dumps(result.top5_profitable),
                    json.dumps(result.top5_leaky),
                    result.status,
                    result.engine_version,
                    result.started_at,
                    result.finished_at,
                ),
            )
        log.info("eval.result.persisted", match_id=result.match_id, status=result.status)
    finally:
        if own_conn:
            conn.close()


def load_match(match_id: str, *, _tsdb_conn: Any = None) -> MatchResult | None:
    """Load a single match by ``match_id`` (most-recent row when duplicates exist).

    Args:
        match_id: UUID to look up.
        _tsdb_conn: Optional injected psycopg connection (tests).

    Returns:
        MatchResult instance if a row exists; None otherwise.
    """
    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_id, opponent, hands, seed, bb_per_100, ci_low, ci_high, "
                "       per_street, per_texture, top5_profitable, top5_leaky, status, "
                "       engine_version, started_at, finished_at "
                "FROM matches WHERE match_id = %s ORDER BY started_at DESC LIMIT 1",
                (match_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return MatchResult(
            match_id=str(row[0]),
            opponent=row[1],
            hands=int(row[2]),
            seed=int(row[3]),
            bb_per_100=float(row[4]) if row[4] is not None else 0.0,
            ci_low=float(row[5]) if row[5] is not None else 0.0,
            ci_high=float(row[6]) if row[6] is not None else 0.0,
            per_street=row[7] or {},
            per_texture=row[8] or {},
            top5_profitable=row[9] or [],
            top5_leaky=row[10] or [],
            status=row[11],
            engine_version=row[12],
            started_at=row[13],
            finished_at=row[14],
        )
    finally:
        if own_conn:
            conn.close()


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (TSDB_PASSWORD required).

    Copied from src/patch_engine.py + src/metrics/sink.py to keep this module
    self-contained — same pattern used by every TSDB-touching helper in the
    codebase.
    """
    import os

    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
