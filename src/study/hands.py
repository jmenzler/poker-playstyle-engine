# rot-allow-file
"""Hands panel — observation browser + per-observation replay loader.  D-NEW-25.

Phase 10 W3 expands the SELECT to include the three columns added by migration 016:
``hand_id``, ``decision_id``, ``felt_snapshot``.  SIM rows carry populated values;
HM rows and pre-Phase-10 rows carry NULL (which the API layer returns as None).

Schema (migrations/001 + 008 + 016):
    obs_id, cluster_key, embedding, action_taken, ev_realized,
    source, session_id, solver_label, ts, flagged_sparse,
    max_neighbor_distance, hand_id, decision_id, felt_snapshot
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from src._log import get_logger
from src.db import timescale

log = get_logger("study.hands")

_DISTINCT_WHITELIST = frozenset({"session_id", "cluster_key", "obs_id"})


# "solved" = the hand has harvested solver ranges (the range-viewer solve artifact;
# harvest_ranges carries hand_id directly, so no decision_id LIKE/escaping needed).
_SOLVED_EXISTS = "EXISTS (SELECT 1 FROM harvest_ranges hr WHERE hr.hand_id = h.hand_id)"

# "corpus_solved" = a real solver_cache solve (exploitability>0, not a skip marker) covers a
# decision point of this hand — the kNN-corpus artifact (cluster_key/decision_id keyed, written
# by the queue driver), distinct from the interactive harvest_ranges solve the `solved` flag tracks.
_CORPUS_SOLVED_EXISTS = (
    "EXISTS (SELECT 1 FROM observations o "
    "JOIN solver_cache sc ON sc.decision_id = o.decision_id AND sc.exploitability_pct > 0 "
    "WHERE o.hand_id = h.hand_id)"
)

_HANDS_SOURCES = frozenset({"all", "hh", "sim"})


def list_hands(
    *,
    source: str = "all",
    solved: bool | None = None,
    corpus_solved: bool | None = None,
    stake: str | None = None,
    limit: int = 50,
    offset: int = 0,
    _tsdb_conn: Any = None,
) -> list[dict]:
    """Hand-level list from hands_index for the Hands browser, played_ts DESC.

    source: 'all' | 'hh' (real) | 'sim'. solved filters the interactive harvest_ranges
    solve; corpus_solved filters kNN-corpus coverage. Rows carry both derived bools.
    """
    if source not in _HANDS_SOURCES:
        raise ValueError(f"source {source!r} not in {sorted(_HANDS_SOURCES)}")
    log.info("study.hands.list_hands.started", source=source, solved=solved, limit=limit)
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        sql = (
            "SELECT h.hand_id, h.source, h.hero_position, h.stake, h.n_decisions, "
            f"h.street_reached, h.played_ts, {_SOLVED_EXISTS} AS solved, "
            f"{_CORPUS_SOLVED_EXISTS} AS corpus_solved "
            "FROM hands_index h WHERE 1=1"
        )
        params: list[object] = []
        if source != "all":
            sql += " AND h.source = %s"
            params.append(source)
        if stake is not None:
            sql += " AND h.stake = %s"
            params.append(stake)
        if solved is True:
            sql += f" AND {_SOLVED_EXISTS}"
        elif solved is False:
            sql += f" AND NOT {_SOLVED_EXISTS}"
        if corpus_solved is True:
            sql += f" AND {_CORPUS_SOLVED_EXISTS}"
        elif corpus_solved is False:
            sql += f" AND NOT {_CORPUS_SOLVED_EXISTS}"
        sql += " ORDER BY h.played_ts DESC NULLS LAST LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        with conn.cursor() as cur:
            cur.execute(sql, params)
            col = [d[0] for d in cur.description]
            rows = cur.fetchall()
        result = [dict(zip(col, r, strict=True)) for r in rows]
        log.info("study.hands.list_hands.complete", n_rows=len(result))
        return result
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def get_replay(obs_id: str, *, _tsdb_conn: Any = None) -> list[dict]:
    """Single-observation replay payload.

    Migration 016 (Phase 10 W2) added ``hand_id``, ``decision_id``, and
    ``felt_snapshot`` to the observations hypertable.  SIM rows carry all three;
    HM rows and pre-Phase-10 rows carry NULL (returned as ``None``).

    The frontend renders ``felt_snapshot`` inline for SIM rows and falls back to
    text for NULL rows (D-08).
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT obs_id, session_id, ts, cluster_key, action_taken, source, "
                "       solver_label, max_neighbor_distance, flagged_sparse, embedding, "
                "       hand_id, decision_id, felt_snapshot "
                "FROM observations WHERE obs_id = %s ORDER BY ts ASC",
                (obs_id,),
            )
            col = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(col, r, strict=True)) for r in rows]
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def get_replay_by_hand(hand_id: str, *, _tsdb_conn: Any = None) -> list[dict]:
    """All dp rows for a hand, ordered by decision_id ascending (dp0..dpN).

    Sibling of get_replay() — same own-conn pattern and same SELECT column list.
    Intended for the unified by-hand_id replay endpoint (D-06/D-07). Parameterized
    query only; hand_id is never interpolated into the SQL string.
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT obs_id, session_id, ts, cluster_key, action_taken, source, "
                "       solver_label, max_neighbor_distance, flagged_sparse, embedding, "
                "       hand_id, decision_id, felt_snapshot "
                "FROM observations WHERE hand_id = %s "
                # decision_id is f"{hand_id}_dp{idx}"; lexical sort puts _dp10 before _dp2,
                # so order by the integer suffix to keep dp0..dpN in true sequence.
                "ORDER BY NULLIF(split_part(decision_id, '_dp', 2), '')::int ASC NULLS LAST, ts ASC",
                (hand_id,),
            )
            col = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(col, r, strict=True)) for r in rows]
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def list_distinct(field: str, *, limit: int = 200, _tsdb_conn: Any = None) -> list[str]:
    """Distinct values for a whitelisted field — D-14 distinct-values endpoint.

    Args:
        field:      Column name.  Must be in ``_DISTINCT_WHITELIST``; any other
                    value raises ``ValueError`` (no DB access).
        limit:      Maximum number of distinct values to return.
        _tsdb_conn: Test-injection hook.

    Returns:
        Sorted ``list[str]`` of distinct values (up to ``limit``).

    Raises:
        ValueError: ``field`` is not in ``_DISTINCT_WHITELIST``.
    """
    if field not in _DISTINCT_WHITELIST:
        raise ValueError(f"field {field!r} not in whitelist {sorted(_DISTINCT_WHITELIST)}")
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        # Safe: field is allowlist-checked above — not user-supplied SQL injection surface
        sql = f"SELECT DISTINCT {field} FROM observations ORDER BY {field} LIMIT %s"
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return [r[0] for r in cur.fetchall()]
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def _tsdb_dsn_from_env() -> str:
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
