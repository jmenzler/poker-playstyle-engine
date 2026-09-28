"""Persistence layer for CFR-narrowed per-seat ranges (harvest_ranges table).

Mirrors src/eval/solver_cache.py: msgspec model + writer (upsert) + reader.
Migration: 019_harvest_ranges.sql.

Security: T-17-03 — all SQL uses parameterized %s placeholders; hand_id and
decision_id are never interpolated into the query string.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import msgspec

from src._log import get_logger
from src.db import timescale

log = get_logger("study.harvest_ranges")


class HarvestRangeRow(msgspec.Struct, frozen=True, kw_only=True):
    """One row of the harvest_ranges table.

    seat: 0=OOP, 1=IP (matches postflop-solver player indexing).
    payload: per-seat slice of the harvest output for one decision-point.
    """

    hand_id: str
    decision_id: str
    seat: int
    payload: dict[str, Any]
    solved_at: datetime | None = None


def assert_grid_block(
    hero_grid: dict[str, Any], villain_grid: dict[str, Any], actions: list[str], decision_id: str
) -> None:
    """Assert per-block length invariants before persisting. Fail loudly on violation."""
    n_h = len(hero_grid["combos"])
    n_a = len(actions)
    n_v = len(villain_grid["combos"])

    if len(hero_grid["weights"]) != n_h:
        raise ValueError(f"{decision_id}: hero_grid weights len {len(hero_grid['weights'])} != combos {n_h}")
    if len(hero_grid["equity"]) != n_h:
        raise ValueError(f"{decision_id}: hero_grid equity len {len(hero_grid['equity'])} != combos {n_h}")
    if len(hero_grid["strategy"]) != n_a * n_h:
        raise ValueError(
            f"{decision_id}: hero_grid strategy len {len(hero_grid['strategy'])} != n_actions({n_a}) * combos({n_h})"
        )
    if len(hero_grid["ev_detail"]) != n_a * n_h:
        raise ValueError(
            f"{decision_id}: hero_grid ev_detail len {len(hero_grid['ev_detail'])} != n_actions({n_a}) * combos({n_h})"
        )
    if len(villain_grid["weights"]) != n_v:
        raise ValueError(
            f"{decision_id}: villain_grid weights len {len(villain_grid['weights'])} != combos {n_v}"
        )
    if len(villain_grid["equity"]) != n_v:
        raise ValueError(
            f"{decision_id}: villain_grid equity len {len(villain_grid['equity'])} != combos {n_v}"
        )


def build_seat_payloads(
    raw_result: dict[str, Any],
    *,
    street: str,
    hero_seat: int,
    hero_action: str | None,
    decision_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the (hero, villain) harvest payloads from one raw nav_ok harvest result.

    Asserts the grid block invariants first (fail loud before any persist).
    """
    hero_grid = raw_result["hero_grid"]
    villain_grid = raw_result["villain_grid"]
    actions = raw_result["actions"]
    assert_grid_block(hero_grid, villain_grid, actions, decision_id)

    hero_payload: dict[str, Any] = {
        "narrowing": "ok",
        "street": street,
        "hero_seat": hero_seat,
        "combos": hero_grid["combos"],
        "weights": hero_grid["weights"],
        "equity": hero_grid["equity"],
        "strategy": hero_grid["strategy"],
        "ev_detail": hero_grid["ev_detail"],
        "actions": actions,
        "hero_action": hero_action,
    }
    villain_payload: dict[str, Any] = {
        "narrowing": "ok",
        "street": street,
        "hero_seat": hero_seat,
        "combos": villain_grid["combos"],
        "weights": villain_grid["weights"],
        "equity": villain_grid["equity"],
    }
    return hero_payload, villain_payload


def persist_harvest_range(row: HarvestRangeRow, *, _tsdb_conn: Any = None) -> None:
    """Upsert one HarvestRangeRow into harvest_ranges.

    Uses upsert (ON CONFLICT ... DO UPDATE) — safe to call multiple times for
    the same DP+seat; later payload replaces earlier.

    Args:
        row: The HarvestRangeRow to persist.
        _tsdb_conn: Optional injected psycopg connection (tests). When None,
            opens a fresh connection via timescale.connect and closes on exit.

    Raises:
        psycopg.Error: any underlying DB failure propagates.
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO harvest_ranges (hand_id, decision_id, seat, payload, solved_at) "
                "VALUES (%s, %s, %s, %s::jsonb, COALESCE(%s, now())) "
                "ON CONFLICT (decision_id, seat) DO UPDATE "
                "  SET hand_id = EXCLUDED.hand_id, payload = EXCLUDED.payload, solved_at = EXCLUDED.solved_at",
                (
                    row.hand_id,
                    row.decision_id,
                    row.seat,
                    json.dumps(row.payload),
                    row.solved_at,
                ),
            )
        log.info(
            "study.harvest_ranges.persisted",
            hand_id=row.hand_id,
            decision_id=row.decision_id,
            seat=row.seat,
        )
    finally:
        if own_conn:
            conn.close()


def load_harvest_ranges_by_hand(hand_id: str, *, _tsdb_conn: Any = None) -> list[HarvestRangeRow]:
    """All harvest_ranges rows for a hand, ordered dp0..dpN then by seat.

    Uses the integer-suffix sort trick from get_replay_by_hand so dp10 sorts
    after dp9, not between dp1 and dp2. Parameterized query only — hand_id is
    never interpolated into the SQL string.

    Args:
        hand_id: The hand to load rows for.
        _tsdb_conn: Optional injected psycopg connection (tests).

    Returns:
        list[HarvestRangeRow] ordered by dp-index ASC, seat ASC.

    Raises:
        psycopg.Error: any underlying DB failure propagates.
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT hand_id, decision_id, seat, payload, solved_at "
                "FROM harvest_ranges WHERE hand_id = %s "
                "ORDER BY NULLIF(split_part(decision_id, '_dp', 2), '')::int ASC NULLS LAST, seat ASC",
                (hand_id,),
            )
            rows = cur.fetchall()
        return [
            HarvestRangeRow(
                hand_id=str(r[0]),
                decision_id=str(r[1]),
                seat=int(r[2]),
                payload=r[3] if isinstance(r[3], dict) else json.loads(r[3]),
                solved_at=r[4],
            )
            for r in rows
        ]
    finally:
        if own_conn:
            conn.close()


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (TSDB_PASSWORD required)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
