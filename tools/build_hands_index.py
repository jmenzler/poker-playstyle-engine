"""Populate hands_index (hand-level backing for the Hands tab) from the decision
corpus JSONL (source='hh', played_ts from HM3) + observations (source='sim').
observations is left untouched. CLI: python -m tools.build_hands_index.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

from src._log import get_logger
from src.db import timescale

log = get_logger("tools.build_hands_index")

_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
DEFAULT_BATCH_SIZE = 5_000
PROGRESS_EVERY = 25_000

_UPSERT_SQL = (
    "INSERT INTO hands_index "
    "(hand_id, source, hero_position, stake, n_decisions, street_reached, played_ts) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (hand_id) DO UPDATE SET "
    "source = EXCLUDED.source, hero_position = EXCLUDED.hero_position, "
    "stake = EXCLUDED.stake, n_decisions = EXCLUDED.n_decisions, "
    "street_reached = EXCLUDED.street_reached, played_ts = EXCLUDED.played_ts"
)


def _deepest_street(streets: Iterable[str | None]) -> str | None:
    """The deepest street seen, by preflop<flop<turn<river order. None if all unknown."""
    best: str | None = None
    best_rank = -1
    for s in streets:
        rank = _STREET_ORDER.get(s or "", -1)
        if rank > best_rank:
            best_rank, best = rank, s
    return best


def _parse_hm3_ts(raw: str | None) -> datetime | None:
    """Parse an HM3 handtimestamp ('2023-02-10 04:52:36') to a UTC datetime. None on failure."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def aggregate_real_hands(
    decisions: Iterable[dict], ts_lookup: dict[str, datetime | None]
) -> list[dict[str, Any]]:
    """Group decision records by hand_id into one hands_index row each (source='hh')."""
    acc: dict[str, dict[str, Any]] = {}
    for d in decisions:
        hid = d.get("hand_id")
        if not hid:
            continue
        row = acc.get(hid)
        if row is None:
            acc[hid] = {
                "hand_id": hid,
                "source": "hh",
                "hero_position": d.get("hero_pos"),
                "stake": d.get("stake"),
                "n_decisions": 1,
                "_streets": [d.get("street")],
                "played_ts": ts_lookup.get(hid),
            }
        else:
            row["n_decisions"] += 1
            row["_streets"].append(d.get("street"))
    out = []
    for row in acc.values():
        row["street_reached"] = _deepest_street(row.pop("_streets"))
        out.append(row)
    return out


def load_hm3_timestamps(hm3_path: str, hand_ids: set[str]) -> dict[str, datetime | None]:
    """Bulk-load handtimestamp for the given hand_ids from the HM3 sqlite (read-only)."""
    if not hand_ids:
        return {}
    conn = sqlite3.connect(f"file:{hm3_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        out: dict[str, datetime | None] = {}
        # Pull all handtimestamps once; the handhistories table is keyed by gamenumber.
        cur.execute("SELECT gamenumber, handtimestamp FROM handhistories")
        for gamenumber, ts in cur:
            if gamenumber in hand_ids:
                out[gamenumber] = _parse_hm3_ts(ts)
        return out
    finally:
        conn.close()


def aggregate_sim_hands(conn: Any) -> list[dict[str, Any]]:
    """Aggregate observations (source='sim') into one hands_index row per hand_id."""
    sql = (
        "SELECT hand_id, "
        "COUNT(*) AS n_decisions, "
        "MIN(ts) AS played_ts, "
        "(array_agg(felt_snapshot->>'hero_position' ORDER BY ts))[1] AS hero_position, "
        "array_agg(felt_snapshot->>'street') AS streets "
        "FROM observations WHERE source = 'sim' AND hand_id IS NOT NULL "
        "GROUP BY hand_id"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    out = []
    for hand_id, n_decisions, played_ts, hero_position, streets in rows:
        out.append(
            {
                "hand_id": hand_id,
                "source": "sim",
                "hero_position": hero_position,
                "stake": None,
                "n_decisions": int(n_decisions),
                "street_reached": _deepest_street(streets or []),
                "played_ts": played_ts,
            }
        )
    return out


def upsert_hands_index(conn: Any, rows: list[dict[str, Any]], *, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """Batched idempotent upsert. Returns rows written. Commits per batch."""
    written = 0
    batch: list[tuple] = []
    for row in rows:
        batch.append(
            (
                row["hand_id"],
                row["source"],
                row.get("hero_position"),
                row.get("stake"),
                row.get("n_decisions", 0),
                row.get("street_reached"),
                row.get("played_ts"),
            )
        )
        if len(batch) >= batch_size:
            written += _flush(conn, batch)
            batch = []
            if written % PROGRESS_EVERY < batch_size:
                log.info("hands_index.upsert.progress", written=written)
    if batch:
        written += _flush(conn, batch)
    return written


def _flush(conn: Any, batch: list[tuple]) -> int:
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, batch)
    conn.commit()
    return len(batch)


def _iter_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_hands_index(
    *,
    decisions_path: str | None,
    dsn: str | None = None,
    hm3_path: str | None = None,
    _conn: Any = None,
) -> dict[str, int]:
    """Populate hands_index from the decision corpus (HM) + observations (SIM).

    Returns {"hh": n_real, "sim": n_sim, "total": n}.
    """
    conn = _conn if _conn is not None else timescale.connect(dsn)  # type: ignore[arg-type]
    try:
        n_hh = 0
        if decisions_path:
            decisions = list(_iter_jsonl(decisions_path))
            hand_ids = {d["hand_id"] for d in decisions if d.get("hand_id")}
            ts_lookup = load_hm3_timestamps(hm3_path, hand_ids) if hm3_path else {}
            real_rows = aggregate_real_hands(decisions, ts_lookup)
            n_hh = upsert_hands_index(conn, real_rows)
            log.info("hands_index.real.done", n=n_hh)
        sim_rows = aggregate_sim_hands(conn)
        n_sim = upsert_hands_index(conn, sim_rows)
        log.info("hands_index.sim.done", n=n_sim)
        return {"hh": n_hh, "sim": n_sim, "total": n_hh + n_sim}
    finally:
        if _conn is None:
            conn.close()


def _build_dsn_from_env() -> str:
    import os

    return (
        f"host={os.environ['TSDB_HOST']} port={os.environ['TSDB_PORT']} "
        f"dbname={os.environ['TSDB_DB']} user={os.environ['TSDB_USER']} "
        f"password={os.environ['TSDB_PASSWORD']}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default=None, help="decision corpus JSONL (HM source)")
    ap.add_argument("--hm3", default=None, help="HM3 sqlite path for played_ts join")
    args = ap.parse_args()
    result = build_hands_index(decisions_path=args.decisions, dsn=_build_dsn_from_env(), hm3_path=args.hm3)
    print(f"hands_index populated: {result}")
