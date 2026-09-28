"""SIM-01 + SIM-02 integration tests (PC-gated).

Runs full RECORD sessions against live Milvus + TimescaleDB. Cleans up rows
tagged with the test session_ids in a finally block.

Pre-requisites:
    - explicitly configured test services reachable
    - Phase 2 Milvus collections populated WITH hero_action_type (Plan 01 schema
      bump requires re-running tools/phase2_pipeline.py --rebuild-from build_embedding)
    - tools/zscore_preflop.json + tools/zscore_postflop.json present
    - TSDB_PASSWORD + MILVUS_HOST + MILVUS_PORT + MILVUS_TOKEN env vars set
"""

from __future__ import annotations

import hashlib
import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration


def _run_session(tsdb_dsn: str, session_seed: int, n_hands: int, session_id: str) -> dict:
    """Build engine + adapter + writer and drive one RECORD session.

    Engine is constructed via ``engine_from_env`` so the same Milvus + manifest
    setup that the CLI runner uses is exercised end-to-end.
    """
    from src.decision_engine.engine import engine_from_env
    from src.sim import ObservationWriter, SimAdapter, run_record_session

    engine = engine_from_env(rng_seed=session_seed)
    adapter = SimAdapter()
    with ObservationWriter(tsdb_dsn, session_id, batch_size=100) as writer:
        summary = run_record_session(
            adapter,
            engine,
            writer,
            session_seed=session_seed,
            n_hands=n_hands,
            session_id=session_id,
        )
    return summary


def _fetch_ordered_rows(tsdb_dsn: str, session_id: str) -> list[tuple]:
    """Fetch all observation rows for a session, ordered by (ts, obs_id).

    Returns list of (cluster_key, embedding, action_taken, flagged_sparse,
    max_neighbor_distance) tuples. obs_id and ts are EXCLUDED — they are
    non-deterministic (uuid4 / datetime.now) and are not part of the SIM-02
    checksum.

    SIM-02 contract (Phase 4 Plan 02 extension): all 5 deterministic fields
    must produce byte-identical sequences across two same-seed runs:
        - cluster_key: derived from hard_filter (deterministic given same seed)
        - embedding: from normalize(encode(gs)) (deterministic given same seed)
        - action_taken: sample_action with fixed rng_seed (deterministic)
        - flagged_sparse: derived from same Milvus results (deterministic)
        - max_neighbor_distance: max of same Milvus distances (deterministic)

    Ordering by (ts, obs_id) is deterministic within a single session: rows
    are inserted in decision order, ts is monotonically increasing per flush,
    and obs_id breaks ties within a single batch.
    """
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT cluster_key, embedding, action_taken, "
            "flagged_sparse, max_neighbor_distance "
            "FROM observations "
            "WHERE session_id = %s ORDER BY ts ASC, obs_id ASC",
            (session_id,),
        )
        return list(cur.fetchall())


def _compute_checksum(rows: list[tuple]) -> str:
    """SHA256 over the ordered rows (5-field SIM-02 contract).

    Each row contributes:
        ``cluster_key | embedding(6-dp) | action_taken | flagged_sparse | max_dist \\n``.
    Fixed-precision embedding and distance formatting avoids float-bit-representation
    flakes across runs.
    max_neighbor_distance may be NULL (None in Python) for NoStrategy fallback rows;
    formatted as the literal string "None".
    """
    h = hashlib.sha256()
    for cluster_key, embedding, action_taken, flagged_sparse, max_neighbor_distance in rows:
        h.update(cluster_key.encode())
        h.update(b"|")
        h.update(",".join(f"{x:.6f}" for x in embedding).encode())
        h.update(b"|")
        h.update(action_taken.encode())
        h.update(b"|")
        h.update(str(flagged_sparse).encode())
        h.update(b"|")
        # max_neighbor_distance is REAL (nullable); format to 6dp or "None"
        if max_neighbor_distance is None:
            h.update(b"None")
        else:
            h.update(f"{max_neighbor_distance:.6f}".encode())
        h.update(b"\n")
    return h.hexdigest()


def _cleanup(tsdb_dsn: str, session_ids: list[str]) -> None:
    """Delete all observation rows tagged with the given session_ids."""
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.executemany(
            "DELETE FROM observations WHERE session_id = %s",
            [(sid,) for sid in session_ids],
        )
        conn.commit()


def test_dual_run_checksum(tsdb_dsn: str) -> None:
    """SIM-02: Same RNG seed -> byte-identical Observation rows (verified via SHA256).

    Two independent runs with session_seed=42, n_hands=20 each must produce
    identical (cluster_key, embedding, action_taken) sequences after ordering
    by ts. Failure -> Phase 3 determinism contract broken.
    """
    session_a = f"sim02-a-{uuid.uuid4().hex[:8]}"
    session_b = f"sim02-b-{uuid.uuid4().hex[:8]}"
    try:
        summary_a = _run_session(tsdb_dsn, session_seed=42, n_hands=20, session_id=session_a)
        summary_b = _run_session(tsdb_dsn, session_seed=42, n_hands=20, session_id=session_b)
        assert summary_a["total_decisions"] == summary_b["total_decisions"], (
            f"decision counts diverge: a={summary_a['total_decisions']} b={summary_b['total_decisions']}"
        )

        rows_a = _fetch_ordered_rows(tsdb_dsn, session_a)
        rows_b = _fetch_ordered_rows(tsdb_dsn, session_b)
        assert len(rows_a) == len(rows_b) > 0, f"row count mismatch / empty: a={len(rows_a)} b={len(rows_b)}"

        cs_a = _compute_checksum(rows_a)
        cs_b = _compute_checksum(rows_b)
        assert cs_a == cs_b, (
            f"SIM-02 violation: 5-field checksums diverge "
            f"(cluster_key, embedding, action_taken, flagged_sparse, max_neighbor_distance)\n"
            f"  a={cs_a}\n  b={cs_b}\n"
            f"  total_decisions_a={summary_a['total_decisions']} "
            f"total_decisions_b={summary_b['total_decisions']}"
        )
    finally:
        _cleanup(tsdb_dsn, [session_a, session_b])


def test_throughput_30k_per_hour(tsdb_dsn: str) -> None:
    """SIM-01: RECORD mode >= 30,000 hands/hr (measured over 200 hands).

    200 hands at 30k/hr = 24 seconds upper bound; PC observed rate per
    RESEARCH.md is much higher (~2.4M/hr), so this is a smoke test with
    significant headroom. Failure surfaces if Milvus latency or LRU cache
    behaviour regresses.
    """
    session_id = f"sim01-{uuid.uuid4().hex[:8]}"
    try:
        summary = _run_session(tsdb_dsn, session_seed=99, n_hands=200, session_id=session_id)
        rate = summary["hands_per_hour"]
        assert rate >= 30_000, (
            f"SIM-01 violation: {rate} hands/hr < 30,000 hands/hr target. summary={summary}"
        )
    finally:
        _cleanup(tsdb_dsn, [session_id])
