"""PC-only integration test: ev-loss CLI subcommand end-to-end.

Requires:
    TSDB_HOST, TSDB_PASSWORD — caller-supplied TimescaleDB is reachable
    MILVUS_HOST, MILVUS_TOKEN — Milvus reachable

Inserts ~12 synthetic observations under a unique test session_id,
invokes the ev_loss function directly (in-process, faster than subprocess),
asserts JSON result is parseable with ev_loss float >= 0.
Cleans up inserted rows in finally block.

Marker: @pytest.mark.integration
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pytestmark = pytest.mark.integration

_N_OBS = 12
_TEST_CLUSTER_KEY = "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop"
_POSTFLOP_DIM = 80


def _tsdb_dsn_from_env() -> str:
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def _milvus_client_from_env():
    from src.db import milvus as milvus_db

    host = os.environ["MILVUS_HOST"]
    port = os.environ.get("MILVUS_PORT", "51530")
    raw = os.environ.get("MILVUS_TOKEN")
    token = raw if (raw and ":" in raw) else (f"root:{raw}" if raw else None)
    return milvus_db.connect(f"http://{host}:{port}", token=token)


def _insert_observations(conn, session_id: str) -> None:
    """Insert N_OBS test observations into the observations table."""
    import uuid as _uuid
    from datetime import UTC, datetime

    rows = []
    actions = ["fold", "call", "raise_min"] * 4 + ["fold", "fold"]  # 12 rows, non-uniform
    embedding = [0.01 * i for i in range(_POSTFLOP_DIM)]

    for i, action in enumerate(actions):
        rows.append(
            (
                str(_uuid.uuid4()),  # obs_id
                _TEST_CLUSTER_KEY,  # cluster_key
                embedding,  # embedding
                action,  # action_taken
                "sim",  # source
                session_id,  # session_id
                False,  # flagged_sparse
                datetime.now(UTC),  # ts
            )
        )

    sql = (
        "INSERT INTO observations "
        "(obs_id, cluster_key, embedding, action_taken, source, session_id, flagged_sparse, ts) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    )
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()


def _cleanup(conn, session_id: str) -> None:
    """Delete all observations for the test session."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM observations WHERE session_id = %s", (session_id,))
    conn.commit()


@pytest.mark.integration
def test_ev_loss_cli_integration(tsdb_dsn, milvus_uri, milvus_token):
    """Insert 12 observations, call ev_loss, assert non-negative finite float output."""
    if not os.environ.get("TSDB_PASSWORD"):
        pytest.skip("TSDB_PASSWORD not set — PC-only integration test")
    if not os.environ.get("MILVUS_HOST"):
        pytest.skip("MILVUS_HOST not set — PC-only integration test")

    from src.db import milvus as milvus_db
    from src.db import timescale
    from src.metrics.ev_loss import ev_loss

    session_id = f"test-ev-loss-{uuid.uuid4().hex[:8]}"

    conn = timescale.connect(tsdb_dsn)
    milvus_client = milvus_db.connect(milvus_uri, token=milvus_token)

    try:
        _insert_observations(conn, session_id)

        kl = ev_loss(
            _TEST_CLUSTER_KEY,
            session_id=session_id,
            _tsdb_conn=conn,
            _milvus=milvus_client,
        )

        assert isinstance(kl, float), f"ev_loss must return float; got {type(kl)}"
        assert kl >= 0.0, f"KL divergence must be >= 0; got {kl}"
        assert kl < float("inf"), f"KL divergence must be finite; got {kl}"

    finally:
        _cleanup(conn, session_id)
        conn.close()
