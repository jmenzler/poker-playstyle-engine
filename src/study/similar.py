"""CLI-02 / FastAPI ``/api/similar`` — k nearest clusters via Milvus.

Output columns (D-13): ``rank, cluster_key, distance, source, n_obs``.
JSON form additionally includes ``action_dist`` preview per cluster
(``include_action_dist=True``) by joining strategy_nodes.

Collection dispatch:
    Preflop clusters (cluster_key contains ``street_class=preflop``) hit the
    ``preflop_decisions`` Milvus collection.  Postflop hits
    ``postflop_decisions``.  Mirrors ``src/metrics/ev_loss.py`` dispatch
    logic exactly (locked per CONTEXT.md Decision 3).
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from src._log import get_logger
from src.db import milvus as milvus_db
from src.db import timescale

log = get_logger("study.similar")


def find_similar(
    cluster_key: str,
    *,
    k: int = 10,
    include_action_dist: bool = False,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
) -> list[dict]:
    """Return top-k nearest clusters from Milvus.

    Args:
        cluster_key:          Source cluster (must exist in ``strategy_nodes``
                              with ``active = TRUE``).
        k:                    Number of neighbors (default 10 per CLI-02).
        include_action_dist:  If True, JOIN strategy_nodes for action_dist
                              preview per row (JSON form per D-13).
        _tsdb_conn:           Test-injection hook.
        _milvus:              Test-injection hook.

    Returns:
        ``list[dict]`` (length ≤ k); empty list if the source cluster is not
        present in strategy_nodes.
    """
    log.info("study.similar.started", cluster_key=cluster_key, k=k)
    own_tsdb = _tsdb_conn is None
    own_milvus = _milvus is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    client = _milvus if _milvus is not None else milvus_db.connect_from_env()
    try:
        # Step 1: Source-cluster embedding lookup (representative = first
        # active row for this cluster_key).
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM strategy_nodes WHERE cluster_key = %s AND active = TRUE LIMIT 1",
                (cluster_key,),
            )
            row = cur.fetchone()
        if row is None:
            log.warning("study.similar.not_found", cluster_key=cluster_key)
            return []
        embedding = row[0]

        # Step 2: collection dispatch by street_class token (same idiom as
        # src/metrics/ev_loss.py:98).
        collection = "preflop_decisions" if "street_class=preflop" in cluster_key else "postflop_decisions"

        # Step 3: ANN search — ask for k+1 so we can drop the source cluster
        # if Milvus returns it first.
        results = client.search(
            collection_name=collection,
            data=[embedding],
            limit=k + 1,
            anns_field="embedding",
            output_fields=["cluster_key", "source"],
        )

        hits = results[0] if results else []
        out: list[dict] = []
        rank = 1
        for hit in hits:
            ck = hit.entity.get("cluster_key")
            if ck == cluster_key:
                continue  # skip self
            out.append(
                {
                    "rank": rank,
                    "cluster_key": ck,
                    "distance": float(hit.distance),
                    "source": hit.entity.get("source", "unknown"),
                    "n_obs": _get_n_obs(conn, ck),
                }
            )
            rank += 1
            if rank > k:
                break

        # Step 4: optional action_dist preview (JSON form per D-13).
        if include_action_dist:
            for r in out:
                r["action_dist"] = _get_action_dist(conn, r["cluster_key"])
        log.info("study.similar.complete", n_hits=len(out))
        return out
    finally:
        if own_tsdb:
            with contextlib.suppress(Exception):
                conn.close()
        if own_milvus and hasattr(client, "close"):
            with contextlib.suppress(Exception):
                client.close()


def _get_n_obs(conn: Any, cluster_key: str) -> int:
    """Per-cluster cumulative observation count."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM observations WHERE cluster_key = %s", (cluster_key,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def _get_action_dist(conn: Any, cluster_key: str) -> dict | None:
    """Active strategy_nodes.action_dist for the given cluster_key, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT action_dist FROM strategy_nodes WHERE cluster_key = %s AND active = TRUE LIMIT 1",
            (cluster_key,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _tsdb_dsn_from_env() -> str:
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
