"""kNN-Rebalance: generate candidate action distribution for a leaked cluster.  long-ok

Algorithm (05-CONTEXT.md Decision 1, locked):
  1. Fetch the k_representatives most-recent observation embeddings from TSDB
     for the given cluster_key.
     Raise NoStrategyError if cluster has zero observations.
  2. Capture representative_embedding = list(rep_rows[0][0]) — first row is
     latest by ts DESC, reused as the embedding for the new strategy node.
  3. Dispatch Milvus collection: 'preflop_decisions' if street_class == 'preflop',
     else 'postflop_decisions'.
  4. For each representative embedding: search Milvus for k_neighbors nearest
     active neighbors. If search returns non-empty, call blend_distributions()
     to get a weighted action distribution; catch NoStrategyError (zero-weight
     result — e.g. all confidence=0) and skip that rep.
  5. If no valid blends found across all reps: raise NoStrategyError.
  6. Average the blended distributions across all successful reps.
  7. Renormalize to ensure sum == 1.0 exactly.
  8. Return (action_dist, representative_embedding, representative_decision_id),
     the rep's decision_id keying the patch's Milvus row.

Both read-only: zero TSDB writes, zero Milvus writes.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from src._errors import NoStrategyError
from src._log import get_logger
from src.db import milvus as milvus_db
from src.db import timescale
from src.decision_engine.blending import CANONICAL_ACTIONS, blend_distributions
from src.metrics.ev_loss import _street_class_from_cluster_key

log = get_logger("autoloop.rebalance")

# k defaults (05-CONTEXT.md Decision 1, locked)
_K_REPRESENTATIVES: int = 10
_K_NEIGHBORS: int = 10

# SQL: fetch k most-recent representative observations by ts DESC.
_REPRESENTATIVE_OBS_SQL = (
    "SELECT embedding, decision_id FROM observations WHERE cluster_key = %s ORDER BY ts DESC LIMIT %s"
)


def generate_candidate_action_dist(
    cluster_key: str,
    *,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
    k_representatives: int = _K_REPRESENTATIVES,
    k_neighbors: int = _K_NEIGHBORS,
) -> tuple[dict[str, float], list[float], str]:
    """Generate candidate action distribution via kNN-rebalance.  long-ok

    Args:
        cluster_key: cluster bucket identifier (e.g. "street_class=postflop|...").
        _tsdb_conn: test-injection hook for TSDB connection. Production callers
            NEVER pass this.
        _milvus: test-injection hook for Milvus client. Production callers
            NEVER pass this.
        k_representatives: number of representative observations to fetch from TSDB.
            Default: 10 (locked by 05-CONTEXT.md Decision 1).
        k_neighbors: number of Milvus neighbors per representative search.
            Default: 10 (locked).

    Returns:
        Tuple of:
          - action_dist: dict[action_str -> probability], sums to 1.0 ± 1e-9.
            Contains all CANONICAL_ACTIONS keys.
          - representative_embedding: list[float] from the first (latest-by-ts)
            representative observation. Used as embedding for the new strategy node.
          - representative_decision_id: str — the first rep's decision_id; keys the
            patch's Milvus row. Raises NoStrategyError if that rep has none.

    Raises:
        NoStrategyError: if cluster has zero observations in TSDB, or if Milvus
            returns empty results for ALL representative embeddings.
        ValueError: if cluster_key missing street_class token.
    """
    _owns_conn = _tsdb_conn is None
    _owns_client = _milvus is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    client = _milvus if _milvus is not None else milvus_db.connect_from_env()
    try:
        return _generate_candidate_action_dist_inner(
            cluster_key, conn, client, k_representatives, k_neighbors
        )
    finally:
        if _owns_conn:
            with contextlib.suppress(Exception):
                conn.close()
        if _owns_client:
            with contextlib.suppress(Exception):
                client.close()


def _generate_candidate_action_dist_inner(
    cluster_key: str,
    conn: Any,
    client: Any,
    k_representatives: int = _K_REPRESENTATIVES,
    k_neighbors: int = _K_NEIGHBORS,
) -> tuple[dict[str, float], list[float], str]:
    """Inner implementation of generate_candidate_action_dist (connections already established)."""
    # Step 1: fetch representative observations.
    with conn.cursor() as cur:
        cur.execute(_REPRESENTATIVE_OBS_SQL, (cluster_key, k_representatives))
        rep_rows = cur.fetchall()

    if not rep_rows:
        raise NoStrategyError(f"rebalance: cluster_key={cluster_key!r} has zero observations in TSDB")

    # Step 2: capture representative embedding + DP id (first row = latest by ts).
    representative_embedding: list[float] = list(rep_rows[0][0])
    representative_decision_id = rep_rows[0][1]
    # observations.decision_id is nullable (HM-ingest rows carry none); a patch must be
    # keyed by a replayable DP, so a null rep cannot tag one.
    if representative_decision_id is None:
        raise NoStrategyError(
            f"rebalance: cluster_key={cluster_key!r} — latest observation has no "
            "decision_id (HM-ingest row); cannot tag a replayable patch"
        )

    # Step 3: dispatch collection.
    street_class = _street_class_from_cluster_key(cluster_key)
    collection = "preflop_decisions" if street_class.lower() == "preflop" else "postflop_decisions"

    log.info(
        "autoloop.rebalance.start",
        cluster_key=cluster_key,
        n_reps=len(rep_rows),
        collection=collection,
    )

    # Steps 4-5: search Milvus per rep, blend, collect successful blends.
    blended_dists: list[dict[str, float]] = []

    for i, row in enumerate(rep_rows):
        embedding: list[float] = list(row[0])
        results = client.search(
            collection_name=collection,
            data=[embedding],
            limit=k_neighbors,
            filter="active == True",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["decision_id", "hero_action_type", "confidence", "gto_score", "action_dist"],
        )

        if not results or not results[0]:
            log.info(
                "autoloop.rebalance.milvus_empty_rep",
                cluster_key=cluster_key,
                rep_index=i,
            )
            continue

        try:
            blended = blend_distributions(results[0], collection=collection)
            blended_dists.append(blended)
        except NoStrategyError:
            # All neighbors had zero blend weight for this rep — skip.
            log.info(
                "autoloop.rebalance.blend_zero_weight_rep",
                cluster_key=cluster_key,
                rep_index=i,
            )
            continue

    # Step 5 check: if no reps produced valid blends, fail loudly.
    if not blended_dists:
        raise NoStrategyError(
            f"rebalance: cluster_key={cluster_key!r} -- "
            f"Milvus returned empty results for all {len(rep_rows)} representatives "
            f"(collection={collection!r})"
        )

    # Step 6: average blended distributions across successful reps.
    n = len(blended_dists)
    averaged: dict[str, float] = {a: sum(d.get(a, 0.0) for d in blended_dists) / n for a in CANONICAL_ACTIONS}

    # Step 7: renormalize (defensive — float drift from averaging).
    total = sum(averaged.values())
    if total > 0:
        averaged = {a: p / total for a, p in averaged.items()}

    log.info(
        "autoloop.rebalance.done",
        cluster_key=cluster_key,
        n_valid_reps=n,
        top_action=max(averaged, key=averaged.get),  # type: ignore[arg-type]
    )
    return averaged, representative_embedding, representative_decision_id


# ---------------------------------------------------------------------------
# Env-var helpers (production path only — tests inject mocks directly)
# ---------------------------------------------------------------------------


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables.

    TSDB_PASSWORD is required; all other vars have sensible defaults.
    Copied verbatim from src/metrics/ev_loss.py (same pattern).
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
