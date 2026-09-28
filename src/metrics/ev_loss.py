"""ev_loss: KL divergence between observed and expected action distributions.

Public API:
    ev_loss(cluster_key, session_id=None, *, _tsdb_conn=None, _milvus=None) -> float

Algorithm (CONTEXT.md Decision 3, locked):
    1. Query TimescaleDB observations for rows in cluster_key (+ optional session filter).
       Collect action_taken strings → observed frequency dict P.
       If count < K=10 → raise NoStrategyError.
    2. Pick a representative observation (latest by ts or last row if already ordered DESC).
       Retrieve its embedding, query Milvus k=10 neighbors → blend via blend_distributions
       (same formula as decide-time) → expected distribution Q.
       If Milvus returns empty → raise NoStrategyError.
    3. Compute KL(P || Q) with additive smoothing ε=1e-6 (CONTEXT.md Decision 3).
       Return float.

Idempotency (LABL-02):
    Within a single DB snapshot, the same cluster_key produces byte-identical floats.
    Between snapshots (new observations arrive) the float may change — this is expected.

Smoothing (ε=1e-6):
    Both P and Q are smoothed by adding ε to every action in the union vocabulary,
    then renormalized. This prevents log(0) when one side has a zero-probability action.
    scipy.stats.entropy is NOT used (returns inf on zero Q — defeats smoothing intent).

Collection dispatch:
    street_class parsed from cluster_key token (e.g. "street_class=postflop").
    "preflop" → preflop_decisions (32d); everything else → postflop_decisions (80d).

Representative embedding:
    ev_loss uses a SINGLE representative embedding per cluster (the most-recent
    observation row, which TimescaleDB returns first when queried DESC by ts).
    This makes ev_loss O(1) Milvus calls. Trade-off: variance if cluster membership
    changes between calls — bounded by LABL-02 (same snapshot → same float).

Test injection:
    _tsdb_conn and _milvus kwargs accept pre-built connections for unit tests.
    Production callers NEVER pass these; connections are opened from env vars.
"""

from __future__ import annotations

import math
import os
from typing import Any

from src._errors import NoStrategyError
from src._log import get_logger
from src.db import milvus as milvus_db
from src.db import timescale
from src.decision_engine.blending import blend_distributions

log = get_logger("metrics.ev_loss")

# k=10: matches decide-time k (CONTEXT.md Decision 3, locked).
_K: int = 10

# ε smoothing: additive smoothing to prevent log(0) in KL computation.
# Value locked by CONTEXT.md Decision 3.
EPSILON: float = 1e-6


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def ev_loss(
    cluster_key: str,
    session_id: str | None = None,
    *,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
) -> float:
    """Compute KL divergence between observed and expected action distributions.

    Args:
        cluster_key: cluster identifier (format: "key=val|key=val|...").
            Must contain a "street_class=preflop" or "street_class=postflop" token.
        session_id: optional session filter; when None, all sessions are included.
        _tsdb_conn: test-injection hook. Pre-built psycopg connection.
            Production callers NEVER pass this.
        _milvus: test-injection hook. Pre-built MilvusClient.
            Production callers NEVER pass this.

    Returns:
        KL(observed_action_freq || expected_action_dist) as a float >= 0.

    Raises:
        NoStrategyError: if cluster_key has < k=10 observations or Milvus returns
            empty results for the representative embedding.
        ValueError: if cluster_key does not contain a street_class token.
    """
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    client = _milvus if _milvus is not None else milvus_db.connect_from_env()

    street_class = _street_class_from_cluster_key(cluster_key)
    collection = "preflop_decisions" if street_class.lower() == "preflop" else "postflop_decisions"

    # Step 1: query observations — returns (action_taken, embedding) tuples.
    # Representative = first row (latest by ts, DESC order).
    obs_rows = _observed_action_rows(conn, cluster_key, session_id)
    if len(obs_rows) < _K:
        raise NoStrategyError(
            f"ev_loss: cluster_key={cluster_key!r} has {len(obs_rows)} observations, "
            f"need >= {_K} (NoStrategyError per CONTEXT.md Decision 3)"
        )

    # Step 2: observed frequency dict P
    p_dict = _observed_action_freq(obs_rows)

    # Step 3: representative embedding → Milvus kNN → expected distribution Q
    representative_embedding: list[float] = list(obs_rows[0][1])
    q_dict = _expected_action_dist(client, representative_embedding, collection)

    # Step 4: KL(P || Q) with additive smoothing
    kl = _kl_divergence(p_dict, q_dict)

    log.info(
        "metrics.ev_loss.computed",
        cluster_key=cluster_key,
        session_id=session_id,
        n_obs=len(obs_rows),
        kl=round(kl, 6),
    )
    return kl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _street_class_from_cluster_key(cluster_key: str) -> str:
    """Extract street_class value from a cluster_key string.

    cluster_key format: "key1=val1|key2=val2|..." (sorted hard_filter items).
    Example: "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop"

    Raises:
        ValueError: if no "street_class=" token is present.
    """
    for token in cluster_key.split("|"):
        if token.startswith("street_class="):
            return token.split("=", 1)[1]
    raise ValueError(f"cluster_key missing street_class token: {cluster_key!r}")


def _observed_action_rows(
    conn: Any,
    cluster_key: str,
    session_id: str | None,
) -> list[Any]:
    """Query TimescaleDB for (action_taken, embedding) rows in cluster_key.

    Returns rows ordered by ts DESC so obs_rows[0] is the most recent —
    used as the representative embedding for Milvus retrieval.

    Args:
        conn: psycopg connection.
        cluster_key: cluster bucket to query.
        session_id: optional session filter; None = all sessions.

    Returns:
        list of (action_taken, embedding) tuples, ordered DESC by ts.
    """
    sql = "SELECT action_taken, embedding FROM observations WHERE cluster_key = %s"
    params: list[Any] = [cluster_key]

    if session_id is not None:
        sql += " AND session_id = %s"
        params.append(session_id)

    sql += " ORDER BY ts DESC"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows: list[Any] = cur.fetchall()
        return rows


def _observed_action_freq(obs_rows: list[tuple[str, Any]]) -> dict[str, float]:
    """Compute normalized action frequency dict from observation rows.

    Args:
        obs_rows: list of (action_taken, embedding) tuples.

    Returns:
        dict mapping action_str → frequency (values sum to 1.0).
    """
    counts: dict[str, int] = {}
    for action, _ in obs_rows:
        counts[action] = counts.get(action, 0) + 1
    total = sum(counts.values())
    return {action: cnt / total for action, cnt in counts.items()}


def _expected_action_dist(
    client: Any,
    embedding: list[float],
    collection: str,
) -> dict[str, float]:
    """Retrieve k=10 Milvus neighbors for embedding and blend into expected dist Q.

    Re-uses blend_distributions() from the decision engine — provably identical
    formula to decide-time blending (zero drift).

    Args:
        client: pymilvus MilvusClient.
        embedding: representative observation embedding vector.
        collection: "preflop_decisions" or "postflop_decisions".

    Returns:
        dict mapping action_str → probability (decide-time blended distribution).

    Raises:
        NoStrategyError: if Milvus search returns empty results.
    """
    results = client.search(
        collection_name=collection,
        data=[embedding],
        limit=_K,
        filter="active == True",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=["decision_id", "hero_action_type", "confidence", "gto_score", "action_dist"],
    )

    if not results or not results[0]:
        raise NoStrategyError(f"ev_loss: Milvus returned empty results for collection={collection!r}")

    return blend_distributions(results[0], collection=collection)


def _total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """TVD between two action distributions over the union vocabulary.

    No smoothing needed — TVD is defined at zero entries unlike KL.
    Result is always in [0.0, 1.0].

    Args:
        p: first action distribution (may be partial vocab).
        q: second action distribution (may be partial vocab).

    Returns:
        0.5 * sum(|p_a - q_a|) over the union of keys.
    """
    actions = set(p) | set(q)
    total = 0.0
    for a in actions:
        total += abs(p.get(a, 0.0) - q.get(a, 0.0))
    return total / 2.0


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(P || Q) with additive smoothing ε=1e-6.

    Both P and Q may have zero entries for actions not present in one side.
    The union of their keys is the full action vocabulary. Additive smoothing
    ε=EPSILON is applied to every action in the vocabulary for both P and Q,
    then both are renormalized. This prevents log(0) regardless of which
    actions appear in P vs Q.

    NOT using scipy.stats.entropy: that function returns inf when Q[a]=0,
    defeating the smoothing intent. Manual implementation per CONTEXT.md
    Decision 3.

    Args:
        p: observed action frequency dict (may be partial vocab).
        q: expected action distribution dict (may be partial vocab).

    Returns:
        KL divergence as float >= 0.
    """
    actions = set(p) | set(q)

    # Additive smoothing: add ε to every action for both distributions.
    p_smooth = {a: p.get(a, 0.0) + EPSILON for a in actions}
    q_smooth = {a: q.get(a, 0.0) + EPSILON for a in actions}

    # Renormalize
    p_sum = sum(p_smooth.values())
    q_sum = sum(q_smooth.values())
    p_norm = {a: v / p_sum for a, v in p_smooth.items()}
    q_norm = {a: v / q_sum for a, v in q_smooth.items()}

    # KL(P || Q) = sum_a p_a * log(p_a / q_a)
    return sum(p_norm[a] * math.log(p_norm[a] / q_norm[a]) for a in actions)


# ---------------------------------------------------------------------------
# Env-var helpers (production path only — tests inject mocks directly)
# ---------------------------------------------------------------------------


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables.

    TSDB_PASSWORD is required; all other vars have sensible defaults.
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
