"""CLI-07 list + FastAPI ``/api/patches`` — reverse-chronological patches.

Per D-14 + D-NEW-29:
    * Flat reverse-chrono list with columns
      ``patch_id, ts, cluster_key, source, pre_ev_loss, post_ev_loss,
      status, prev_node_id, new_node_id, prev_patch_id``.
    * ``prev_patch_id`` is computed by a correlated subquery against the
      same ``patches`` table — the frontend uses this to reconstruct a chain
      tree without re-fetching.
    * ``patch_detail(patch_id, include_chain=True)`` returns a single dict
      plus a flat ``chain`` list of all patches sharing that cluster_key
      (chronological).
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from src._log import get_logger
from src.db import timescale

log = get_logger("study.patches")


def list_patches(
    *,
    limit: int = 20,
    cluster_key: str | None = None,
    source: str | None = None,
    _tsdb_conn: Any = None,
) -> list[dict]:
    """Reverse-chronological patches list with chain metadata.

    Args:
        limit:        Cap on returned rows (CLI default = 20).
        cluster_key:  Optional filter — only rows for this cluster.
        source:       Optional filter — ``'autoloop'`` or ``'manual'``
                      (per migration 004 CHECK constraint).
        _tsdb_conn:   Test-injection hook.

    Returns:
        ``list[dict]`` ordered by ``ts DESC``.  Empty list when no rows match.
    """
    log.info("study.patches.list_patches.started", limit=limit, cluster_key=cluster_key)
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        sql = (
            "SELECT p.patch_id, p.ts, p.cluster_key, p.source, "
            "       p.pre_ev_loss, p.post_ev_loss, "
            "       COALESCE(p.status, 'applied') AS status, "
            "       p.prev_node_id, p.new_node_id, "
            "       (SELECT pp.patch_id FROM patches pp "
            "        WHERE pp.cluster_key = p.cluster_key AND pp.ts < p.ts "
            "        ORDER BY pp.ts DESC LIMIT 1) AS prev_patch_id "
            "FROM patches p WHERE 1=1"
        )
        params: list[object] = []
        if cluster_key is not None:
            sql += " AND p.cluster_key = %s"
            params.append(cluster_key)
        if source is not None:
            sql += " AND p.source = %s"
            params.append(source)
        sql += " ORDER BY p.ts DESC LIMIT %s"
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            col = [d[0] for d in cur.description]
            rows = cur.fetchall()
        result = [dict(zip(col, r, strict=True)) for r in rows]
        log.info("study.patches.list_patches.complete", n_rows=len(result))
        return result
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def patch_detail(
    patch_id: str,
    *,
    include_chain: bool = True,
    _tsdb_conn: Any = None,
) -> dict | None:
    """Single-patch detail + optional chain reconstruction.

    Args:
        patch_id:       Target patch UUID (string form accepted).
        include_chain:  If True, attach ``chain`` key with all patches sharing
                        the same cluster_key, chronologically ordered.
        _tsdb_conn:     Test-injection hook.

    Returns:
        ``dict`` with the patch's columns + optional ``chain`` list, or
        ``None`` when no patch matches ``patch_id``.
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT p.patch_id, p.ts, p.cluster_key, p.source, p.pre_ev_loss, p.post_ev_loss, "
                "       COALESCE(p.status, 'applied') AS status, p.prev_node_id, p.new_node_id, "
                "       p.decision_id, "
                "       sn_new.action_dist AS override_action_dist, "
                "       sn_prev.action_dist AS original_action_dist "
                "FROM patches p "
                "LEFT JOIN strategy_nodes sn_new ON sn_new.node_id = p.new_node_id "
                "LEFT JOIN strategy_nodes sn_prev ON sn_prev.node_id = p.prev_node_id "
                "WHERE p.patch_id = %s",
                (patch_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            col = [d[0] for d in cur.description]
        detail = dict(zip(col, row, strict=True))
        # hand_id is the DP id minus its _dp<idx> suffix; replay endpoints key off it.
        decision_id = detail.get("decision_id")
        detail["hand_id"] = decision_id.rsplit("_dp", 1)[0] if decision_id else None
        if include_chain:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT patch_id, ts, source, pre_ev_loss, post_ev_loss "
                    "FROM patches WHERE cluster_key = %s ORDER BY ts ASC",
                    (detail["cluster_key"],),
                )
                chain_col = [d[0] for d in cur.description]
                chain_rows = cur.fetchall()
            detail["chain"] = [dict(zip(chain_col, r, strict=True)) for r in chain_rows]
        return detail
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
