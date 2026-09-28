"""CLI-04 / FastAPI ``/api/leaks`` — leak ranking.

Returns:
    * ``coverage`` rows: sparse-kNN observations (delegates to
      ``src/study/leaks_coverage.py``).
    * ``strategy`` rows: ev_loss-ranked clusters with empirical-Bayes shrinkage
      per ``(street_class, pot_type)`` bucket (D-05, D-06).

NEVER recomputes ``ev_loss`` — reads pre-computed values from the ``metrics``
hypertable (populated by Phase 5 sink).  Stage B (solver verification) lives in
``src/study/solver_verify.py`` (Wave 2 sibling plan, owned by 06-05).

Stage A filters applied (per D-12 + D-NEW-28):
    1. ``min_n`` guard (default 20 per D-08).
    2. ``strategy_nodes.locked_from_autoloop = TRUE`` excluded (D-NEW-26).
    3. ``leak_suppressions WHERE active = TRUE`` excluded (D-NEW-28).
"""

from __future__ import annotations

import contextlib
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np

from src._log import get_logger
from src.db import timescale
from src.study.eb_shrinkage import ci_95, is_shrunk, normal_normal_eb
from src.study.leaks_coverage import rank_coverage_gaps

log = get_logger("study.leaks")

# A5: within-cluster variance default; StudyConfig.eb_default_sigma2 override OK.
_DEFAULT_SIGMA2 = 0.05

# SQL: per-cluster aggregated ev_loss + session count + source, excluding
# locked-from-autoloop and active-suppression cluster_keys.
#
# Notes:
# - We treat ``metrics`` as authoritative for ``ev_loss`` (Phase 5 sink writes
#   one row per (session, cluster) per metric_name).  AVG over rows in the
#   recent window collapses session-level noise; the bucket-level EB then
#   shrinks per-cluster means toward the bucket prior.
# - LEFT JOIN on strategy_nodes so clusters without an active node still
#   surface (source defaults to 'unknown' via COALESCE).
# - The leak_suppressions LEFT JOIN + WHERE ls.suppression_id IS NULL is an
#   anti-join — equivalent to ``NOT EXISTS`` but reads cleaner here.
_STRATEGY_AGGREGATE_QUERY = """
SELECT
    m.cluster_key,
    AVG(m.value)                       AS xbar,
    COUNT(*)                           AS n_sessions,
    COALESCE(sn.source, 'unknown')     AS source
FROM metrics m
LEFT JOIN strategy_nodes sn
       ON sn.cluster_key = m.cluster_key AND sn.active = TRUE
LEFT JOIN leak_suppressions ls
       ON ls.cluster_key = m.cluster_key AND ls.active = TRUE
WHERE m.metric_name = 'ev_loss'
  AND m.ts > now() - INTERVAL '30 days'
  AND COALESCE(sn.locked_from_autoloop, FALSE) = FALSE
  AND ls.suppression_id IS NULL
GROUP BY m.cluster_key, sn.source
HAVING COUNT(*) >= 1
"""

# Per-cluster cumulative observation count.
_OBS_COUNT_BY_CLUSTER = "SELECT cluster_key, COUNT(*) AS n_obs FROM observations GROUP BY cluster_key"


def _bucket_from_cluster_key(ck: str) -> tuple[str, str]:
    """Parse ``(street_class, pot_type)`` from a cluster_key for EB pooling.

    cluster_key format example:
        ``street_class=flop|pot_type=srp|hero_pos_rel=ip|...``

    Mirrors ``src/patch_engine.py::_parse_cluster_key`` but returns only the
    two bucket-relevant fields.  Missing fields default to ``"?"`` (the
    "unknown bucket") so a malformed key falls into its own pool and never
    crashes ranking.
    """
    sc = re.search(r"street_class=([^|]+)", ck)
    pt = re.search(r"pot_type=([^|]+)", ck)
    return (sc.group(1) if sc else "?", pt.group(1) if pt else "?")


def rank_leaks(
    *,
    leak_type: str = "both",  # 'coverage' | 'strategy' | 'both'
    min_n: int = 20,  # D-08
    limit: int = 20,
    sigma2: float = _DEFAULT_SIGMA2,
    _tsdb_conn: Any = None,
) -> dict:
    """Top-level entry point used by both CLI-04 and FastAPI ``/api/leaks``.

    Args:
        leak_type:    ``"coverage"``, ``"strategy"``, or ``"both"`` (default).
        min_n:        Strategy-leak min-observations gate per D-08.
        limit:        Cap on rows in EACH section.
        sigma2:       Within-cluster variance assumption for EB.
        _tsdb_conn:   Test-injection hook.

    Returns:
        Dict with keys ``coverage``, ``strategy``, ``bucket_stats``.  When a
        section is filtered out by ``leak_type`` it is returned as an empty
        list / empty dict.
    """
    log.info("study.leaks.started", leak_type=leak_type, min_n=min_n, limit=limit)
    out: dict[str, Any] = {"coverage": [], "strategy": [], "bucket_stats": {}}
    if leak_type in ("coverage", "both"):
        out["coverage"] = rank_coverage_gaps(limit=limit, _tsdb_conn=_tsdb_conn)
    if leak_type in ("strategy", "both"):
        strategy_rows, bucket_stats = _rank_strategy_with_eb(min_n, limit, sigma2, _tsdb_conn)
        out["strategy"] = strategy_rows
        out["bucket_stats"] = bucket_stats
    log.info("study.leaks.complete", n_coverage=len(out["coverage"]), n_strategy=len(out["strategy"]))
    return out


def _rank_strategy_with_eb(
    min_n: int,
    limit: int,
    sigma2: float,
    conn: Any,
) -> tuple[list[dict], dict]:
    """Stage A: empirical-Bayes shrinkage applied per ``(street_class, pot_type)`` bucket."""
    own_conn = conn is None
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
    try:
        # Step 1: fetch aggregated ev_loss per cluster.
        with conn.cursor() as cur:
            cur.execute(_STRATEGY_AGGREGATE_QUERY)
            rows = cur.fetchall()
            col = [d[0] for d in cur.description]
        agg = [dict(zip(col, r, strict=True)) for r in rows]

        # Step 2: fetch n_obs per cluster (filter < min_n early).
        with conn.cursor() as cur:
            cur.execute(_OBS_COUNT_BY_CLUSTER)
            obs_rows = cur.fetchall()
        n_obs_map = {ck: int(n) for ck, n in obs_rows}

        # Step 3: bucket by (street_class, pot_type) and apply EB per bucket.
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in agg:
            ck = r["cluster_key"]
            n_obs = n_obs_map.get(ck, 0)
            if n_obs < min_n:
                continue
            r["n_obs"] = n_obs
            buckets[_bucket_from_cluster_key(ck)].append(r)

        bucket_stats: dict[str, dict] = {}
        results: list[dict] = []
        for (street_class, pot_type), items in buckets.items():
            xbar = np.array([float(it["xbar"]) for it in items])
            n = np.array([float(it["n_obs"]) for it in items])
            post_mean, post_var, mu_hat, tau2_hat = normal_normal_eb(xbar, n, sigma2)
            ci_low, ci_high = ci_95(post_mean, post_var)
            bucket_key = f"{street_class}|{pot_type}"
            bucket_stats[bucket_key] = {
                "mu_hat": float(mu_hat),
                "tau2_hat": float(tau2_hat),
                "k_clusters": len(items),
            }
            for i, it in enumerate(items):
                results.append(
                    {
                        "cluster_key": it["cluster_key"],
                        "ev_loss": float(post_mean[i]),
                        "ci_low": float(ci_low[i]),
                        "ci_high": float(ci_high[i]),
                        "n_obs": int(it["n_obs"]),
                        "source": it["source"],
                        # Solver-verified clusters (source IN ('solver','solver_verify')) -> Stage B.
                        "stage": "B" if it["source"] in ("solver", "solver_verify") else "A",
                        "shrunk": is_shrunk(float(it["n_obs"]), tau2_hat, sigma2),
                        # Score = post_mean * log(max(n_obs, 2))
                        # matches src/autoloop/leak_detector.py ranking formula.
                        "score": float(post_mean[i]) * math.log(max(int(it["n_obs"]), 2)),
                    }
                )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit], bucket_stats
    finally:
        if own_conn:
            with contextlib.suppress(Exception):
                conn.close()


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (mirrors leak_detector.py)."""
    import os

    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
