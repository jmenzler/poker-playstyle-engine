"""Tier-1 LOO action-accuracy regression gate (D-09-1..4).

Public API: run_loo_gate, _build_exclude_filter, _cluster_key_to_hard_filter, LooGateResult.
Security (T-9-10): hard_filter values come from the canonicalizer; never raw user input.
"""

from __future__ import annotations

import os
from typing import Any

import msgspec

from src._config import EvalConfig, load_eval_config
from src._log import get_logger
from src.db import milvus as milvus_db
from src.db import timescale
from src.decision_engine.blending import blend_distributions
from src.metrics.ev_loss import (
    _K,
    _observed_action_rows,
    _street_class_from_cluster_key,
    _total_variation_distance,
)

log = get_logger("eval.loo_gate")

# Phase-9 minimum: only 2-3 sample solves exist; raise in future as coverage grows.
_MIN_CLUSTERS: int = 1


class LooGateResult(msgspec.Struct, frozen=True, kw_only=True):
    """Tier-1 LOO gate output.

    passed = coverage_ok AND tier1_tvd < tvd_floor AND regression_delta <= cfg limit.
    regression_delta = tier1_tvd - last_run_tvd (0.0 on first run).
    """

    tier1_tvd: float
    tier1_top1: float
    n_clusters: int
    coverage_ok: bool
    passed: bool
    regression_delta: float
    last_run_tvd: float | None


def _build_exclude_filter(hard_filter: dict) -> str:
    """Build Milvus filter excluding a cluster's own nodes from kNN (LOO negation).

    Milvus 2.4.13 parses lowercase ``not (...)``; uppercase ``NOT`` is rejected
    with a query-plan parse error. Keys sorted for deterministic output (SIM-02).
    Str values double-quoted; int bare.
    """
    parts: list[str] = []
    for key, val in sorted(hard_filter.items()):
        if isinstance(val, bool):
            parts.append(f"{key} == {val}")
        elif isinstance(val, str):
            parts.append(f'{key} == "{val}"')
        else:
            parts.append(f"{key} == {val}")
    inner = " and ".join(parts)
    return f"active == True and not ({inner})"


def _cluster_key_to_hard_filter(cluster_key: str) -> dict:
    """Decompose "k1=v1|k2=v2" cluster_key into a typed hard_filter dict.

    n_players_active is cast to int; all other scalar values stay as str.
    """
    hard_filter: dict = {}
    for token in cluster_key.split("|"):
        key, val = token.split("=", 1)
        if key == "n_players_active":
            hard_filter[key] = int(val)
        else:
            hard_filter[key] = val
    return hard_filter


def run_loo_gate(
    *,
    session_id: str | None = None,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
    _cfg: EvalConfig | None = None,
) -> LooGateResult:
    """Run Tier-1 LOO gate: excludes each cluster's nodes, blends neighbors, computes TVD vs solver target.

    Frequency-weighted aggregate; gates on tvd_floor OR regression vs last run.
    """
    cfg = _cfg if _cfg is not None else load_eval_config()
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    client = _milvus if _milvus is not None else milvus_db.connect_from_env()

    cluster_keys = _fetch_solver_cluster_keys(conn)
    n_clusters = len(cluster_keys)
    visit_counts = _fetch_visit_counts(conn)

    tvd_list: list[float] = []
    top1_list: list[float] = []
    weight_list: list[float] = []

    for cluster_key in cluster_keys:
        entry = _load_solve_safe(conn, cluster_key)
        if entry is None:
            continue

        target = entry.action_dist
        obs_rows = _observed_action_rows(conn, cluster_key, session_id)
        if not obs_rows:
            continue

        visit_count = visit_counts.get(cluster_key, len(obs_rows))
        if visit_count <= 0:
            continue

        street_class = _street_class_from_cluster_key(cluster_key)
        collection = "preflop_decisions" if street_class.lower() == "preflop" else "postflop_decisions"
        hard_filter = _cluster_key_to_hard_filter(cluster_key)
        representative_embedding: list[float] = list(obs_rows[0][1])

        # Exclude held-out cluster's own nodes so the engine reconstructs from neighbors only.
        exclude_filter = _build_exclude_filter(hard_filter)
        results = client.search(
            collection_name=collection,
            data=[representative_embedding],
            limit=_K,
            filter=exclude_filter,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["decision_id", "hero_action_type", "confidence", "gto_score", "action_dist"],
        )

        if not results or not results[0]:
            log.warning("eval.loo_gate.empty_neighbors", cluster_key=cluster_key)
            continue

        q_neighbors = blend_distributions(results[0], collection=collection)
        tvd_i = _total_variation_distance(q_neighbors, target)
        top1_i = 1.0 if _argmax(q_neighbors) == _argmax(target) else 0.0

        tvd_list.append(tvd_i)
        top1_list.append(top1_i)
        weight_list.append(float(visit_count))

    n_clusters_with_obs = len(tvd_list)
    coverage_ok = n_clusters_with_obs >= _MIN_CLUSTERS

    if weight_list:
        total_weight = sum(weight_list)
        tier1_tvd = sum(w * t for w, t in zip(weight_list, tvd_list, strict=True)) / total_weight
        tier1_top1 = sum(w * t for w, t in zip(weight_list, top1_list, strict=True)) / total_weight
    else:
        tier1_tvd = 0.0
        tier1_top1 = 0.0

    last_run_tvd = _fetch_last_run_tvd(conn)
    regression_delta = (tier1_tvd - last_run_tvd) if last_run_tvd is not None else 0.0

    passed = coverage_ok and tier1_tvd < cfg.tvd_floor and regression_delta <= cfg.regression_delta

    result = LooGateResult(
        tier1_tvd=tier1_tvd,
        tier1_top1=tier1_top1,
        n_clusters=n_clusters,
        coverage_ok=coverage_ok,
        passed=passed,
        regression_delta=regression_delta,
        last_run_tvd=last_run_tvd,
    )

    log.info(
        "eval.loo_gate.computed",
        tier1_tvd=round(tier1_tvd, 6),
        n_clusters=n_clusters,
        passed=passed,
    )

    return result


def _fetch_solver_cluster_keys(conn: Any) -> list[str]:
    """Return all distinct cluster_keys from the solver_cache table."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT cluster_key FROM solver_cache")
        rows = cur.fetchall()
    return [row[0] for row in rows]


def _fetch_visit_counts(conn: Any) -> dict[str, int]:
    """Return per-cluster observation counts for frequency weighting."""
    with conn.cursor() as cur:
        cur.execute("SELECT cluster_key, COUNT(*) FROM observations GROUP BY cluster_key")
        rows = cur.fetchall()
    return {row[0]: int(row[1]) for row in rows}


def _fetch_last_run_tvd(conn: Any) -> float | None:
    """Return the most-recent tvd_loo metric value (None if no prior run)."""
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM metrics WHERE metric_name = 'tvd_loo' ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
    return float(row[0]) if row is not None else None


def _load_solve_safe(conn: Any, cluster_key: str) -> Any:
    """Load a solver cache entry, returning None on any error."""
    from src.eval.solver_cache import load_solve

    return load_solve(cluster_key, _tsdb_conn=conn)


def _argmax(dist: dict[str, float]) -> str | None:
    """Return the action with the highest probability, or None if empty."""
    if not dist:
        return None
    return max(dist, key=lambda a: dist[a])


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from env vars (TSDB_PASSWORD required)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
