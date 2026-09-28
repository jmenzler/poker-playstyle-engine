"""Phase 9 / LOO gate — unit tests (no real DB/Milvus required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.eval.loo_gate import (
    LooGateResult,
    _build_exclude_filter,
    _cluster_key_to_hard_filter,
    run_loo_gate,
)

# _build_exclude_filter behavior assertions


def test_exclude_filter():
    """LOO Milvus filter excludes the held-out cluster_key from kNN search."""
    # String values are double-quoted; keys sorted deterministically
    filt = _build_exclude_filter({"street_class": "postflop", "pot_type": "srp"})
    assert filt == 'active == True and not (pot_type == "srp" and street_class == "postflop")'

    # Int values are unquoted
    filt_int = _build_exclude_filter({"n_players_active": 2, "street_class": "postflop"})
    assert filt_int == 'active == True and not (n_players_active == 2 and street_class == "postflop")'

    # Must contain lowercase not with parens (Milvus 2.4.13 rejects uppercase NOT)
    assert "not (" in filt
    assert "not (" in filt_int

    # Round-trip: decompose a cluster_key, build filter — scalar fields survive
    cluster_key = "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop"
    hard_filter = _cluster_key_to_hard_filter(cluster_key)
    round_trip = _build_exclude_filter(hard_filter)
    assert "not (" in round_trip
    assert 'hero_pos_rel == "IP"' in round_trip
    assert "n_players_active == 2" in round_trip
    assert 'pot_type == "srp"' in round_trip
    assert 'street_class == "postflop"' in round_trip


def test_cluster_key_to_hard_filter_types():
    """n_players_active is cast to int; all other scalar values stay as str."""
    cluster_key = "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop"
    hf = _cluster_key_to_hard_filter(cluster_key)
    assert isinstance(hf["n_players_active"], int)
    assert hf["n_players_active"] == 2
    assert isinstance(hf["street_class"], str)
    assert hf["street_class"] == "postflop"
    assert hf["pot_type"] == "srp"
    assert hf["hero_pos_rel"] == "IP"


# run_loo_gate unit tests (mock TSDB + Milvus)

_CLUSTER_KEY = "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop"

_SOLVER_TARGET = {
    "check": 0.0,
    "fold": 0.0,
    "call": 0.5,
    "bet_50": 0.3,
    "raise_2_5x": 0.2,
}

_NEIGHBOR_DIST = {
    "check": 0.0,
    "fold": 0.0,
    "call": 0.4,
    "bet_50": 0.4,
    "raise_2_5x": 0.2,
}


def _make_mock_tsdb(cluster_keys, visit_counts, obs_embedding, last_tvd=None):
    """Build a mock psycopg connection that serves fixture data."""
    conn = MagicMock()

    def cursor_factory():
        cur = MagicMock()
        cur.__enter__ = lambda s: s
        cur.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def execute_side_effect(sql, params=None):
            call_count[0] += 1
            sql_lower = sql.lower().strip()

            if "from solver_cache" in sql_lower:
                cur._last_query = "solver_cache_distinct"
            elif "from observations group by" in sql_lower:
                cur._last_query = "visit_counts"
            elif "metric_name = 'tvd_loo'" in sql_lower or "metric_name = %s" in sql_lower:
                cur._last_query = "last_tvd"
            elif "from observations where" in sql_lower:
                cur._last_query = "obs_rows"
            elif "from solver_cache where" in sql_lower:
                cur._last_query = "solver_cache_load"
            else:
                cur._last_query = "unknown"

        def fetchall_side_effect():
            q = getattr(cur, "_last_query", "unknown")
            if q == "solver_cache_distinct":
                return [(k,) for k in cluster_keys]
            elif q == "visit_counts":
                return [(k, v) for k, v in visit_counts.items()]
            return []

        def fetchone_side_effect():
            q = getattr(cur, "_last_query", "unknown")
            if q == "last_tvd":
                return (last_tvd,) if last_tvd is not None else None
            elif q == "solver_cache_load":
                import json

                return (
                    _CLUSTER_KEY,
                    json.dumps(_SOLVER_TARGET),
                    5.0,
                    None,
                    "fixture",
                    None,
                )
            elif q == "obs_rows":
                return (obs_embedding,)
            return None

        cur.execute = execute_side_effect
        cur.fetchall = fetchall_side_effect
        cur.fetchone = fetchone_side_effect
        return cur

    conn.cursor = cursor_factory
    return conn


def _make_mock_milvus(neighbor_dist):
    """Build a mock MilvusClient that returns a fixed neighbor set via blend_distributions."""
    client = MagicMock()

    # Simulate two neighbor hits that will blend to neighbor_dist proportionally
    # We mock the entire path: search returns results, blend_distributions produces dist
    mock_hit_1 = {
        "decision_id": "d1",
        "hero_action_type": "call",
        "confidence": 0.8,
        "gto_score": 0.5,
        "distance": 0.95,
    }
    mock_hit_2 = {
        "decision_id": "d2",
        "hero_action_type": "bet_50",
        "confidence": 0.9,
        "gto_score": 0.4,
        "distance": 0.9,
    }
    client.search.return_value = [[mock_hit_1, mock_hit_2]]
    return client


def test_run_loo_gate_frequency_weighting():
    """Frequency weighting: clusters with more observations dominate the aggregate TVD."""
    embedding = [0.1] * 32

    mock_conn = _make_mock_tsdb(
        cluster_keys=[_CLUSTER_KEY],
        visit_counts={_CLUSTER_KEY: 10},
        obs_embedding=embedding,
        last_tvd=None,
    )
    mock_client = _make_mock_milvus(_NEIGHBOR_DIST)

    from src._config import EvalConfig

    cfg = EvalConfig(tvd_floor=0.99, regression_delta=0.99)

    with (
        patch("src.eval.loo_gate._observed_action_rows") as mock_obs,
        patch("src.eval.loo_gate.blend_distributions") as mock_blend,
        patch("src.eval.solver_cache.load_solve") as mock_load_solve,
    ):
        mock_obs.return_value = [("call", embedding)]
        mock_blend.return_value = _NEIGHBOR_DIST

        from src.eval.solver_cache import SolverCacheEntry

        mock_load_solve.return_value = SolverCacheEntry(
            cluster_key=_CLUSTER_KEY,
            action_dist=_SOLVER_TARGET,
            exploitability_pct=5.0,
        )

        result = run_loo_gate(_tsdb_conn=mock_conn, _milvus=mock_client, _cfg=cfg)

    assert isinstance(result, LooGateResult)
    assert result.n_clusters == 1
    assert result.coverage_ok is True
    # call: |0.4-0.5|=0.1, bet_50:|0.4-0.3|=0.1, raise_2_5x:|0.2-0.2|=0 → TVD=0.1
    assert abs(result.tier1_tvd - 0.1) < 1e-6
    assert result.last_run_tvd is None
    assert result.regression_delta == 0.0


def test_run_loo_gate_pass_fail_logic():
    """Gate fails if TVD >= tvd_floor OR regression_delta > cfg.regression_delta."""
    embedding = [0.1] * 32

    from src._config import EvalConfig
    from src.eval.solver_cache import SolverCacheEntry

    def _run_with_cfg(tvd_floor, regression_delta_cfg, last_tvd):
        mock_conn = _make_mock_tsdb(
            cluster_keys=[_CLUSTER_KEY],
            visit_counts={_CLUSTER_KEY: 10},
            obs_embedding=embedding,
            last_tvd=last_tvd,
        )
        mock_client = _make_mock_milvus(_NEIGHBOR_DIST)
        cfg = EvalConfig(tvd_floor=tvd_floor, regression_delta=regression_delta_cfg)

        with (
            patch("src.eval.loo_gate._observed_action_rows") as mock_obs,
            patch("src.eval.loo_gate.blend_distributions") as mock_blend,
            patch("src.eval.solver_cache.load_solve") as mock_load_solve,
        ):
            mock_obs.return_value = [("call", embedding)]
            mock_blend.return_value = _NEIGHBOR_DIST
            mock_load_solve.return_value = SolverCacheEntry(
                cluster_key=_CLUSTER_KEY,
                action_dist=_SOLVER_TARGET,
                exploitability_pct=5.0,
            )

            return run_loo_gate(_tsdb_conn=mock_conn, _milvus=mock_client, _cfg=cfg)

    # TVD ~0.1; floor=0.99; last_tvd=None → PASS
    r = _run_with_cfg(tvd_floor=0.99, regression_delta_cfg=0.99, last_tvd=None)
    assert r.passed is True

    # TVD ~0.1; floor=0.05 (below actual TVD) → FAIL
    r_fail_floor = _run_with_cfg(tvd_floor=0.05, regression_delta_cfg=0.99, last_tvd=None)
    assert r_fail_floor.passed is False

    # TVD ~0.1; last_tvd=0.01; delta=0.09 > regression_delta_cfg=0.05 → FAIL (regression)
    r_fail_regr = _run_with_cfg(tvd_floor=0.99, regression_delta_cfg=0.05, last_tvd=0.01)
    assert r_fail_regr.passed is False
    assert abs(r_fail_regr.regression_delta - 0.09) < 1e-6


def test_run_loo_gate_insufficient_coverage():
    """Gate reports coverage_ok=False and passed=False when no clusters have observations."""
    embedding = [0.1] * 32

    mock_conn = _make_mock_tsdb(
        cluster_keys=[_CLUSTER_KEY],
        visit_counts={_CLUSTER_KEY: 10},
        obs_embedding=embedding,
        last_tvd=None,
    )
    mock_client = _make_mock_milvus(_NEIGHBOR_DIST)

    from src._config import EvalConfig
    from src.eval.solver_cache import SolverCacheEntry

    cfg = EvalConfig(tvd_floor=0.99, regression_delta=0.99)

    with (
        patch("src.eval.loo_gate._observed_action_rows") as mock_obs,
        patch("src.eval.solver_cache.load_solve") as mock_load_solve,
    ):
        # Return empty obs → cluster skipped → n_clusters_with_obs = 0
        mock_obs.return_value = []
        mock_load_solve.return_value = SolverCacheEntry(
            cluster_key=_CLUSTER_KEY,
            action_dist=_SOLVER_TARGET,
            exploitability_pct=5.0,
        )

        result = run_loo_gate(_tsdb_conn=mock_conn, _milvus=mock_client, _cfg=cfg)

    assert result.coverage_ok is False
    assert result.passed is False


@pytest.mark.integration
def test_exclude_filter_actually_excludes_integration():
    """Integration: NOT (...) filter really excludes held-out cluster nodes from Milvus.

    Requires real Milvus at MILVUS_HOST:MILVUS_PORT with postflop_decisions populated.
    Validates RESEARCH A1 — pymilvus 3.0 NOT-syntax works as expected.
    """
    from src.db.milvus import connect_from_env

    client = connect_from_env()

    cluster_key = "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop"
    hard_filter = _cluster_key_to_hard_filter(cluster_key)
    exclude_filter = _build_exclude_filter(hard_filter)

    # Use a random embedding — we're only checking filter behavior, not result quality
    dummy_embedding = [0.0] * 80

    results = client.search(
        collection_name="postflop_decisions",
        data=[dummy_embedding],
        limit=10,
        filter=exclude_filter,
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=["hero_pos_rel", "n_players_active", "pot_type", "street_class"],
    )

    # Verify none of the returned nodes match all fields of the held-out cluster
    for hit in results[0]:
        entity = hit if isinstance(hit, dict) else hit.get("entity", hit)
        is_held_out = (
            entity.get("hero_pos_rel") == "IP"
            and entity.get("n_players_active") == 2
            and entity.get("pot_type") == "srp"
            and entity.get("street_class") == "postflop"
        )
        assert not is_held_out, (
            f"LOO filter failed — held-out cluster node appeared in neighbor set: {entity}"
        )
