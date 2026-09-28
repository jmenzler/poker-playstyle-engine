"""Phase 3 Plan 02 contract tests — DecisionEngine + blending.

Covers requirements: ENGN-01, ENGN-02, ENGN-03, ENGN-04, ENGN-05, ERR-03.
Tests are SKIPPED until Plan 02 implements src/decision_engine/. The skip
reason names the implementing plan so /gsd-execute-phase knows when to unskip.
"""

from __future__ import annotations

import pytest

# Plan-tagged skip-reason constants were used by earlier waves to stub tests
# until each plan unskipped its owned tests. All Phase-3 decision-engine tests
# are now unskipped (Plans 02 + 04), so PLAN_02/PLAN_04 are no longer needed.


def test_decide_returns_canonical_action(mock_milvus_client, preflop_game_state, tmp_manifest_dir):
    """ENGN-01: decide() returns one of 15 canonical actions; no DB writes."""
    from src.decision_engine.blending import CANONICAL_ACTIONS
    from src.decision_engine.engine import KNNDecisionEngine

    engine = KNNDecisionEngine(
        mock_milvus_client,
        tmp_manifest_dir / "zscore_preflop.json",
        tmp_manifest_dir / "zscore_postflop.json",
    )
    action = engine.decide(preflop_game_state)
    assert action in CANONICAL_ACTIONS, f"{action!r} not in 15-action vocab"
    # No DB writes: mock_milvus_client.upsert was NEVER called
    assert mock_milvus_client.upsert.call_count == 0
    assert mock_milvus_client.insert.call_count == 0


def test_milvus_filter_active_true(mock_milvus_client, preflop_game_state, tmp_manifest_dir):
    """ENGN-02: Milvus search filter MUST contain `active == True` AND canonical scalars."""
    from src.decision_engine.engine import KNNDecisionEngine

    engine = KNNDecisionEngine(
        mock_milvus_client,
        tmp_manifest_dir / "zscore_preflop.json",
        tmp_manifest_dir / "zscore_postflop.json",
        use_preflop_charts=False,
    )
    engine.decide(preflop_game_state)
    call_kwargs = mock_milvus_client.search.call_args.kwargs
    filter_expr = call_kwargs.get("filter", "")
    # ENGN-02 requires the `active=true` filter on the kNN retrieval. After Plan 01
    # added the `active` BOOL scalar field to the Milvus DP collections, the engine
    # MUST include it in the filter expression.
    assert "active == True" in filter_expr or "active==True" in filter_expr, (
        f"ENGN-02 violation: filter must include `active == True`: got {filter_expr!r}"
    )
    # Plus canonical hard_filter scalars from the encoder
    assert "street_class" in filter_expr or "street" in filter_expr, (
        f"filter must include street scalar: got {filter_expr!r}"
    )


def test_blend_formula_weighted_distribution(synthetic_kNN_results):
    """ENGN-03: blend formula weights each neighbor by similarity * confidence * gto_score.

    All three synthetic neighbors have confidence=1.0 and gto_score=1.0, so the
    blended weights collapse to similarity alone. A separate test
    (test_blend_formula_with_confidence_gto) below covers the full formula.
    """
    from src.decision_engine.blending import blend_distributions

    dist = blend_distributions(synthetic_kNN_results, collection="postflop_decisions")
    # 3 neighbors: distances 0.1, 0.2, 0.4 -> similarities 0.9, 0.8, 0.6
    # Confidence=gto_score=1.0 -> weight = similarity * 1.0 * 1.0 = similarity
    # Actions: call, raise_3x, fold
    # Total weight = 2.3; call=0.9/2.3=0.391, raise_3x=0.8/2.3=0.348, fold=0.6/2.3=0.261
    assert abs(dist["call"] - 0.391) < 0.01, f"call weight = {dist['call']}"
    assert abs(dist["raise_3x"] - 0.348) < 0.01, f"raise_3x weight = {dist['raise_3x']}"
    assert abs(dist["fold"] - 0.261) < 0.01, f"fold weight = {dist['fold']}"
    # Distribution sums to 1.0 (within float epsilon)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_blend_formula_with_confidence_gto():
    """ENGN-03 (full formula): weight = similarity * confidence * gto_score.

    Three neighbors all at the same similarity (sim=0.5); their relative blend
    weight is driven entirely by confidence * gto_score.
    """
    from src.decision_engine.blending import blend_distributions

    results = [
        {
            "distance": 0.5,
            "entity": {
                "decision_id": "n1",
                "hero_action_type": "call",
                "confidence": 1.0,
                "gto_score": 1.0,
            },
        },
        {
            "distance": 0.5,
            "entity": {
                "decision_id": "n2",
                "hero_action_type": "fold",
                "confidence": 0.5,
                "gto_score": 1.0,
            },
        },
        {
            "distance": 0.5,
            "entity": {
                "decision_id": "n3",
                "hero_action_type": "raise_3x",
                "confidence": 1.0,
                "gto_score": 0.25,
            },
        },
    ]
    dist = blend_distributions(results, collection="postflop_decisions")
    # weights: call=0.5*1.0*1.0=0.50, fold=0.5*0.5*1.0=0.25, raise_3x=0.5*1.0*0.25=0.125
    # total=0.875; call=0.571, fold=0.286, raise_3x=0.143
    assert abs(dist["call"] - 0.571) < 0.01, f"call={dist['call']}"
    assert abs(dist["fold"] - 0.286) < 0.01, f"fold={dist['fold']}"
    assert abs(dist["raise_3x"] - 0.143) < 0.01, f"raise_3x={dist['raise_3x']}"
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_no_strategy_error(mock_milvus_client_empty, preflop_game_state, tmp_manifest_dir):
    """ERR-03: NoStrategyError raised when Milvus returns zero results.

    The error message MUST include the collection name and the hard_filter dict
    so the caller can diagnose the empty-neighbor cause.
    """
    from src._errors import NoStrategyError
    from src.decision_engine.engine import KNNDecisionEngine

    engine = KNNDecisionEngine(
        mock_milvus_client_empty,
        tmp_manifest_dir / "zscore_preflop.json",
        tmp_manifest_dir / "zscore_postflop.json",
        use_preflop_charts=False,
    )
    with pytest.raises(NoStrategyError) as exc_info:
        engine.decide(preflop_game_state)
    msg = str(exc_info.value)
    assert "preflop_decisions" in msg, f"collection missing from error: {msg}"
    assert "street_class" in msg or "pot_type" in msg, f"hard_filter missing: {msg}"


def test_p99_mac_benchmark(mock_milvus_client, preflop_game_state, tmp_manifest_dir):
    """ENGN-04: p99 < 50ms over 1,000 decisions on Mac (mocked Milvus)."""
    import time

    import numpy as np

    from src.decision_engine.engine import KNNDecisionEngine

    engine = KNNDecisionEngine(
        mock_milvus_client,
        tmp_manifest_dir / "zscore_preflop.json",
        tmp_manifest_dir / "zscore_postflop.json",
        rng_seed=42,
    )
    # Warmup: 10 iterations discarded (warm import-time caches, numpy/encoder/
    # manifest loading). Match tools/bench_knn_p99.py warm-cache pattern.
    for _ in range(10):
        engine.decide(preflop_game_state)
    # Bench loop
    latencies_ms = []
    for _ in range(1000):
        t0 = time.perf_counter()
        engine.decide(preflop_game_state)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    p99 = float(np.percentile(latencies_ms, 99))
    p50 = float(np.percentile(latencies_ms, 50))
    print(f"\nENGN-04 Mac bench: n=1000 p50={p50:.3f}ms p99={p99:.3f}ms")
    assert p99 < 50.0, f"ENGN-04 violated: p99={p99:.2f}ms exceeds 50ms budget"


def test_decide_with_encoding_single_encode(mock_milvus_client, preflop_game_state, tmp_manifest_dir):
    """ENGN-05 prereq: decide_with_encoding(gs) returns (action, EncodeResult) with ONE encode call.

    The harness uses this to avoid double-encoding (encode once for cluster_key,
    encode again inside decide). Plan 04 task 1 uses enc.cluster_key directly
    for the observation row.
    """
    from unittest.mock import patch

    from src.canonicalizer import EncodeResult
    from src.decision_engine.blending import CANONICAL_ACTIONS
    from src.decision_engine.engine import KNNDecisionEngine

    engine = KNNDecisionEngine(
        mock_milvus_client,
        tmp_manifest_dir / "zscore_preflop.json",
        tmp_manifest_dir / "zscore_postflop.json",
    )
    # Spy on the encoder; expect EXACTLY one call per decide_with_encoding
    with patch.object(engine._canon, "encode", wraps=engine._canon.encode) as spy:
        action, _flagged_sparse, _max_dist, enc = engine.decide_with_encoding(preflop_game_state)
        assert spy.call_count == 1, f"expected 1 encode call, got {spy.call_count}"
    assert isinstance(enc, EncodeResult)
    assert action in CANONICAL_ACTIONS


def test_preflop_lru_cache_hits(mock_milvus_client, preflop_game_state, tmp_manifest_dir):
    """ENGN-05: LRU preflop result cache hits on repeated identical inputs.

    Second decide() with the same GameState MUST NOT call client.search again
    (cache key = (hard_filter, embedding bytes hash)). Cache size = 4096.
    """
    from src.decision_engine.engine import KNNDecisionEngine

    engine = KNNDecisionEngine(
        mock_milvus_client,
        tmp_manifest_dir / "zscore_preflop.json",
        tmp_manifest_dir / "zscore_postflop.json",
        use_preflop_charts=False,
    )
    engine.decide(preflop_game_state)
    first_call_count = mock_milvus_client.search.call_count
    assert first_call_count == 1, f"expected 1 Milvus search after first decide, got {first_call_count}"
    # Second decide with identical state -> LRU cache MUST serve, no new search
    engine.decide(preflop_game_state)
    second_call_count = mock_milvus_client.search.call_count
    assert second_call_count == 1, (
        f"LRU cache miss: expected 1 Milvus search after second identical decide, got {second_call_count}. "
        f"Cache key (hard_filter, embedding bytes hash) failed."
    )


@pytest.mark.integration
def test_p99_sim_benchmark(milvus_uri, milvus_token, tmp_manifest_dir):
    """ENGN-05: p99 < 5ms over 10,000 decisions on PC with in-process Milvus + hot cache.

    BINARY pass/fail — no defer escape hatch. If p99 >= 5 ms, the test FAILS and
    Phase 3 cannot advance. Plan 04 step 5 runs this on PC against live Milvus
    populated by Plan 01's schema rebuild.

    Integration-marked but NOT skip-gated: when env vars MILVUS_HOST + MILVUS_TOKEN
    + TSDB_PASSWORD are set and the test runs with `-m integration`, it executes.
    On Mac it is excluded by `-m 'not integration'` (default Mac quick run).
    """
    import os
    import time

    import numpy as np

    from src.decision_engine.engine import engine_from_env
    from src.protocols.game_state import GameState

    # Hard-skip ONLY if PC env vars are absent — never silently pass.
    if not os.environ.get("MILVUS_HOST") or not os.environ.get("MILVUS_PORT"):
        pytest.skip("requires PC env vars MILVUS_HOST + MILVUS_PORT — run on external test services")

    engine = engine_from_env(rng_seed=1234)
    # Use a sweep of GameStates to exercise hot cache + cache misses realistically
    gs_list = [
        GameState(
            street="preflop",
            hero_position=pos,
            hero_hole_cards=hc,
            board_cards=(),
            pot_size_bb=3.0,
            effective_stack_bb=100.0,
            hero_facing_bet_bb=2.5,
            hero_bet_size_bb=0.0,
            action_sequence=(),
            opponents_remaining=2,
            prior_street_aggressor=None,
        )
        for pos in ("BTN", "CO", "MP")
        for hc in (("Ah", "Kd"), ("Qs", "Qd"), ("7c", "7d"), ("Ts", "Jh"))
    ]
    # Warmup (cache + Milvus load + Python imports) — discarded
    for _ in range(100):
        engine.decide(gs_list[0])
    # Bench: 10,000 decisions over the sweep
    latencies_ms = []
    for i in range(10_000):
        gs = gs_list[i % len(gs_list)]
        t0 = time.perf_counter()
        engine.decide(gs)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    p50 = float(np.percentile(latencies_ms, 50))
    p99 = float(np.percentile(latencies_ms, 99))
    print(f"\nENGN-05 PC sim-path bench: n=10000 p50={p50:.3f}ms p99={p99:.3f}ms")
    assert p99 < 5.0, (
        f"ENGN-05 violation: p99={p99:.2f}ms exceeds 5ms budget (binary fail; no defer). "
        f"p50={p50:.2f}ms. Inspect LRU hit rate, ef tuning, hot-node warmup population."
    )


def test_legalize_unfaced_remaps_passive_and_fold_to_check() -> None:
    """Facing no bet: fold and call both become check (no free-fold, no phantom call)."""
    from src.decision_engine.blending import legalize_actions

    d = legalize_actions({"fold": 0.5, "call": 0.2, "bet_50": 0.3}, facing_bet_bb=0.0)
    assert "fold" not in d and "call" not in d
    assert d["check"] == pytest.approx(0.7)
    assert d["bet_50"] == pytest.approx(0.3)


def test_legalize_unfaced_remaps_raise_family_to_bet() -> None:
    """Facing no bet, a raise vote is a bet of the matching size (not a 1bb min-raise)."""
    from src.decision_engine.blending import legalize_actions

    d = legalize_actions(
        {"raise_min": 0.4, "raise_2_5x": 0.3, "raise_3x": 0.2, "raise_pot": 0.1}, facing_bet_bb=0.0
    )
    assert d == pytest.approx({"bet_25": 0.4, "bet_50": 0.3, "bet_75": 0.2, "bet_100": 0.1})


def test_legalize_faced_remaps_check_and_bet_family_to_raise() -> None:
    """Facing a bet: check becomes call; a bet vote becomes a raise of matching size."""
    from src.decision_engine.blending import legalize_actions

    d = legalize_actions({"check": 0.3, "bet_25": 0.3, "bet_50": 0.2, "bet_100": 0.2}, facing_bet_bb=2.0)
    assert "check" not in d and not any(k.startswith("bet") for k in d)
    assert d == pytest.approx({"call": 0.3, "raise_min": 0.3, "raise_2_5x": 0.2, "raise_pot": 0.2})


def test_legalize_keeps_node_correct_verbs_unchanged() -> None:
    """Verbs already legal for the node pass through untouched (renormalized)."""
    from src.decision_engine.blending import legalize_actions

    assert legalize_actions({"fold": 0.5, "call": 0.5}, facing_bet_bb=2.0) == pytest.approx(
        {"fold": 0.5, "call": 0.5}
    )
    assert legalize_actions({"check": 0.6, "bet_50": 0.4}, facing_bet_bb=0.0) == pytest.approx(
        {"check": 0.6, "bet_50": 0.4}
    )
