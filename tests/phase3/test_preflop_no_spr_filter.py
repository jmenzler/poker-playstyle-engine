"""Preflop SPR is ill-defined and the stored preflop spr_x100 values are noisy,
so banding on it strips ~98% of 4bet examples. The decide-time spr predicate
must apply postflop only.
"""

from __future__ import annotations


def test_preflop_search_filter_omits_spr(mock_milvus_client, preflop_game_state, tmp_manifest_dir):
    from src.decision_engine.engine import KNNDecisionEngine

    engine = KNNDecisionEngine(
        mock_milvus_client,
        tmp_manifest_dir / "zscore_preflop.json",
        tmp_manifest_dir / "zscore_postflop.json",
        use_preflop_charts=False,
    )
    engine.decide(preflop_game_state)
    filter_expr = mock_milvus_client.search.call_args.kwargs.get("filter", "")
    assert "spr_x100" not in filter_expr, f"preflop must not band on spr: {filter_expr!r}"


def test_postflop_search_filter_keeps_spr(mock_milvus_client, flop_game_state, tmp_manifest_dir):
    from src.decision_engine.engine import KNNDecisionEngine

    engine = KNNDecisionEngine(
        mock_milvus_client,
        tmp_manifest_dir / "zscore_preflop.json",
        tmp_manifest_dir / "zscore_postflop.json",
    )
    engine.decide(flop_game_state)
    filter_expr = mock_milvus_client.search.call_args.kwargs.get("filter", "")
    assert "spr_x100" in filter_expr, f"postflop must still band on spr: {filter_expr!r}"
