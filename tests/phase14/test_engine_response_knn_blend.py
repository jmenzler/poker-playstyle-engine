"""engine_response falls back to the kNN blend when no strategy_nodes override exists.

Mirrors how KNNDecisionEngine actually decides (blend_distributions over neighbors),
so the probe shows the engine's real action — not an empty 'no strategy node' panel —
in the RAG topology where strategy_nodes is empty (D-07-11e).
"""

from __future__ import annotations

import src.study.probe as probe_mod


def _nbr(distance, action, gto=1.0, conf=1.0):
    return {"distance": distance, "hero_action_type": action, "gto_score": gto, "confidence": conf}


def test_engine_response_from_knn_blends_neighbors():
    neighbors = [_nbr(0.1, "call"), _nbr(0.2, "fold"), _nbr(0.15, "call")]
    r = probe_mod._engine_response_from_knn(neighbors, "postflop_decisions")
    assert r is not None
    assert r["source"] == "knn_blend"
    assert "empty_reason" not in r
    assert r["n_obs"] == 3
    assert abs(sum(r["action_dist"].values()) - 1.0) < 1e-6
    # two close 'call' neighbors outweigh one 'fold'
    assert r["action_dist"]["call"] > r["action_dist"]["fold"]
    # zero-prob actions are filtered out of the display dist
    assert all(p > 0 for p in r["action_dist"].values())


def test_engine_response_from_knn_none_on_zero_weight():
    # all non-canonical labels → zero blend weight → None (keep the empty sentinel)
    assert probe_mod._engine_response_from_knn([_nbr(0.1, "bogus_label")], "postflop_decisions") is None


def test_engine_response_from_knn_defaults_missing_scores():
    neighbors = [{"distance": 0.1, "hero_action_type": "check", "gto_score": None, "confidence": None}]
    r = probe_mod._engine_response_from_knn(neighbors, "postflop_decisions")
    assert r is not None and r["action_dist"].get("check", 0) > 0


def test_engine_response_from_knn_preflop_snaps_raise_to_3bet():
    """In an srp spot (facing an open), a neighbor 'raise' must surface as a 3bet,
    not the misleading raise_2_5x open bucket."""
    neighbors = [_nbr(0.1, "raise"), _nbr(0.2, "fold")]
    ctx = {"pot_type": "srp", "hero_pos_rel": "OOP", "hero_pos": "SB"}
    r = probe_mod._engine_response_from_knn(neighbors, "preflop_decisions", preflop_ctx=ctx)
    assert r is not None
    assert "3bet_4x" in r["action_dist"]  # OOP 3bet
    assert "raise_2_5x" not in r["action_dist"]
    assert r["action_dist"]["3bet_4x"] > r["action_dist"]["fold"]


def test_engine_response_from_knn_postflop_snaps_bet_size():
    """Postflop: a neighbor 'bet' snaps to its real size bucket (not the bet_50 alias)."""
    neighbors = [
        {
            "distance": 0.1,
            "hero_action_type": "bet",
            "hero_action_size_pot_frac": 0.75,
            "raise_ratio": -1.0,
            "hero_action_allin": False,
        },
        {"distance": 0.5, "hero_action_type": "check"},
    ]
    r = probe_mod._engine_response_from_knn(neighbors, "postflop_decisions")
    assert r is not None
    assert "bet_75" in r["action_dist"]
    assert "bet_50" not in r["action_dist"]
