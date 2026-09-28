"""Engine integration: chart hit before Milvus, miss falls back to kNN, contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.decision_engine.blending import CANONICAL_ACTIONS
from src.decision_engine.engine import KNNDecisionEngine
from src.protocols.game_state import GameState

CHART_DIR = "charts/preflop/inferred_6max"


@pytest.fixture
def manifests(tmp_path: Path) -> tuple[Path, Path]:
    import json

    out = tmp_path / "m"
    out.mkdir()
    for name, dim in (("preflop", 34), ("postflop", 80)):
        (out / f"zscore_{name}.json").write_text(
            json.dumps(
                {
                    "feature_spec_version": 3,
                    "collection": f"{name}_decisions",
                    "dim_stats": [{"dim": i, "name": f"d{i}", "mean": 0.0, "std": 1.0} for i in range(dim)],
                }
            )
        )
    return out / "zscore_preflop.json", out / "zscore_postflop.json"


@pytest.fixture
def mock_milvus():
    client = MagicMock()
    client.search.return_value = [
        [
            {"distance": 0.9, "entity": {"hero_action_type": "fold", "confidence": 1.0, "gto_score": 1.0}},
        ]
    ]
    return client


def _engine(mock_milvus, manifests, *, use_charts=True, seed=7):
    pre, post = manifests
    return KNNDecisionEngine(
        mock_milvus, pre, post, rng_seed=seed, use_preflop_charts=use_charts, preflop_chart_dir=CHART_DIR
    )


def _btn_open_gs() -> GameState:
    return GameState(
        street="preflop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=(),
        pot_size_bb=1.5,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("UTG:fold", "MP:fold", "CO:fold"),
        opponents_remaining=3,
        prior_street_aggressor=None,
    )


def _flop_gs() -> GameState:
    return GameState(
        street="flop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kh"),
        board_cards=("Qh", "Jh", "2s"),
        pot_size_bb=6.5,
        effective_stack_bb=97.5,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("BB:check",),
        opponents_remaining=1,
        prior_street_aggressor="BTN",
    )


def test_chart_hit_returns_canonical_action_no_milvus(mock_milvus, manifests):
    eng = _engine(mock_milvus, manifests)
    action = eng.decide(_btn_open_gs())
    assert action in CANONICAL_ACTIONS
    # chart hit must NOT touch Milvus
    assert mock_milvus.search.call_count == 0


def test_chart_hit_contract_flagged_sparse_false_maxdist_zero(mock_milvus, manifests):
    eng = _engine(mock_milvus, manifests)
    action, flagged_sparse, max_dist, enc = eng.decide_with_encoding(_btn_open_gs())
    assert action in CANONICAL_ACTIONS
    assert flagged_sparse is False
    assert max_dist == 0.0
    assert enc is not None


def test_chart_disabled_uses_knn(mock_milvus, manifests):
    eng = _engine(mock_milvus, manifests, use_charts=False)
    eng.decide(_btn_open_gs())
    assert mock_milvus.search.call_count == 1  # kNN path used


def test_chart_miss_falls_back_to_knn(mock_milvus, manifests):
    # 72o on UTG is a pure fold (absent from chart) -> miss -> kNN
    gs = GameState(
        street="preflop",
        hero_position="UTG",
        hero_hole_cards=("7c", "2d"),
        board_cards=(),
        pot_size_bb=1.5,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=(),
        opponents_remaining=5,
        prior_street_aggressor=None,
    )
    eng = _engine(mock_milvus, manifests)
    eng.decide(gs)
    assert mock_milvus.search.call_count == 1


def test_postflop_never_calls_chart(mock_milvus, manifests):
    eng = _engine(mock_milvus, manifests)
    h0 = eng._preflop_chart.chart_hits
    eng.decide(_flop_gs())
    assert mock_milvus.search.call_count == 1  # postflop kNN
    assert eng._preflop_chart.chart_hits == h0  # chart untouched


def test_determinism_same_seed_same_action(mock_milvus, manifests):
    a1 = _engine(mock_milvus, manifests, seed=42).decide(_btn_open_gs())
    a2 = _engine(mock_milvus, manifests, seed=42).decide(_btn_open_gs())
    assert a1 == a2


def test_chart_hit_samples_by_frequency_not_argmax(mock_milvus, manifests):
    """A material-fold combo must sample fold at ~chart frequency, not 0%.

    KJo facing a CO open (BTN_vs_CO) = {3bet_3x: 0.667, call: 0.044}; the missing
    0.289 is the implicit fold remainder. The engine must inject that fold mass
    before legalizing — otherwise it renormalizes the played verbs to 1.0 and the
    bottom of every range never folds.
    """
    gs = GameState(
        street="preflop",
        hero_position="BTN",
        hero_hole_cards=("Kh", "Jc"),
        board_cards=(),
        pot_size_bb=3.5,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=2.5,
        hero_bet_size_bb=0.0,
        action_sequence=("UTG:fold", "MP:fold", "CO:open_2_2bb"),
        opponents_remaining=1,
        prior_street_aggressor=None,
    )
    from collections import Counter

    counts = Counter(_engine(mock_milvus, manifests, seed=s).decide(gs) for s in range(200))
    assert "fold" in counts, f"material-fold combo never folded: {dict(counts)}"
    # ~28.9% fold per chart; allow wide tolerance for sampling noise across 200 seeds.
    fold_frac = counts["fold"] / 200
    assert 0.15 < fold_frac < 0.45, f"fold freq {fold_frac:.3f} off chart ~0.289: {dict(counts)}"
