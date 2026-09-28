"""PreflopChartStrategy: load + validate chart tree, derive situation, lookup."""

from __future__ import annotations

import json

import pytest

from src.decision_engine.preflop_chart import PreflopChartStrategy
from src.protocols.game_state import GameState

CHART_DIR = "charts/preflop/inferred_6max"


def _gs(**over) -> GameState:
    base = dict(
        street="preflop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=(),
        pot_size_bb=3.0,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=(),
        opponents_remaining=5,
        prior_street_aggressor=None,
    )
    base.update(over)
    return GameState(**base)


@pytest.fixture(scope="module")
def strategy() -> PreflopChartStrategy:
    return PreflopChartStrategy.from_dir(CHART_DIR)


# --- loader validation -------------------------------------------------------


def test_loads_real_tree(strategy):
    assert strategy.n_charts > 0


def test_reject_non_vocab_action_key(tmp_path):
    sit = tmp_path / "RFI"
    sit.mkdir(parents=True)
    (sit / "BTN.json").write_text(json.dumps({"AA": {"shove_it": 1.0}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"source": "x"}))
    with pytest.raises(ValueError, match=r"(?i)action|vocab"):
        PreflopChartStrategy.from_dir(str(tmp_path))


def test_reject_dist_sum_gt_one(tmp_path):
    sit = tmp_path / "RFI"
    sit.mkdir(parents=True)
    (sit / "BTN.json").write_text(json.dumps({"AA": {"open_2_2bb": 0.7, "call": 0.5}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"source": "x"}))
    with pytest.raises(ValueError, match=r"(?i)sum|dist"):
        PreflopChartStrategy.from_dir(str(tmp_path))


def test_reject_bad_combo_key(tmp_path):
    sit = tmp_path / "RFI"
    sit.mkdir(parents=True)
    (sit / "BTN.json").write_text(json.dumps({"ZZ9": {"open_2_2bb": 1.0}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"source": "x"}))
    with pytest.raises(ValueError, match=r"(?i)combo"):
        PreflopChartStrategy.from_dir(str(tmp_path))


def test_reject_negative_freq(tmp_path):
    # sums to 0.9 (<= 1) but a negative freq must fail at load, not at decide-time
    sit = tmp_path / "RFI"
    sit.mkdir(parents=True)
    (sit / "BTN.json").write_text(json.dumps({"AA": {"3bet_3x": 1.2, "call": -0.3}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"source": "x"}))
    with pytest.raises(ValueError, match=r"(?i)negative"):
        PreflopChartStrategy.from_dir(str(tmp_path))


# --- situation derivation + lookup -------------------------------------------


def test_rfi_lookup_returns_open_dist(strategy):
    gs = _gs(hero_position="BTN", action_sequence=(), opponents_remaining=5)
    hf = {"street_class": "preflop", "pot_type": "limp", "hero_pos_rel": "OOP", "n_players_active": 6}
    dist = strategy.lookup(gs, hf)
    assert dist is not None
    # AhKd -> AKo opens on BTN
    assert "open_2_2bb" in dist


def test_vs_rfi_lookup_opener_seat(strategy):
    # CO opens, hero BTN faces it (srp, IP)
    gs = _gs(
        hero_position="BTN",
        hero_hole_cards=("Ah", "Ks"),
        action_sequence=("UTG:fold", "MP:fold", "CO:open_2_2bb"),
        opponents_remaining=1,
    )
    hf = {"street_class": "preflop", "pot_type": "srp", "hero_pos_rel": "IP", "n_players_active": 2}
    dist = strategy.lookup(gs, hf)
    assert dist is not None
    # AKo 3bets IP vs a CO open
    assert "3bet_3x" in dist


def test_vs_3bet_lookup(strategy):
    # hero CO opens, BTN 3bets -> hero faces 3bet
    gs = _gs(
        hero_position="CO",
        hero_hole_cards=("Ah", "Ad"),
        action_sequence=("CO:open_2_2bb", "BTN:3bet_3x"),
        opponents_remaining=1,
    )
    hf = {"street_class": "preflop", "pot_type": "3bet", "hero_pos_rel": "OOP", "n_players_active": 2}
    dist = strategy.lookup(gs, hf)
    assert dist is not None
    assert "4bet_2_5x" in dist or "call" in dist


def test_multiway_returns_none(strategy):
    # srp but 3 players active -> not a clean HU line
    gs = _gs(
        hero_position="BTN",
        action_sequence=("UTG:open_2_2bb", "MP:call"),
        opponents_remaining=2,
    )
    hf = {"street_class": "preflop", "pot_type": "srp", "hero_pos_rel": "IP", "n_players_active": 3}
    assert strategy.lookup(gs, hf) is None


def test_non_numeric_n_active_degrades_to_fallback(strategy):
    # a non-numeric n_players_active must not crash decide(); it degrades to the HU default
    gs = _gs(
        hero_position="BTN",
        hero_hole_cards=("Ah", "Ks"),
        action_sequence=("UTG:fold", "MP:fold", "CO:open_2_2bb"),
        opponents_remaining=1,
    )
    hf = {"street_class": "preflop", "pot_type": "srp", "hero_pos_rel": "IP", "n_players_active": "foo"}
    dist = strategy.lookup(gs, hf)
    assert dist is None or isinstance(dist, dict)


def test_uncovered_combo_returns_none(strategy):
    # 72o is a pure fold (absent from open chart) -> chart miss -> None
    gs = _gs(hero_position="UTG", hero_hole_cards=("7c", "2d"), action_sequence=(), opponents_remaining=5)
    hf = {"street_class": "preflop", "pot_type": "limp", "hero_pos_rel": "OOP", "n_players_active": 6}
    assert strategy.lookup(gs, hf) is None


def test_missing_chart_returns_none(strategy):
    # vs_4bet for a seat with no 5bet file -> None
    gs = _gs(
        hero_position="MP",
        hero_hole_cards=("3c", "2d"),
        action_sequence=("MP:open_2_2bb", "BTN:3bet_3x", "MP:4bet_2_5x", "BTN:allin"),
        opponents_remaining=1,
    )
    hf = {"street_class": "preflop", "pot_type": "4bet", "hero_pos_rel": "OOP", "n_players_active": 2}
    # may be None (no chart / combo miss); must never raise
    assert strategy.lookup(gs, hf) is None or isinstance(strategy.lookup(gs, hf), dict)


def test_hit_miss_counters(strategy):
    h0, m0 = strategy.chart_hits, strategy.chart_misses
    gs = _gs(hero_position="BTN", hero_hole_cards=("Ah", "Kd"), action_sequence=(), opponents_remaining=5)
    hf = {"street_class": "preflop", "pot_type": "limp", "hero_pos_rel": "OOP", "n_players_active": 6}
    strategy.lookup(gs, hf)  # hit
    miss_gs = _gs(
        hero_position="UTG", hero_hole_cards=("7c", "2d"), action_sequence=(), opponents_remaining=5
    )
    strategy.lookup(miss_gs, hf)  # miss (72o absent)
    assert strategy.chart_hits == h0 + 1
    assert strategy.chart_misses == m0 + 1


def test_rfi_open_mass_not_overfold_artifact(strategy):
    """RFI aggregate non-fold mass per position is GTO-scale, not the ~4% blend artifact."""

    def mass(pos: str) -> float:
        ranks = "AKQJT98765432"
        suits = "cdhs"
        deck = [r + s for r in ranks for s in suits]
        import itertools

        from tools.preflop_combo import combo_169

        seen_combo: dict[str, float] = {}
        nonfold = 0
        total = 0
        for a, b in itertools.combinations(deck, 2):
            gs = _gs(hero_position=pos, hero_hole_cards=(a, b), action_sequence=(), opponents_remaining=5)
            hf = {
                "street_class": "preflop",
                "pot_type": "limp",
                "hero_pos_rel": "OOP",
                "n_players_active": 6,
            }
            cls = combo_169(a, b)
            dist = seen_combo.get(cls)
            if dist is None:
                got = strategy.lookup(gs, hf)
                seen_combo[cls] = sum(got.values()) if got else 0.0
                dist = seen_combo[cls]
            nonfold += dist
            total += 1
        return nonfold / total

    btn = mass("BTN")
    utg = mass("UTG")
    assert 0.35 <= btn <= 0.50, f"BTN open mass {btn}"
    assert 0.13 <= utg <= 0.22, f"UTG open mass {utg}"
    assert utg > 0.10, "UTG must not reproduce the ~4% over-fold blend artifact"
