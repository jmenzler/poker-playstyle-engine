"""Distance properties of the position encoders in tools/position_encoding_eval.
One-hot makes every hero-position change equidistant (can't express MP between UTG
and BTN). These tests pin the RED (one-hot equidistance) and each encoding's GREEN.
"""

from __future__ import annotations

import numpy as np

from tools.position_encoding_eval import (
    OPEN_ORDER,
    pos_block_baseline,
    pos_block_ordinal,
    pos_block_range_character,
    pos_block_relationship,
    pos_block_relationship_v2,
)


def _dp(hero: str, vill: str, *, hero_rel: str = "IP", pf_agg: str | None = None) -> dict:
    """Minimal decision row carrying only the fields the position blocks read.

    pf_agg defaults to hero so villain_was_pf_agg is 0 for all spots in a triplet
    (neutralizes that dim when isolating the position axis).
    """
    return {
        "hero_pos": hero,
        "facing_pos": vill,
        "preflop_aggressor": pf_agg if pf_agg is not None else hero,
        "hero_pos_rel": hero_rel,
        "n_players_at_street": 2,
        "pot_type": "srp",
    }


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---- Dim counts --------------------------------------------------------------


def test_block_dim_counts():
    assert pos_block_baseline(_dp("MP", "CO")).shape == (18,)
    assert pos_block_ordinal(_dp("MP", "CO"), OPEN_ORDER).shape == (10,)
    assert pos_block_relationship(_dp("MP", "CO")).shape == (8,)


# ---- RED: one-hot makes hero positions equidistant ---------------------------


def test_baseline_onehot_hero_equidistant():
    """Villain fixed=CO; UTG/MP/BTN heroes are mutually equidistant on the hero one-hot slice."""
    utg = pos_block_baseline(_dp("UTG", "CO"))[:6]  # hero one-hot = first 6 dims
    mp = pos_block_baseline(_dp("MP", "CO"))[:6]
    btn = pos_block_baseline(_dp("BTN", "CO"))[:6]
    d_um = _euclid(utg, mp)
    d_mb = _euclid(mp, btn)
    d_ub = _euclid(utg, btn)
    assert np.isclose(d_um, d_mb) and np.isclose(d_mb, d_ub)
    assert d_um > 0  # they ARE different positions, just equidistant — the defect


# ---- GREEN: ordinal ranks hero adjacency -------------------------------------


def test_ordinal_openorder_mp_between_utg_btn():
    """Villain fixed=CO; MP closer to UTG than to BTN on the open-order axis."""
    utg = pos_block_ordinal(_dp("UTG", "CO"), OPEN_ORDER)
    mp = pos_block_ordinal(_dp("MP", "CO"), OPEN_ORDER)
    btn = pos_block_ordinal(_dp("BTN", "CO"), OPEN_ORDER)
    assert _euclid(mp, utg) < _euclid(mp, btn)


def test_ordinal_dataderived_respects_injected_ranking():
    """The ordinal scalar must follow the injected seat_value, not a baked-in axis."""
    forward = {"UTG": 0.0, "MP": 0.2, "CO": 0.4, "BTN": 0.6, "SB": 0.8, "BB": 1.0}
    reverse = {"UTG": 1.0, "MP": 0.8, "CO": 0.6, "BTN": 0.4, "SB": 0.2, "BB": 0.0}
    # Forward: MP nearer UTG. Reverse: MP nearer BTN (axis flipped).
    f_utg = pos_block_ordinal(_dp("UTG", "CO"), forward)
    f_mp = pos_block_ordinal(_dp("MP", "CO"), forward)
    f_btn = pos_block_ordinal(_dp("BTN", "CO"), forward)
    assert _euclid(f_mp, f_utg) < _euclid(f_mp, f_btn)
    r_mp = pos_block_ordinal(_dp("MP", "CO"), reverse)[0]
    f_mp_scalar = f_mp[0]
    assert not np.isclose(r_mp, f_mp_scalar)  # scalar actually changed with ranking


# ---- GREEN: relationship collapses similar pair dynamics ---------------------


def test_relationship_pair_dynamics():
    """MP-vs-CO sits near UTG-vs-CO and far from BTN-vs-BB (the design intuition)."""
    a = pos_block_relationship(_dp("MP", "CO"))  # early hero, CO villain, small gap
    b = pos_block_relationship(_dp("UTG", "CO"))  # early hero, CO villain, small gap
    c = pos_block_relationship(_dp("BTN", "BB"))  # button + blind villain + IP
    assert _euclid(a, b) < _euclid(a, c)
    assert _euclid(b, a) < _euclid(b, c)


# ---- Data-driven variants: measured range character handles blinds ----------


def _char(by_pos: dict) -> tuple:
    """Build a char tuple ({}, by_pos, global) for injection."""
    return ({}, {p: np.array(v) for p, v in by_pos.items()}, np.array([0.3, 0.3, 0.4]))


def test_range_character_distinguishes_sb_from_bb():
    """The whole point of the blind fix: measured character separates SB (polar) from BB (wide/passive)."""
    char = _char(
        {
            "BB": [0.10, 0.05, 0.85],  # wide, rarely folds, very passive
            "SB": [0.40, 0.40, 0.20],  # polar: aggresses or folds
            "CO": [0.50, 0.10, 0.40],
        }
    )
    bb = pos_block_range_character(_dp("BB", "CO"), char)
    sb = pos_block_range_character(_dp("SB", "CO"), char)
    assert not np.allclose(bb, sb)  # baseline/relationship collapse these to 0.5 — this must not
    assert bb[1] != sb[1] and bb[2] != sb[2]  # hero agg + fold dims differ


def test_relationship_v2_range_advantage_follows_data():
    """range_adv (dim 5) reflects measured strength gap, not a hand-picked tightness map."""
    char = _char(
        {
            "UTG": [0.20, 0.50, 0.30],  # weak/cappy as measured
            "BTN": [0.60, 0.10, 0.30],  # strong/aggressive
            "CO": [0.40, 0.20, 0.40],
        }
    )
    strong_villain = pos_block_relationship_v2(_dp("UTG", "BTN"), char)[5]
    weak_villain = pos_block_relationship_v2(_dp("BTN", "UTG"), char)[5]
    assert strong_villain > weak_villain


def test_relationship_drops_absolute_seat():
    """Same pair geometry at different absolute seats encodes identically (seat dropped)."""
    # CO-vs-BTN and MP-vs-CO: both IP-side, 1 seat apart, no blinds. Relationship
    # should treat their non-range dims (is_ip, seats_between, blind flags) the same.
    p1 = pos_block_relationship(_dp("MP", "CO"))
    p2 = pos_block_relationship(_dp("UTG", "MP"))
    # seats_between identical (gap 1), blind flags identical, is_ip identical
    assert np.isclose(p1[1], p2[1])  # seats_between
    assert np.isclose(p1[2], p2[2]) and np.isclose(p1[3], p2[3])  # blind flags
