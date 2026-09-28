"""Preflop feature extractor — 34-dim float32 embedding vector.

Implements FEATURES-v2.md §Preflop Model (v3 amendment).

Dim layout (1-indexed per spec table):
  1-6   Group A: equity deciles (p10/p30/p50/p70/p90) + mean  [6 dims]
  7-11  Group I: hand-class features                          [5 dims]
  12    Group D: hero_pos_rel (OOP=0, IP=1)                   [1 dim]
  13-18 Group D: hero_pos one-hot (BTN/CO/MP/UTG/SB/BB)       [6 dims]
  19    Group D: n_players_active                              [1 dim]
  20    Group D: players_yet_to_act (zeroed — not reconstructable from schema v1.1) [1 dim]
  21    Group E: pot_odds = call/(pot+call)  [0,1) — no clip   [1 dim]
  22    Group E: log-scaled pot_bb                             [1 dim]
  23    Group E: log(1 + eff_stack/pot)                        [1 dim]
  24-26 Group F: num_raises, hero_raised_already, facing_type [3 dims]
  27    Group F: last_bet / pot (capped 5)                     [1 dim]
  28-32 Group J: villain pos 5-bucket one-hot + tightness     [5 dims]
  33    Group K: value-blocker score (premium combos removed)  [1 dim]
  34    Group K: range-blocker score (full range removed)      [1 dim]

  Total = 6 + 5 + 1 + 6 + 2 + 1 + 1 + 1 + 3 + 1 + 5 + 2 = 34
"""

from __future__ import annotations

import math

import numpy as np

from tools.equity_lookup import EquityLookup
from tools.joint_canonicalize import joint_canonicalize

# ---- Constants ---------------------------------------------------------------

_POS_ORDER = ["BTN", "CO", "MP", "UTG", "SB", "BB"]
_POS_IDX = {p: i for i, p in enumerate(_POS_ORDER)}

# Villain position 5-bucket mapping (abbreviated, per spec §J)
# Buckets: BTN | early (UTG) | middle (MP/CO) | blinds (SB/BB) | cold (unknown/NA)
_VILL_BUCKET = {
    "BTN": 0,
    "CO": 2,  # middle
    "MP": 2,  # middle
    "UTG": 1,  # early
    "SB": 3,  # blinds
    "BB": 3,  # blinds
}
_VILL_BUCKET_COUNT = 5  # BTN/early/middle/blinds/cold

# Sklansky-Malmuth hand groups (normalized to [0,1]; lower group = stronger hand)
# Source: SM table 1-8, 8 groups; normalized as (8 - group) / 7
_SM_GROUPS: dict[str, float] = {
    # Group 1 → 1.0
    "AA": 1.0,
    "KK": 1.0,
    "QQ": 1.0,
    "JJ": 1.0,
    "AKs": 1.0,
    # Group 2 → 6/7 ≈ 0.857
    "TT": 6 / 7,
    "AQs": 6 / 7,
    "AJs": 6 / 7,
    "KQs": 6 / 7,
    "AKo": 6 / 7,
    # Group 3 → 5/7 ≈ 0.714
    "99": 5 / 7,
    "JTs": 5 / 7,
    "QJs": 5 / 7,
    "KJs": 5 / 7,
    "ATs": 5 / 7,
    "AQo": 5 / 7,
    # Group 4 → 4/7 ≈ 0.571
    "88": 4 / 7,
    "QTs": 4 / 7,
    "98s": 4 / 7,
    "J9s": 4 / 7,
    "AJo": 4 / 7,
    "KQo": 4 / 7,
    # Group 5 → 3/7 ≈ 0.429
    "77": 3 / 7,
    "T9s": 3 / 7,
    "KTs": 3 / 7,
    "87s": 3 / 7,
    "Q9s": 3 / 7,
    "76s": 3 / 7,
    "97s": 3 / 7,
    "Axs": 3 / 7,
    "65s": 3 / 7,
    "KJo": 3 / 7,
    "QJo": 3 / 7,
    # Group 6 → 2/7 ≈ 0.286
    "66": 2 / 7,
    "55": 2 / 7,
    "86s": 2 / 7,
    "75s": 2 / 7,
    "K9s": 2 / 7,
    "T8s": 2 / 7,
    "64s": 2 / 7,
    "KTo": 2 / 7,
    "QTo": 2 / 7,
    "JTo": 2 / 7,
    "ATo": 2 / 7,
    # Group 7 → 1/7 ≈ 0.143
    "44": 1 / 7,
    "33": 1 / 7,
    "22": 1 / 7,
    "K8s": 1 / 7,
    "K7s": 1 / 7,
    "K6s": 1 / 7,
    "K5s": 1 / 7,
    "K4s": 1 / 7,
    "K3s": 1 / 7,
    "K2s": 1 / 7,
    "J8s": 1 / 7,
    "T7s": 1 / 7,
    "96s": 1 / 7,
    "54s": 1 / 7,
    "74s": 1 / 7,
    "Q8s": 1 / 7,
    "K9o": 1 / 7,
    "J9o": 1 / 7,
    "T9o": 1 / 7,
    # Group 8 → 0
}

RANKS = "23456789TJQKA"
_RANK_VAL = {r: i for i, r in enumerate(RANKS)}  # 2=0, A=12

# ---- Group weights per spec §Feature Group Weights ---------------------------
# NOTE (Phase 2 Decision 1): group weights are applied at upsert time
# in tools/upsert_milvus.py, NOT here. Extractor returns raw min-maxed
# vectors so zscore_fit operates on the unweighted population.
_GROUP_WEIGHTS: dict[str, float] = {
    "A": 1.5,
    "I": 1.2,
    "D": 0.9,
    "E": 1.0,
    "F": 1.0,
    "J": 0.8,
    "K": 1.0,
}

# Dim-layout for upsert_milvus group-weight vector construction.
_PREFLOP_DIM_LAYOUT: list[tuple[str, int]] = [
    ("A", 6),
    ("I", 5),
    ("D", 9),
    ("E", 3),
    ("F", 4),
    ("J", 5),
    ("K", 2),
]

# Sklansky premium classes used for value-blocker computation (Groups 1-2).
_PREMIUM_CLASSES: frozenset[str] = frozenset(
    ["AA", "KK", "QQ", "JJ", "AKs", "TT", "AQs", "AJs", "KQs", "AKo"]
)


# ---- Hand-class helpers -------------------------------------------------------


def _parse_hole_class(hc: str) -> tuple[str, str, bool]:
    """Return (rank1, rank2, suited) from hole class like 'AKs', 'TTo', 'AKo'."""
    if len(hc) == 3:
        r1, r2, suitness = hc[0], hc[1], hc[2]
        suited = suitness == "s"
        return r1, r2, suited
    if len(hc) == 2:
        # Pair: e.g. 'AA'
        return hc[0], hc[1], False
    raise ValueError(f"Unrecognised hole class: {hc!r}")


def _is_pair(r1: str, r2: str) -> bool:
    return r1 == r2


def _gap(r1: str, r2: str) -> int:
    """Connector gap between two cards (0 = connector, 1 = one-gap, ...)."""
    hi = max(_RANK_VAL[r1], _RANK_VAL[r2])
    lo = min(_RANK_VAL[r1], _RANK_VAL[r2])
    return hi - lo - 1


def _pair_rank_normalized(rank: str) -> float:
    """Pair rank normalized to [0,1]. 22=0, AA=1."""
    return _RANK_VAL[rank] / 12.0


def _is_broadway(r1: str, r2: str) -> bool:
    broadway = set("TJQKA")
    return r1 in broadway and r2 in broadway


def _sklansky_normalized(hc: str) -> float:
    """Look up SM group; return normalized score; default 0 for unlisted hands."""
    return _SM_GROUPS.get(hc, 0.0)


# ---- Group extractors ---------------------------------------------------------


def _group_a(dp: dict, equity_lookup: EquityLookup, pop_mean: dict[str, float]) -> np.ndarray:
    """6 dims: p10/p30/p50/p70/p90 + mean from equity table."""
    hole = dp.get("hero_hole")
    board = dp.get("board") or []  # preflop → empty board
    scenario = dp.get("scenario_key", "")

    eq: dict[str, float] | None = None
    if hole and scenario:
        h_can, b_can = joint_canonicalize(hole, board)
        eq = equity_lookup.get(h_can, b_can, scenario)

    if eq is None:
        # Use population mean; log already emitted by EquityLookup
        eq = pop_mean

    dims = np.array([eq["p10"], eq["p30"], eq["p50"], eq["p70"], eq["p90"], eq["mean"]], dtype=np.float32)
    return np.clip(dims, 0.0, 1.0)


def _hole_to_class(hole: list[str]) -> str:
    """Convert raw hole cards (['Ac','Kd']) to 169-class notation ('AKo'/'AKs'/'AA')."""
    if not hole or len(hole) != 2:
        return ""
    c1, c2 = hole[0], hole[1]
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    if _RANK_VAL.get(r2, -1) > _RANK_VAL.get(r1, -1):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def _group_i(dp: dict) -> np.ndarray:
    """5 dims: pair_rank, suited, gap_normalized, is_broadway, sklansky_normalized."""
    hc = dp.get("hero_hole_class", "")
    try:
        r1, r2, suited = _parse_hole_class(hc)
    except ValueError:
        # hero_hole_class may be raw canonical cards ('AcKd') or absent;
        # derive the 169-class label from the raw hole cards instead.
        hc = _hole_to_class(dp.get("hero_hole") or [])
        try:
            r1, r2, suited = _parse_hole_class(hc)
        except ValueError:
            return np.zeros(5, dtype=np.float32)

    if _is_pair(r1, r2):
        pair_rank = _pair_rank_normalized(r1)
        gap_norm = 0.0  # pairs have no gap
    else:
        pair_rank = 0.0
        gap_val = _gap(r1, r2)
        gap_norm = float(np.clip(gap_val / 12.0, 0.0, 1.0))

    dims = np.array(
        [
            pair_rank,
            1.0 if suited else 0.0,
            gap_norm,
            1.0 if _is_broadway(r1, r2) else 0.0,
            _sklansky_normalized(hc),
        ],
        dtype=np.float32,
    )
    return dims


def _group_d(dp: dict) -> np.ndarray:
    """9 dims: hero_pos_rel (1) + hero_pos one-hot (6) + n_players + players_yet_to_act (2)."""
    # Dim 12: OOP=0, IP=1
    pos_rel = 1.0 if dp.get("hero_pos_rel") == "IP" else 0.0

    # Dims 13-18: one-hot BTN/CO/MP/UTG/SB/BB
    hero_pos = dp.get("hero_pos", "")
    pos_onehot = np.zeros(6, dtype=np.float32)
    if hero_pos in _POS_IDX:
        pos_onehot[_POS_IDX[hero_pos]] = 1.0

    # Dims 19-20: n_players_active, players_yet_to_act
    n_players = float(dp.get("n_players_at_street", 2))
    # players_yet_to_act zeroed: preflop action ordering is not reliably reconstructable from schema v1.1
    players_yet = 0.0

    # Normalize: n_players /6, players_yet /6
    dims = np.array([pos_rel], dtype=np.float32)
    dims = np.concatenate([dims, pos_onehot, [n_players / 6.0, players_yet]])
    return dims.astype(np.float32)


def _group_e(dp: dict) -> np.ndarray:
    """3 dims: pot_odds, log-scaled pot_bb, log(1+eff_stack/pot)."""
    # pot_odds = call / (pot + call)
    facing_size = dp.get("facing_size_pot_frac") or 0.0
    pot_bb = dp.get("pot_bb") or (dp.get("pot_cents", 0) / max(dp.get("bb_cents", 10), 1))
    spr = dp.get("spr") or 0.0

    # Reconstruct call amount from facing_size_pot_frac
    # facing_size_pot_frac = bet / pot_when_facing; call_frac/(1+call_frac) is self-bounded [0,1)
    call_frac = float(facing_size) if facing_size else 0.0
    pot_odds = call_frac / (1.0 + call_frac) if call_frac > 0 else 0.0

    # log-scaled pot_bb: log(1 + pot_bb) / log(1 + 200) (200bb cap)
    pot_bb_f = float(pot_bb) if pot_bb else 0.0
    pot_bb_norm = math.log1p(pot_bb_f) / math.log1p(200.0)
    pot_bb_norm = float(np.clip(pot_bb_norm, 0.0, 1.0))

    # log(1 + eff_stack/pot): SPR is already eff_stack/pot
    spr_f = float(spr) if spr else 0.0
    spr_log = math.log1p(spr_f) / math.log1p(100.0)  # cap at SPR=100
    spr_log = float(np.clip(spr_log, 0.0, 1.0))

    return np.array([pot_odds, pot_bb_norm, spr_log], dtype=np.float32)


def _group_f(dp: dict) -> np.ndarray:
    """4 dims: num_raises, hero_raised_already, facing_type (3-class), last_bet/pot."""
    action_so_far = dp.get("action_so_far_street", [])
    facing = dp.get("facing", "cold")

    num_raises = sum(1 for a in action_so_far if a.get("action") in ("raise", "allin"))
    # Normalize: cap 4
    num_raises_norm = float(np.clip(num_raises / 4.0, 0.0, 1.0))

    # hero_raised_already: check preflop_action_seq for hero raises
    hero_pos = dp.get("hero_pos", "")
    pf_seq = dp.get("preflop_action_seq", [])
    hero_raised = any(a.get("pos") == hero_pos and a.get("action") in ("raise", "allin") for a in pf_seq)
    hero_raised_f = 1.0 if hero_raised else 0.0

    # facing_type 3-class: 0=cold/check_to, 1=bet_to, 2=raise_to
    facing_map = {"cold": 0.0, "check_to": 0.0, "bet_to": 0.5, "raise_to": 1.0}
    facing_type = facing_map.get(str(facing), 0.0)

    # last_bet/pot capped at 5, normalized /5
    facing_size_pf = dp.get("facing_size_pot_frac") or 0.0
    last_bet_norm = float(np.clip(float(facing_size_pf) / 5.0, 0.0, 1.0))

    return np.array([num_raises_norm, hero_raised_f, facing_type, last_bet_norm], dtype=np.float32)


# Position-proxy tightness — fallback only, used when no observed palette resolves.
_TIGHTNESS_BY_POS: dict[str, float] = {
    "UTG": 1.0,
    "MP": 0.67,
    "CO": 0.5,
    "BTN": 0.33,
    "SB": 0.5,
    "BB": 0.5,
}

# pot_type → villain's preflop action label prefix (their aggressive action).
# Keys are the labels emitted by tools.position.preflop_pot_type ("5bet+" caps deep spots).
_POT_TYPE_TO_ACTION = {
    "limp": "limp",
    "srp": "open",
    "3bet": "3bet",
    "4bet": "4bet",
    "5bet+": "5bet",
}


def _group_j_preflop(dp: dict, tightness_lookup: dict[str, float] | None = None) -> np.ndarray:
    """5 dims: villain_pos 4-bucket one-hot (BTN/early/middle/blinds) + tightness scalar.

    Tightness is the strength-weighted concentration of the villain's OBSERVED
    range at (pos, action) when a palette resolves (FEATURES-v2 §J intent), else
    a position proxy. Looks up "{pos}/{action}" where action derives from pot_type.
    """
    # Identify villain position — use facing_pos first, fall back to preflop_aggressor
    villain_pos = dp.get("facing_pos") or dp.get("preflop_aggressor")

    # 4 one-hot slots: 0=BTN, 1=early(UTG), 2=middle(MP/CO), 3=blinds(SB/BB)
    # all-zero = cold/unknown (5th bucket per spec is implicit all-zero)
    bucket_onehot = np.zeros(4, dtype=np.float32)

    if villain_pos and villain_pos in _VILL_BUCKET:
        bucket = _VILL_BUCKET[villain_pos]
        if bucket < 4:
            bucket_onehot[bucket] = 1.0
        # bucket >= 4 (cold) → leave all-zero

    tightness: float | None = None
    if tightness_lookup and villain_pos:
        action = _POT_TYPE_TO_ACTION.get(str(dp.get("pot_type", "")))
        if action:
            tightness = tightness_lookup.get(f"{villain_pos}/{action}")
    if tightness is None:
        tightness = _TIGHTNESS_BY_POS.get(str(villain_pos), 0.5)

    return np.array([*bucket_onehot, tightness], dtype=np.float32)


def _group_k_blockers(dp: dict, range_lookup: dict[str, dict[str, float]] | None = None) -> np.ndarray:
    """2 dims: value-blocker score, range-blocker score.

    Computes card-removal effect of hero's hole cards on the villain's range:
      dim 33 — value-blocker: fraction of villain's premium combos (top SM groups)
               removed by hero's two cards, weighted by villain hand strength.
      dim 34 — range-blocker: total weighted fraction of villain's full range removed.

    Fallback 0.0 when no range resolves (no blocker info).
    """
    hero_hole = dp.get("hero_hole") or []
    if not hero_hole or len(hero_hole) < 2:
        return np.zeros(2, dtype=np.float32)

    villain_pos = dp.get("facing_pos") or dp.get("preflop_aggressor")
    if not villain_pos or not range_lookup:
        return np.zeros(2, dtype=np.float32)

    action = _POT_TYPE_TO_ACTION.get(str(dp.get("pot_type", "")))
    if not action:
        return np.zeros(2, dtype=np.float32)

    villain_range = range_lookup.get(f"{villain_pos}/{action}")
    if not villain_range:
        return np.zeros(2, dtype=np.float32)

    hero_combos = set(hero_hole)

    total_w = 0.0
    removed_w = 0.0
    premium_total = 0.0
    premium_removed = 0.0

    for cls, w in villain_range.items():
        wf = float(w)
        if wf <= 0:
            continue
        strength = _SM_GROUPS.get(cls, 0.0)
        blocked_frac = _combo_collision_survivor(cls, hero_combos)
        total_w += wf
        removed_w += wf * blocked_frac
        if cls in _PREMIUM_CLASSES:
            premium_total += wf * strength
            premium_removed += wf * strength * blocked_frac

    value_blocker = (premium_removed / premium_total) if premium_total > 0 else 0.0
    range_blocker = (removed_w / total_w) if total_w > 0 else 0.0

    return np.array(
        [float(np.clip(value_blocker, 0.0, 1.0)), float(np.clip(range_blocker, 0.0, 1.0))], dtype=np.float32
    )


_SUITS = "cdhs"


def _all_combos_for_class(cls: str) -> list[tuple[str, str]]:
    """All possible villain two-card combos for a canonical hand class."""
    if len(cls) == 2:
        r = cls[0]
        cards = [r + s for s in _SUITS]
        from itertools import combinations

        return list(combinations(cards, 2))
    r1, r2, suitness = cls[0], cls[1], cls[2]
    if _RANK_VAL.get(r2, 0) > _RANK_VAL.get(r1, 0):
        r1, r2 = r2, r1
    if suitness == "s":
        return [(r1 + s, r2 + s) for s in _SUITS]
    return [(r1 + s1, r2 + s2) for s1 in _SUITS for s2 in _SUITS if s1 != s2]


def _combo_collision_survivor(cls: str, hero_combos: set[str]) -> float:
    """Fraction of combos for cls that survive (are NOT blocked) by hero's cards.

    Returns value in [0,1]: fraction removed (blocked). 0.0 = nothing removed.
    """
    all_combos = _all_combos_for_class(cls)
    if not all_combos:
        return 0.0
    n_blocked = sum(1 for c in all_combos if bool(set(c) & hero_combos))
    return n_blocked / len(all_combos)


# ---- Main extractor ----------------------------------------------------------


def extract_preflop(
    dp: dict,
    equity_lookup: EquityLookup,
    pop_mean: dict[str, float] | None = None,
    tightness_lookup: dict[str, float] | None = None,
    range_lookup: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, dict]:
    """Extract 34-dim preflop embedding vector for a decision point.

    Args:
        dp: Decision point dict (DECISIONS-SCHEMA v1.1)
        equity_lookup: Loaded EquityLookup instance
        pop_mean: Population-mean equity dict for miss fallback (auto-computed if None)
        tightness_lookup: {f"{pos}/{action}": tightness} from villain_range_stats
        range_lookup: {f"{pos}/{action}": combo_counts} for blocker computation

    Returns:
        (vector, filter_dict) where vector is float32 ndarray of shape (34,)
        and filter_dict has keys: street_class, pot_type, hero_pos_rel, n_players_active
    """
    if pop_mean is None:
        pop_mean = equity_lookup.population_mean_equity()

    # Extract each group
    a = _group_a(dp, equity_lookup, pop_mean)  # 6 dims
    i = _group_i(dp)  # 5 dims
    d = _group_d(dp)  # 9 dims
    e = _group_e(dp)  # 3 dims
    f = _group_f(dp)  # 4 dims
    j = _group_j_preflop(dp, tightness_lookup)  # 5 dims
    k = _group_k_blockers(dp, range_lookup)  # 2 dims

    vector = np.concatenate([a, i, d, e, f, j, k], dtype=np.float32)

    # Clip to [0, 1]
    vector = np.clip(vector, 0.0, 1.0)

    assert len(vector) == 34, f"Preflop vector has {len(vector)} dims, expected 34"

    filter_dict = {
        "street_class": "preflop",
        "pot_type": dp.get("pot_type", ""),
        "hero_pos_rel": dp.get("hero_pos_rel", ""),
        "n_players_active": int(dp.get("n_players_at_street", 2)),
    }

    return vector, filter_dict
