"""Postflop feature extractor — 80-dim float32 embedding vector.

Implements FEATURES-v2.md §Postflop Model exactly.

Dim layout (1-indexed per spec table):
  1-12  Group A: equity features (9 deciles + variance + mean + p90)    [12 dims]
  13-20 Group B: draw features                                           [8 dims]
  21-26 Group C: made-hand category                                      [6 dims]
  27-34 Group D: position + players                                      [8 dims]
  35-41 Group E: pot geometry                                            [7 dims]
  42-57 Group F: betting history current street                          [16 dims]
  58-63 Group G: betting history prior streets                           [6 dims]
  64-70 Group H: board texture                                           [7 dims]
  71-80 Group J: villain context                                         [10 dims]

  Total = 12+8+6+8+7+16+6+7+10 = 80
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np

from tools.equity_lookup import EquityLookup
from tools.joint_canonicalize import joint_canonicalize

# ---- Constants ---------------------------------------------------------------

_POS_ORDER = ["BTN", "CO", "MP", "UTG", "SB", "BB"]
_POS_IDX = {p: i for i, p in enumerate(_POS_ORDER)}

RANKS = "23456789TJQKA"
_RANK_VAL = {r: i for i, r in enumerate(RANKS)}  # 2=0, A=12
SUITS = "cdhs"

# Hand rank categories (0=high card, ..., 8=straight flush)
_HAND_RANK = {
    "high_card": 0,
    "one_pair": 1,
    "two_pair": 2,
    "three_of_a_kind": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "four_of_a_kind": 7,
    "straight_flush": 8,
}

# NOTE (Phase 2 Decision 1): group weights are applied at upsert time
# in tools/upsert_milvus.py, NOT here. Extractor returns raw min-maxed
# vectors so zscore_fit operates on the unweighted population.
# Group weights per spec §Feature Group Weights
_GROUP_WEIGHTS: dict[str, float] = {
    "A": 1.2,
    "B": 1.2,
    "C": 1.2,
    "D": 0.9,
    "E": 1.8,
    "F": 1.8,
    "G": 0.8,
    "H": 0.7,
    "J": 0.8,
}


# ---- Pure-Python 5/7-card hand evaluator (minimal) ---------------------------


def _rank_val(card: str) -> int:
    return _RANK_VAL[card[0]]


def _suit(card: str) -> str:
    return card[1]


def _evaluate_5(cards: list[str]) -> tuple[int, list[int]]:
    """Evaluate a 5-card hand. Returns (hand_rank_int, tiebreakers).

    hand_rank_int: 0=high_card, 1=one_pair, ..., 8=straight_flush
    """
    ranks = sorted([_rank_val(c) for c in cards], reverse=True)
    suits = [_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # Straight check (including A-2-3-4-5 wheel)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[-1] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Wheel: A-2-3-4-5 → ranks [12,3,2,1,0] → sorted unique [12,3,2,1,0]
        elif unique_ranks == [12, 3, 2, 1, 0]:
            is_straight = True
            straight_high = 3  # five-high straight

    # Count rank frequencies
    freq = Counter(ranks)
    counts = sorted(freq.values(), reverse=True)  # e.g. [3,1,1] for trips
    count_ranks = sorted(freq.keys(), key=lambda r: (freq[r], r), reverse=True)

    if is_straight and is_flush:
        return 8, [straight_high]
    if counts[0] == 4:
        quad_rank = count_ranks[0]
        kicker = count_ranks[1]
        return 7, [quad_rank, kicker]
    if counts[0] == 3 and counts[1] == 2:
        return 6, [count_ranks[0], count_ranks[1]]
    if is_flush:
        return 5, ranks
    if is_straight:
        return 4, [straight_high]
    if counts[0] == 3:
        return 3, [count_ranks[0], *sorted([r for r in ranks if freq[r] == 1], reverse=True)]
    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in freq.items() if c == 2], reverse=True)
        kicker = max(r for r, c in freq.items() if c == 1)
        return 2, [*pairs, kicker]
    if counts[0] == 2:
        pair_rank = count_ranks[0]
        kickers = sorted([r for r in ranks if freq[r] == 1], reverse=True)
        return 1, [pair_rank, *kickers]
    return 0, ranks


def _best_5_from_7(hole: list[str], board: list[str]) -> tuple[int, list[int], list[str]]:
    """Find best 5-card hand from hole + board (up to 7 cards total)."""
    all_cards = hole + board
    best_rank = -1
    best_tb: list[int] = []
    best_cards: list[str] = []
    for combo in combinations(all_cards, 5):
        hr, tb = _evaluate_5(list(combo))
        if (hr, tb) > (best_rank, best_tb):
            best_rank = hr
            best_tb = tb
            best_cards = list(combo)
    return best_rank, best_tb, best_cards


def _evaluate_hand(hole: list[str], board: list[str]) -> dict:
    """Return hand evaluation dict for Group C features."""
    if len(board) < 3:
        return {
            "rank_category": 0,
            "pair_rank": 0,
            "kicker": 0,
            "is_set": 0,
            "top_pair": 0,
            "overpair": 0,
        }

    hand_rank, tb, _best_cards = _best_5_from_7(hole, board)

    # Board ranks (sorted desc)
    board_ranks = sorted([_rank_val(c) for c in board], reverse=True)
    hole_ranks = [_rank_val(c) for c in hole]

    board_rank_counter = Counter(_rank_val(c) for c in board)
    hole_rank_counter = Counter(hole_ranks)

    pair_rank = 0
    kicker = 0
    is_set = 0
    top_pair = 0
    overpair = 0

    if hand_rank == 1:  # one pair
        pair_r = tb[0]
        pair_rank = pair_r
        kicker = tb[1] if len(tb) > 1 else 0
        # top_pair: pair matches top board card
        if board_ranks and pair_r == board_ranks[0]:
            top_pair = 1
        # overpair: pocket pair higher than all board cards
        if _is_pair(hole) and hole_ranks[0] > max(board_ranks):
            overpair = 1
        # is_set: hole pair + matching board card
        if hole_rank_counter[pair_r] == 2 and board_rank_counter[pair_r] >= 1:
            is_set = 1

    elif hand_rank == 3:  # three of a kind
        trip_rank = tb[0]
        pair_rank = trip_rank
        # is_set: hold two of the three
        if hole_rank_counter[trip_rank] == 2:
            is_set = 1
        # top_pair proxy for trips
        if board_ranks and trip_rank == board_ranks[0]:
            top_pair = 1

    elif hand_rank == 6:  # full house
        pair_rank = tb[0]
        if hole_rank_counter[pair_rank] == 2:
            is_set = 1

    return {
        "rank_category": hand_rank,
        "pair_rank": pair_rank,
        "kicker": kicker,
        "is_set": is_set,
        "top_pair": top_pair,
        "overpair": overpair,
    }


def _is_pair(hole: list[str]) -> bool:
    return len(hole) == 2 and _rank_val(hole[0]) == _rank_val(hole[1])


# ---- Draw analyzer -----------------------------------------------------------


def _analyze_draws(hole: list[str], board: list[str], street: str) -> dict:
    """Pure-Python draw feature extractor for Group B.

    Returns draw flags + ppot/npot approximations.
    """
    if street == "river" or len(board) < 3:
        # River: no draws
        return {
            "flush_draw": 0,
            "bdfd": 0,
            "oesd": 0,
            "gutshot": 0,
            "combo_draw": 0,
            "ppot": 0.0,
            "npot": 0.0,
            "outs": 0,
        }

    hole_suits = [_suit(c) for c in hole]
    board_suits = [_suit(c) for c in board]
    hole_ranks = sorted([_rank_val(c) for c in hole], reverse=True)
    board_ranks = sorted([_rank_val(c) for c in board], reverse=True)
    all_ranks = hole_ranks + board_ranks

    # ---- Flush draw
    flush_draw = 0
    bdfd = 0
    for s in SUITS:
        hs = hole_suits.count(s)
        bs = board_suits.count(s)
        total = hs + bs
        if total >= 4 and hs >= 1 and bs < 5:
            flush_draw = 1
        if hs == 2 and bs == 1 and len(board) == 3:
            bdfd = 1

    # ---- Straight draws (using all available cards hole+board)
    rank_set = set(all_ranks)

    def _count_in_window(window_ranks: list[int]) -> int:
        return sum(1 for r in window_ranks if r in rank_set)

    oesd = 0
    gutshot = 0
    # 5-card windows; lo=8 → T-J-Q-K-A is the highest valid straight (A=12).
    for lo in range(0, 9):
        window = list(range(lo, lo + 5))
        present = _count_in_window(window)
        if present == 4 and len([r for r in window if r not in rank_set]) == 1:
            gutshot = 1

    # Open-ended needs 4 consecutive present ranks with an empty, in-range slot
    # on BOTH ends — at the deck edge (e.g. J-Q-K-A) one end is missing → gutshot.
    for r in range(0, 10):
        run = {r, r + 1, r + 2, r + 3}
        if run <= rank_set:
            low_open = r - 1 >= 0 and (r - 1) not in rank_set
            high_open = r + 4 <= 12 and (r + 4) not in rank_set
            if low_open and high_open:
                oesd = 1
                gutshot = 0

    # Ace-low wheel draws: A-2-3-4 (needs 5) or A-2-3-5 (needs 4) — both one-way → gutshot.
    if not oesd and ({12, 0, 1, 2} <= rank_set or {12, 0, 1, 3} <= rank_set):
        gutshot = 1

    combo_draw = 1 if (flush_draw and (oesd or gutshot)) else 0

    # ---- Outs count (simplified)
    outs = 0
    if flush_draw:
        outs += 9
    if oesd:
        outs += 8
    elif gutshot:
        outs += 4
    if combo_draw:
        # Subtract overlap
        outs = min(outs, 20)

    # ---- ppot / npot (simplified approximation)
    # Use outs / remaining cards as proxy
    remaining = 52 - len(hole) - len(board)
    ppot = min(outs / max(remaining, 1), 1.0)
    npot = 0.0  # hard to compute without full simulation; leave as 0

    return {
        "flush_draw": flush_draw,
        "bdfd": bdfd,
        "oesd": oesd,
        "gutshot": gutshot,
        "combo_draw": combo_draw,
        "ppot": ppot,
        "npot": npot,
        "outs": outs,
    }


# ---- Board texture analyzer --------------------------------------------------


def _analyze_board(board: list[str]) -> dict:
    """Board texture features for Group H."""
    if len(board) < 3:
        return {
            "flush_possible": 0,
            "flush_made": 0,
            "straight_possible": 0,
            "paired": 0,
            "monotone": 0,
            "connectedness": 0.0,
            "high_card_rank": 0.0,
        }

    suits = [_suit(c) for c in board]
    ranks = [_rank_val(c) for c in board]
    suit_counter = Counter(suits)
    rank_counter = Counter(ranks)

    flush_possible = 1 if max(suit_counter.values()) >= 3 else 0
    flush_made = 1 if max(suit_counter.values()) >= 4 else 0
    monotone = 1 if len(set(suits)) == 1 else 0
    paired = 1 if max(rank_counter.values()) >= 2 else 0

    # Straight possible: any 3 of 5 consecutive ranks
    rank_set = set(ranks)
    straight_possible = 0
    for lo in range(0, 10):
        window = set(range(lo, lo + 5))
        if len(rank_set & window) >= 3:
            straight_possible = 1
            break

    # Connectedness: max run of 3 consecutive ranks / 13
    connectedness = 0.0
    for lo in range(0, 11):
        window = set(range(lo, lo + 3))
        if rank_set >= window:
            connectedness = 3.0 / 13.0
            break

    high_card_rank = float(max(ranks) / 12.0)

    return {
        "flush_possible": flush_possible,
        "flush_made": flush_made,
        "straight_possible": straight_possible,
        "paired": paired,
        "monotone": monotone,
        "connectedness": connectedness,
        "high_card_rank": high_card_rank,
    }


# ---- Group extractors ---------------------------------------------------------


def _group_a_postflop(
    dp: dict, equity_lookup: EquityLookup, pop_mean: dict[str, float], street: str
) -> np.ndarray:
    """12 dims: 9 deciles + variance + mean + p90 (best-case)."""
    hole = dp.get("hero_hole") or []
    board = dp.get("board") or []
    scenario = dp.get("scenario_key", "")

    eq: dict[str, float] | None = None
    if hole and board and scenario:
        h_can, b_can = joint_canonicalize(hole, board)
        eq = equity_lookup.get(h_can, b_can, scenario)

    if eq is None:
        eq = pop_mean

    # River: all 9 deciles = mean equity (single equity value per spec)
    if street == "river":
        single = float(eq["mean"])
        deciles = [single] * 9
    else:
        deciles = [eq[f"p{d}"] for d in [10, 20, 30, 40, 50, 60, 70, 80, 90]]

    variance = float(eq["variance"])
    mean_eq = float(eq["mean"])
    best_case = float(eq["p90"])

    dims = np.array([*deciles, variance, mean_eq, best_case], dtype=np.float32)
    dims = np.clip(dims, 0.0, 1.0)

    # Variance cap: 0.25 per spec → normalize /0.25
    dims[9] = float(np.clip(dims[9] / 0.25, 0.0, 1.0))

    return dims


def _group_b(dp: dict, street: str) -> np.ndarray:
    """8 dims: draw features."""
    hole = list(dp.get("hero_hole") or [])
    board = list(dp.get("board") or [])
    draws = _analyze_draws(hole, board, street)

    outs_norm = float(np.clip(draws["outs"] / 20.0, 0.0, 1.0))
    dims = np.array(
        [
            float(draws["flush_draw"]),
            float(draws["bdfd"]),
            float(draws["oesd"]),
            float(draws["gutshot"]),
            float(draws["combo_draw"]),
            float(np.clip(draws["ppot"], 0.0, 1.0)),
            float(np.clip(draws["npot"], 0.0, 1.0)),
            outs_norm,
        ],
        dtype=np.float32,
    )
    return dims


def _group_c(dp: dict) -> np.ndarray:
    """6 dims: made-hand category."""
    hole = list(dp.get("hero_hole") or [])
    board = list(dp.get("board") or [])
    hand = _evaluate_hand(hole, board)

    rank_cat_norm = float(hand["rank_category"]) / 8.0
    pair_rank_norm = float(hand["pair_rank"]) / 12.0
    kicker_norm = float(hand["kicker"]) / 12.0

    dims = np.array(
        [
            rank_cat_norm,
            pair_rank_norm,
            kicker_norm,
            float(hand["is_set"]),
            float(hand["top_pair"]),
            float(hand["overpair"]),
        ],
        dtype=np.float32,
    )
    return dims


def _group_d_postflop(dp: dict) -> np.ndarray:
    """8 dims: hero_pos one-hot (6) + is_ip (1) + n_players (1)."""
    hero_pos = dp.get("hero_pos", "")
    pos_onehot = np.zeros(6, dtype=np.float32)
    if hero_pos in _POS_IDX:
        pos_onehot[_POS_IDX[hero_pos]] = 1.0

    is_ip = 1.0 if dp.get("hero_pos_rel") == "IP" else 0.0
    n_players = float(np.clip(dp.get("n_players_at_street", 2) / 6.0, 0.0, 1.0))

    return np.concatenate([pos_onehot, [is_ip, n_players]], dtype=np.float32)


def _group_e_postflop(dp: dict) -> np.ndarray:
    """7 dims: pot_odds, log_spr, commitment, pot_bb, facing_bet_frac, facing_flag, prev_agg."""
    facing_size_pf = float(dp.get("facing_size_pot_frac") or 0.0)
    call_frac = facing_size_pf
    pot_odds = call_frac / (1.0 + call_frac) if call_frac > 0 else 0.0
    pot_odds = float(np.clip(pot_odds, 0.0, 0.5))

    spr = float(dp.get("spr") or 0.0)
    log_spr = float(np.clip(math.log1p(spr) / math.log1p(100.0), 0.0, 1.0))

    # hero_commitment_fraction: hero invested in hand / starting_stack
    # Approximate: hero_action_to_bb / (hero_stack_bb + hero_action_to_bb)
    hero_action_to_bb = float(dp.get("hero_action_to_bb") or dp.get("hero_action_to_cents", 0))
    hero_stack_bb = float(
        dp.get("hero_stack_bb") or (dp.get("hero_stack_cents", 0) / max(dp.get("bb_cents", 10), 1))
    )
    total = hero_stack_bb + hero_action_to_bb
    commitment = float(np.clip(hero_action_to_bb / total if total > 0 else 0.0, 0.0, 1.0))

    pot_bb = float(dp.get("pot_bb") or (dp.get("pot_cents", 0) / max(dp.get("bb_cents", 10), 1)))
    pot_bb_norm = float(np.clip(math.log1p(pot_bb) / math.log1p(200.0), 0.0, 1.0))

    facing_bet_frac = float(np.clip(facing_size_pf / 5.0, 0.0, 1.0))
    facing_flag = 1.0 if facing_size_pf > 0 else 0.0

    # prev_street_aggressor_was_hero
    prev_agg = 0.0
    hero_pos = dp.get("hero_pos", "")
    preflop_aggressor = dp.get("preflop_aggressor", "")
    # Use preflop_aggressor as proxy for previous street aggressor (best available)
    if hero_pos and preflop_aggressor == hero_pos:
        prev_agg = 1.0

    return np.array(
        [pot_odds, log_spr, commitment, pot_bb_norm, facing_bet_frac, facing_flag, prev_agg], dtype=np.float32
    )


def _group_f_postflop(dp: dict, street: str) -> np.ndarray:
    """16 dims: street_index, num_raises, hero_raised, facing_size_bucket(8), total_invested,
    action_sequence_compact(3), check_raise_situation."""
    # street_index: flop=0, turn=0.5, river=1.0
    street_map = {"flop": 0.0, "turn": 0.5, "river": 1.0}
    street_idx = street_map.get(street, 0.0)

    action_so_far = dp.get("action_so_far_street", []) or []
    facing_size_pf = float(dp.get("facing_size_pot_frac") or 0.0)
    facing = dp.get("facing", "check_to")
    hero_pos = dp.get("hero_pos", "")

    num_raises = sum(1 for a in action_so_far if a.get("action") in ("raise", "allin"))
    num_raises_norm = float(np.clip(num_raises / 4.0, 0.0, 1.0))

    hero_raised = any(
        a.get("pos") == hero_pos and a.get("action") in ("bet", "raise", "allin") for a in action_so_far
    )
    hero_raised_f = 1.0 if hero_raised else 0.0

    # facing_bet_size_bucket one-hot (8): 0/no-bet, 0.25x, 0.33x, 0.5x, 0.66x, 0.75x, 1x, 1.5x+
    bucket_bounds = [0.0, 0.125, 0.29, 0.41, 0.58, 0.705, 0.875, 1.25]
    bet_bucket = np.zeros(8, dtype=np.float32)
    if facing_size_pf <= 0 or facing == "check_to":
        bet_bucket[0] = 1.0
    else:
        b = 7  # default to last bucket (1.5x+)
        for k, bound in enumerate(bucket_bounds[1:], start=1):
            if facing_size_pf <= bound:
                b = k
                break
        bet_bucket[b] = 1.0

    # total_hero_invested_this_street / pot — sum hero's own bet/raise/call sizes
    # this street (action_so_far_street size_bb), NOT the villain bet hero faces.
    hero_invested_bb = sum(
        float(a.get("size_bb") or 0.0)
        for a in action_so_far
        if a.get("pos") == hero_pos and a.get("action") in ("bet", "raise", "call", "allin")
    )
    pot_bb = float(dp.get("pot_bb") or 0.0)
    total_invested = hero_invested_bb / pot_bb if pot_bb > 0 else 0.0
    total_invested_norm = float(np.clip(total_invested, 0.0, 1.0))

    # action_sequence_compact: hero_checks, hero_bets, villain_aggressive
    hero_checks = sum(1 for a in action_so_far if a.get("pos") == hero_pos and a.get("action") == "check")
    hero_bets = sum(
        1 for a in action_so_far if a.get("pos") == hero_pos and a.get("action") in ("bet", "raise")
    )
    villain_agg = sum(
        1 for a in action_so_far if a.get("pos") != hero_pos and a.get("action") in ("bet", "raise", "allin")
    )

    checks_norm = float(np.clip(hero_checks / 3.0, 0.0, 1.0))
    bets_norm = float(np.clip(hero_bets / 3.0, 0.0, 1.0))
    vagg_norm = float(np.clip(villain_agg / 3.0, 0.0, 1.0))

    # check_raise_situation: villain bet, hero had checked before
    check_raise = 0.0
    if hero_checks > 0 and villain_agg > 0 and facing_size_pf > 0:
        check_raise = 1.0

    dims = np.array(
        [
            street_idx,
            num_raises_norm,
            hero_raised_f,
            *bet_bucket,
            total_invested_norm,
            checks_norm,
            bets_norm,
            vagg_norm,
            check_raise,
        ],
        dtype=np.float32,
    )
    assert len(dims) == 16, f"Group F has {len(dims)} dims, expected 16"
    return dims


def _group_g(dp: dict, street: str) -> np.ndarray:
    """6 dims: preflop betting history context."""
    hero_pos = dp.get("hero_pos", "")
    preflop_aggressor = dp.get("preflop_aggressor", "")

    pf_agg_was_hero = 1.0 if (hero_pos and preflop_aggressor == hero_pos) else 0.0

    pf_seq = dp.get("preflop_action_seq", []) or []
    pf_raises = sum(1 for a in pf_seq if a.get("action") in ("raise", "allin"))
    pf_raises_norm = float(np.clip(pf_raises / 4.0, 0.0, 1.0))

    # prev_street_raises: approximate from action_so_far (for postflop we don't have prev street info)
    # Not available directly — use 0 as fallback
    prev_street_raises = 0.0
    prev_street_hero_invested = 0.0

    # double_barrel: hero bet two prior streets
    # Only possible on river: cannot determine without full street history — approximate 0
    double_barrel = 0.0

    # turn_was_check_back: no way to determine without prior street history — approximate 0
    turn_check_back = 0.0

    return np.array(
        [
            pf_agg_was_hero,
            pf_raises_norm,
            prev_street_raises,
            prev_street_hero_invested,
            double_barrel,
            turn_check_back,
        ],
        dtype=np.float32,
    )


def _group_h(dp: dict) -> np.ndarray:
    """7 dims: board texture."""
    board = list(dp.get("board") or [])
    tex = _analyze_board(board)

    return np.array(
        [
            float(tex["flush_possible"]),
            float(tex["flush_made"]),
            float(tex["straight_possible"]),
            float(tex["paired"]),
            float(tex["monotone"]),
            float(tex["connectedness"]),
            float(tex["high_card_rank"]),
        ],
        dtype=np.float32,
    )


def _group_j_postflop(dp: dict) -> np.ndarray:
    """10 dims: villain_pos one-hot (6) + villain_was_pf_agg + tightness + breadth + rel_pos."""
    villain_pos = dp.get("facing_pos") or dp.get("preflop_aggressor")

    pos_onehot = np.zeros(6, dtype=np.float32)
    if villain_pos and villain_pos in _POS_IDX:
        pos_onehot[_POS_IDX[villain_pos]] = 1.0

    hero_pos = dp.get("hero_pos", "")
    preflop_aggressor = dp.get("preflop_aggressor", "")
    villain_was_pf_agg = (
        1.0
        if (preflop_aggressor and preflop_aggressor != hero_pos and preflop_aggressor == villain_pos)
        else 0.0
    )

    # tightness + breadth: position-based proxy
    _tightness = {"UTG": 1.0, "MP": 0.8, "CO": 0.6, "BTN": 0.4, "SB": 0.5, "BB": 0.5}
    _breadth = {"UTG": 0.3, "MP": 0.4, "CO": 0.6, "BTN": 0.8, "SB": 0.5, "BB": 0.7}
    tightness = _tightness.get(str(villain_pos), 0.5)
    breadth = _breadth.get(str(villain_pos), 0.5)

    # villain_pos_relative_to_hero: seat difference normalized [-1, 1] → [0, 1]
    hero_seat = _POS_IDX.get(hero_pos, 0)
    vill_seat = _POS_IDX.get(str(villain_pos), 0) if villain_pos else 0
    rel_pos_raw = (vill_seat - hero_seat) / 6.0  # in [-1, 1] range roughly
    rel_pos_norm = float(np.clip((rel_pos_raw + 1.0) / 2.0, 0.0, 1.0))

    return np.concatenate(
        [
            pos_onehot,
            [villain_was_pf_agg, tightness, breadth, rel_pos_norm],
        ]
    ).astype(np.float32)


# ---- Main extractor ----------------------------------------------------------


def extract_postflop(
    dp: dict,
    equity_lookup: EquityLookup,
    pop_mean: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Extract 80-dim postflop embedding vector for a decision point.

    Args:
        dp: Decision point dict (DECISIONS-SCHEMA v1.1)
        equity_lookup: Loaded EquityLookup instance
        pop_mean: Population-mean equity dict for miss fallback (auto-computed if None)

    Returns:
        (vector, filter_dict) where vector is float32 ndarray of shape (80,)
        and filter_dict has keys: street_class, pot_type, hero_pos_rel, n_players_active
    """
    if pop_mean is None:
        pop_mean = equity_lookup.population_mean_equity()

    street = str(dp.get("street", "flop"))

    a = _group_a_postflop(dp, equity_lookup, pop_mean, street)  # 12 dims
    b = _group_b(dp, street)  # 8 dims
    c = _group_c(dp)  # 6 dims
    d = _group_d_postflop(dp)  # 8 dims
    e = _group_e_postflop(dp)  # 7 dims
    f = _group_f_postflop(dp, street)  # 16 dims
    g = _group_g(dp, street)  # 6 dims
    h = _group_h(dp)  # 7 dims
    j = _group_j_postflop(dp)  # 10 dims

    vector = np.concatenate([a, b, c, d, e, f, g, h, j], dtype=np.float32)
    vector = np.clip(vector, 0.0, 1.0)

    assert len(vector) == 80, f"Postflop vector has {len(vector)} dims, expected 80"

    filter_dict = {
        "street_class": "postflop",
        "pot_type": dp.get("pot_type", ""),
        "hero_pos_rel": dp.get("hero_pos_rel", ""),
        "n_players_active": int(dp.get("n_players_at_street", 2)),
    }

    return vector, filter_dict
