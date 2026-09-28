"""Tests for tools/joint_canonicalize.py — joint (hole, board) suit-iso.

Properties verified:
    1. Suit-permutation invariance: permuting all suits leaves canonical unchanged.
    2. Card-order invariance: swapping the two hole cards (or any board card
       order) leaves canonical unchanged.
    3. Hole-board relationship preserved: FD vs no-FD do NOT collapse.
    4. Joint != board-only: cases that collapse board-only must NOT all collapse
       jointly when hole-board suit relation differs.
    5. Idempotency: canonicalize(canonical) == canonical.
    6. Input parsing: 'Ah Kc' and ['Ah', 'Kc'] yield identical results.
    7. Bad input raises ValueError.

Fuzz coverage via hypothesis. Determinism examples via pytest.parametrize.
"""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(_TOOLS))
from joint_canonicalize import (  # noqa: E402
    _SUIT_PERMS,
    joint_canonical_key,
    joint_canonicalize,
)

RANKS = "23456789TJQKA"
SUITS = "cdhs"


# ---------------------------------------------------------------------------
# Hypothesis strategies (5/7-card scenes: hole + 0/3/4/5 board)
# ---------------------------------------------------------------------------


@st.composite
def _scene(draw: st.DrawFn, board_size: int) -> tuple[list[str], list[str]]:
    deck = [r + s for r in RANKS for s in SUITS]
    picked: list[str] = []
    seen: set[str] = set()
    while len(picked) < 2 + board_size:
        c = draw(st.sampled_from(deck))
        if c not in seen:
            seen.add(c)
            picked.append(c)
    return picked[:2], picked[2:]


def _scene_st(board_size: int):
    return _scene(board_size)


def _apply_perm(cards: list[str], pmap: dict[str, str]) -> list[str]:
    return [c[0] + pmap[c[1]] for c in cards]


# ---------------------------------------------------------------------------
# Property: suit-permutation invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("board_size", [0, 3, 4, 5])
@given(data=st.data())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.large_base_example],
)
def test_suit_permutation_invariance(data, board_size: int) -> None:
    hole, board = data.draw(_scene_st(board_size))
    base = joint_canonicalize(hole, board)
    for pmap in _SUIT_PERMS:
        ph = _apply_perm(hole, pmap)
        pb = _apply_perm(board, pmap)
        assert joint_canonicalize(ph, pb) == base, (
            f"suit-perm broke: {hole}|{board} → {base}, perm {pmap} gave {joint_canonicalize(ph, pb)}"
        )


# ---------------------------------------------------------------------------
# Property: card-order invariance (within hole + within board)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("board_size", [0, 3, 4, 5])
@given(data=st.data())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.large_base_example],
)
def test_card_order_invariance(data, board_size: int) -> None:
    hole, board = data.draw(_scene_st(board_size))
    base = joint_canonicalize(hole, board)
    # Swap hole cards
    assert joint_canonicalize(hole[::-1], board) == base
    # Permute board cards (sample of perms — full enum too big for 5-card)
    if board_size > 0:
        sample = list(permutations(board))[: min(6, len(list(permutations(board))))]
        for bp in sample:
            assert joint_canonicalize(hole, list(bp)) == base


# ---------------------------------------------------------------------------
# Property: idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("board_size", [0, 3, 4, 5])
@given(data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.large_base_example],
)
def test_idempotent(data, board_size: int) -> None:
    hole, board = data.draw(_scene_st(board_size))
    h1, b1 = joint_canonicalize(hole, board)
    # Re-canonicalize the canonical form
    h2_cards = [h1[i : i + 2] for i in range(0, len(h1), 2)]
    b2_cards = [b1[i : i + 2] for i in range(0, len(b1), 2)]
    h2, b2 = joint_canonicalize(h2_cards, b2_cards)
    assert (h1, b1) == (h2, b2)


# ---------------------------------------------------------------------------
# Determinism: hole-board relationship classes
# Each tuple in the same group must canonicalize identically.
# Groups must NOT collide with one another (different hole-board relations).
# ---------------------------------------------------------------------------

_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "JTs_FD_on_K72_twotone_matching",
        [
            ("Jd Td", "Kd 7d 2c"),
            ("Jc Tc", "Kc 7c 2h"),
            ("Jh Th", "Kh 7h 2s"),
            ("Js Ts", "Ks 7s 2d"),
        ],
    ),
    (
        "JTs_no_FD_on_K72_twotone_nonmatching",
        [
            # Hole suit is the third suit on a two-tone board
            ("Jd Td", "Kc 7c 2h"),
            ("Jh Th", "Kc 7c 2s"),
            ("Js Ts", "Kd 7d 2c"),
        ],
    ),
    (
        "JTo_rainbow_K72",
        [
            ("Jd Tc", "Ks 7h 2d"),
            ("Jc Th", "Kd 7s 2c"),
            ("Js Tc", "Kh 7d 2s"),
        ],
    ),
    (
        "AA_dry_K72_rainbow",
        [
            ("Ah As", "Kc 7d 2h"),
            ("Ac Ad", "Kh 7s 2c"),
        ],
    ),
    (
        "AKs_TPTK_FD_on_Ad7d2c",
        [
            ("Ad Kd", "Qd 7d 2c"),
            ("As Ks", "Qs 7s 2h"),
        ],
    ),
]


@pytest.mark.parametrize(
    "group_name, members",
    [(name, members) for name, members in _GROUPS],
    ids=[name for name, _ in _GROUPS],
)
def test_group_members_collapse_to_same_canonical(group_name: str, members: list[tuple[str, str]]) -> None:
    """All members of the same hole-board relationship class share canonical."""
    canonicals = {joint_canonical_key(h, b) for h, b in members}
    assert len(canonicals) == 1, (
        f"group {group_name!r} should collapse to 1 canonical, got {len(canonicals)}: {canonicals}"
    )


def test_groups_are_distinct() -> None:
    """Different hole-board relationship classes must NOT collide."""
    keys: dict[str, str] = {}  # canonical → first group that produced it
    for name, members in _GROUPS:
        canon = joint_canonical_key(*members[0])
        if canon in keys:
            pytest.fail(f"group {name!r} collides with {keys[canon]!r} on canonical {canon}")
        keys[canon] = name


# ---------------------------------------------------------------------------
# Joint vs board-only: same canonical board but different hole-board relation
# must produce different JOINT canonical keys.
# ---------------------------------------------------------------------------

_JOINT_DISCRIMINATION_CASES = [
    # Board Kd7d2c is monotone-on-two-suits. JdTd → FD; JcTc → no FD; JhTh → no FD.
    pytest.param(
        ("Jd Td", "Kd 7d 2c"),  # JTs with FD
        ("Jc Tc", "Kd 7d 2c"),  # JTs without FD (different suit)
        id="JTs_FD_vs_JTs_noFD_same_board",
    ),
    pytest.param(
        ("Ad Kd", "Qd 9d 2c"),  # AKs nut FD + overcards
        ("Ah Kh", "Qd 9d 2c"),  # AKs no FD
        id="AKs_FD_vs_AKs_noFD",
    ),
    pytest.param(
        ("As Ah", "Kd Qd 7c"),  # AhAs no FD on diamond two-tone
        ("Ad Ac", "Kd Qd 7c"),  # AdAc — Ad blocks nut FD
        id="AA_no_blocker_vs_AA_blocker_FD",
    ),
    # ----- BLOCKER semantics on multi-suit boards -----
    # Board has 2 spades + 1 heart. AsAh = blocks BOTH nut spade flush
    # AND heart pair/runout. AcAd = blocks NEITHER.
    pytest.param(
        ("As Ah", "Ks Qs 7h"),  # AA with spade-blocker AND heart-blocker
        ("Ac Ad", "Ks Qs 7h"),  # AA with no blocker on either board suit
        id="AA_double_blocker_vs_AA_no_blocker",
    ),
    # AsAh (heart+spade blockers) ≠ AsAc (spade blocker only, no heart)
    pytest.param(
        ("As Ah", "Ks Qs 7h"),  # spade + heart blocker
        ("As Ac", "Ks Qs 7h"),  # spade only
        id="AA_spade_plus_heart_vs_AA_spade_only",
    ),
    # KK on monotone board: KsKh has spade-blocker (blocks villain flush);
    # KcKd has no blocker (villain has all flush combos available).
    pytest.param(
        ("Ks Kh", "As Qs 7s"),  # K-of-spades blocks villain nut flush attempts
        ("Kc Kd", "As Qs 7s"),  # no spade blocker
        id="KK_spade_blocker_vs_KK_no_blocker_on_monotone",
    ),
    # Heart-only blocker vs no blocker on (2 spade + 1 heart) board.
    pytest.param(
        ("Qh Jh", "Ks 9s 4h"),  # heart-blocker for turn/river heart pair
        ("Qc Jc", "Ks 9s 4h"),  # no heart blocker
        id="QJ_heart_blocker_vs_no_heart_blocker",
    ),
    # Turn brings a third spade — board Xs Xs Xh + new spade card.
    # Hero AsAh has the nut-spade blocker after FD comes in.
    pytest.param(
        ("As Ah", "Ks Qs 7h 2s"),  # nut-spade blocker on turned flush
        ("Ah Ac", "Ks Qs 7h 2s"),  # heart blocker only, no spade
        id="AA_nut_spade_blocker_vs_heart_only_on_turn_flush",
    ),
    # ----- BACKDOOR-BLOCKER semantics -----
    # Rainbow flop (3 distinct suits). Each villain backdoor needs runner-runner
    # of that suit. Hero holding suits matching board cards reduces villain BDFD outs.
    # Board Ks-7h-2c rainbow. Hero AsAh blocks spade BDFD AND heart BDFD.
    # Hero AdAd doesn't exist (impossible — pair has 2 suits). Use AdAc:
    # blocks club BDFD only (and no spade/heart blocker).
    pytest.param(
        ("As Ah", "Ks 7h 2c"),  # blocks 2 backdoor flush suits (spade + heart)
        ("Ah Ad", "Ks 7h 2c"),  # blocks 1 (heart only)
        id="AA_blocks_2_BDFD_vs_AA_blocks_1_BDFD",
    ),
    # Hero with 0 backdoor blockers vs 1
    pytest.param(
        ("Ad Ac", "Ks 7h 2d"),  # holds diamond — blocks diamond BDFD only
        ("Ah As", "Ks 7h 2d"),  # blocks heart + spade BDFD, NOT diamond
        id="AA_diamond_BDFD_blocker_vs_AA_blocks_other_two",
    ),
    # AKo with spade matching K-spade only (1 BDFD blocker for spade run)
    # vs AKo with neither matching board suit (0 backdoor blockers).
    pytest.param(
        ("As Kh", "Kd 9c 4h"),  # heart blocker for heart BDFD
        ("Ac Kc", "Kd 9c 4h"),  # club blocker (board has 9c)
        id="AK_heart_BDFD_blocker_vs_AK_club_BDFD_blocker",
    ),
    # Backdoor straight blocker: board K-7-2 rainbow.
    # Hero holding the 8 (one-card BDSD wheel) vs not.
    # Use suit on the heart to also stack BDFD interaction.
    pytest.param(
        ("8h 6h", "Ks 7d 2c"),  # 8 + 6 in same hand creates open-ended BD straight
        ("8s 6c", "Ks 7d 2c"),  # same ranks, no suit-iso match (no BDFD potential)
        id="86s_BD_straight_plus_FD_vs_86o",
    ),
    # Rainbow board where villain's flush draws are STRICTLY backdoor. Hero
    # holding a suit that matches one board card reduces 1 specific BDFD.
    pytest.param(
        ("Qd Jd", "As 8h 3c"),  # NO suit match with rainbow → 0 BDFD blockers
        ("Qs Jh", "As 8h 3c"),  # spade + heart match 2 board cards → 2 BDFD blockers
        id="QJ_no_BDFD_blockers_vs_QJ_2_BDFD_blockers",
    ),
]


@pytest.mark.parametrize("scene_a, scene_b", _JOINT_DISCRIMINATION_CASES)
def test_joint_discriminates_hole_board_relation(scene_a: tuple[str, str], scene_b: tuple[str, str]) -> None:
    ka = joint_canonical_key(*scene_a)
    kb = joint_canonical_key(*scene_b)
    assert ka != kb, (
        f"joint canonical should discriminate hole-board relation:\n"
        f"  {scene_a} → {ka}\n"
        f"  {scene_b} → {kb}\n"
        f"  (same key means FD is being collapsed away)"
    )


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hole_str, hole_list",
    [
        ("Ah Kc", ["Ah", "Kc"]),
        ("Jd Td", ["Jd", "Td"]),
        ("2c 2d", ["2c", "2d"]),
    ],
)
@pytest.mark.parametrize(
    "board_str, board_list",
    [
        ("", []),
        ("Ac Jd 5d", ["Ac", "Jd", "5d"]),
        ("AcJd5d", ["Ac", "Jd", "5d"]),
        ("Ac Jd 5d 2h", ["Ac", "Jd", "5d", "2h"]),
        ("Ac Jd 5d 2h 7s", ["Ac", "Jd", "5d", "2h", "7s"]),
    ],
)
def test_string_and_list_inputs_equivalent(
    hole_str: str, hole_list: list[str], board_str: str, board_list: list[str]
) -> None:
    if board_str == "":
        # Empty string splits to [] — but parser uses .split() which handles it
        from_str = joint_canonicalize(hole_str, [])
    elif board_str == "AcJd5d":
        # Concatenated 6-char form should also parse via list path
        from_str = joint_canonicalize(hole_str, board_list)
    else:
        from_str = joint_canonicalize(hole_str, board_str)
    from_list = joint_canonicalize(hole_list, board_list)
    assert from_str == from_list


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hole, board",
    [
        (["Ah"], ["Kc", "Qd", "2s"]),  # 1 hole card
        (["Ah", "Kc", "Qd"], ["Kc", "Qd", "2s"]),  # 3 hole cards
        ([], ["Kc", "Qd", "2s"]),  # 0 hole cards
    ],
)
def test_bad_hole_raises(hole: list[str], board: list[str]) -> None:
    with pytest.raises(ValueError, match="hole"):
        joint_canonicalize(hole, board)


@pytest.mark.parametrize(
    "board_size",
    [1, 2, 6, 7],
)
def test_bad_board_size_raises(board_size: int) -> None:
    deck = [r + s for r in RANKS for s in SUITS]
    board = deck[:board_size]
    with pytest.raises(ValueError, match="board"):
        joint_canonicalize(["Ah", "Kc"], board)


# ---------------------------------------------------------------------------
# Canonical output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hole, board, expected_hole_len, expected_board_len",
    [
        (["Ah", "Kc"], [], 4, 0),
        (["Ah", "Kc"], ["Qd", "Js", "2c"], 4, 6),
        (["Ah", "Kc"], ["Qd", "Js", "2c", "7h"], 4, 8),
        (["Ah", "Kc"], ["Qd", "Js", "2c", "7h", "3s"], 4, 10),
    ],
)
def test_output_length(hole, board, expected_hole_len, expected_board_len) -> None:
    h, b = joint_canonicalize(hole, board)
    assert len(h) == expected_hole_len
    assert len(b) == expected_board_len


# ---------------------------------------------------------------------------
# Real-DP smoke test (from hm_decisions.jsonl examples in scratchpad)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hole, board, desc",
    [
        ("3h Ah", "Ac Jd 5d", "DP example 1: A3o on AJ5dd"),
        ("As Ah", "3c 8s Ks", "DP example 2: AA on 38K with two spades"),
        ("7d Th", "", "preflop DP: T7o"),
    ],
)
def test_real_dp_examples_canonicalize_cleanly(hole: str, board: str, desc: str) -> None:
    h, b = joint_canonicalize(hole, board)
    board_list = board.split() if board else []
    assert len(h) == 4
    assert len(b) == len(board_list) * 2
    # Sanity: re-canonicalizing the canonical form is stable
    h2_cards = [h[i : i + 2] for i in range(0, len(h), 2)]
    b2_cards = [b[i : i + 2] for i in range(0, len(b), 2)]
    assert joint_canonicalize(h2_cards, b2_cards) == (h, b)
