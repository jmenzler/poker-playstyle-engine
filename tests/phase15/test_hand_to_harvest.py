# long-ok-file
"""Unit tests for build_hand_harvest: flop-rooting, nav_line parallelism,
hero_seat alignment, multiway flagging, and no-flop-DP fail-loud path.
"""

from __future__ import annotations

import pytest

_SRP_PREFLOP = ["BTN:open_2.5", "BB:call"]
_FLOP = ["Ah", "Kd", "2c"]
_TURN = "7s"
_RIVER = "3h"


def _felt(
    street: str,
    board_cards: list[str],
    *,
    hero_pos: str = "BTN",
    action_sequence: list[str] | None = None,
    effective_stack_bb: float = 100.0,
    pot_size_bb: float = 6.0,
    opponents_remaining: int = 1,
) -> dict:
    if action_sequence is None:
        seq: list[str] = list(_SRP_PREFLOP)
        seq.append("/")
        seq.append("BB:check")
        if street in ("turn", "river"):
            seq.append("BTN:bet_half_pot")
            seq.append("BB:call")
            seq.append("/")
            seq.append("BB:check")
        if street == "river":
            seq.append("BTN:bet_half_pot")
            seq.append("BB:call")
            seq.append("/")
            seq.append("BB:check")
        action_sequence = seq
    return {
        "street": street,
        "board_cards": board_cards,
        "hero_position": hero_pos,
        "action_sequence": action_sequence,
        "effective_stack_bb": effective_stack_bb,
        "pot_size_bb": pot_size_bb,
        "opponents_remaining": opponents_remaining,
        "hero_hole_cards": ["Ac", "Kc"],
        "hero_facing_bet_bb": 0.0,
    }


def _row(
    hand_id: str,
    decision_id: str,
    felt: dict,
) -> dict:
    return {
        "hand_id": hand_id,
        "decision_id": decision_id,
        "felt_snapshot": felt,
        "obs_id": f"obs-{decision_id}",
        "session_id": "sess-1",
        "ts": "2026-01-01T00:00:00Z",
        "cluster_key": "",
        "action_taken": "check",
        "source": "sim",
        "solver_label": None,
        "max_neighbor_distance": 0.5,
        "flagged_sparse": False,
        "embedding": None,
    }


def _hu_hand() -> list[dict]:
    """3-DP HU hand: flop check, turn bet/call, river DP."""
    flop_board = list(_FLOP)
    turn_board = [*_FLOP, _TURN]
    river_board = [*_FLOP, _TURN, _RIVER]

    flop_seq = [*_SRP_PREFLOP, "/", "BB:check"]
    turn_seq = [*_SRP_PREFLOP, "/", "BB:check", "BTN:bet_half_pot", "BB:call", "/", "BB:check"]
    river_seq = [
        *_SRP_PREFLOP,
        "/",
        "BB:check",
        "BTN:bet_half_pot",
        "BB:call",
        "/",
        "BB:check",
        "BTN:bet_half_pot",
        "BB:call",
        "/",
        "BB:check",
    ]

    return [
        _row("H1", "H1_dp0", _felt("flop", flop_board, action_sequence=flop_seq)),
        _row("H1", "H1_dp1", _felt("turn", turn_board, action_sequence=turn_seq)),
        _row("H1", "H1_dp2", _felt("river", river_board, action_sequence=river_seq)),
    ]


def test_flop_rooted_spot_board_pot_stack() -> None:
    """spec.spot uses the flop DP's 3-card board and flop-start pot/stack."""
    from src.solver.hand_to_harvest import build_hand_harvest

    rows = _hu_hand()
    spec = build_hand_harvest(rows, palette_lookup={})

    assert spec.spot.board == list(_FLOP), "board must be 3-card flop"
    flop_felt = rows[0]["felt_snapshot"]
    assert spec.spot.pot == round(float(flop_felt["pot_size_bb"]) * 100)
    assert spec.spot.effective_stack == round(float(flop_felt["effective_stack_bb"]) * 100)


def test_nav_lines_dp_meta_parallel_and_runout_cards() -> None:
    """nav_lines and dp_meta are strictly parallel; turn/river cards are set correctly."""
    from src.solver.hand_to_harvest import build_hand_harvest

    rows = _hu_hand()
    spec = build_hand_harvest(rows, palette_lookup={})

    assert len(spec.nav_lines) == len(spec.dp_meta) == 3

    turn_nl = spec.nav_lines[1]
    assert turn_nl["turn_card"] == _TURN
    assert turn_nl["river_card"] is None

    river_nl = spec.nav_lines[2]
    assert river_nl["turn_card"] == _TURN
    assert river_nl["river_card"] == _RIVER


def test_hero_seat_matches_nav_line_hero_player() -> None:
    """Each DpMeta.hero_seat equals the hero_player from its nav_line."""
    from src.solver.hand_to_harvest import build_hand_harvest

    rows = _hu_hand()
    spec = build_hand_harvest(rows, palette_lookup={})

    for i, (nl, meta) in enumerate(zip(spec.nav_lines, spec.dp_meta, strict=True)):
        assert meta.hero_seat == nl["hero_player"], (
            f"DP {i}: DpMeta.hero_seat={meta.hero_seat} != nav_line hero_player={nl['hero_player']}"
        )

    assert spec.dp_meta[0].street == "flop"


def test_multiway_turn_flagged_hu_flop_not_flagged() -> None:
    """Multiway turn DP → multiway=True, nav_ok_expected=False; HU flop → False/True."""
    from src.solver.hand_to_harvest import build_hand_harvest

    rows = _hu_hand()

    turn_seq = [*_SRP_PREFLOP, "/", "BB:check", "BTN:bet_half_pot", "BB:call", "/", "BB:check"]
    multiway_turn = _row(
        "H1",
        "H1_dp1_mw",
        _felt(
            "turn",
            [*_FLOP, _TURN],
            action_sequence=turn_seq,
            opponents_remaining=2,
        ),
    )
    rows_mw = [rows[0], multiway_turn]
    spec = build_hand_harvest(rows_mw, palette_lookup={})

    flop_meta = spec.dp_meta[0]
    assert not flop_meta.multiway
    assert flop_meta.nav_ok_expected

    turn_meta = spec.dp_meta[1]
    assert turn_meta.multiway
    assert not turn_meta.nav_ok_expected


def test_no_flop_dp_raises_value_error() -> None:
    """A hand with only turn/river DPs raises ValueError (fail loudly)."""
    from src.solver.hand_to_harvest import build_hand_harvest

    rows = _hu_hand()[1:]  # drop the flop row
    with pytest.raises(ValueError, match="no flop DP"):
        build_hand_harvest(rows, palette_lookup={})
