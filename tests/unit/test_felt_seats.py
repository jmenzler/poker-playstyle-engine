"""Unit tests for god-view SIM replay reconstruction (src/study/felt_seats.py)."""

import pytest

from src.study.felt_seats import reconstruct_sim_replay


def _row(
    dp: int,
    pos: str,
    hole: list[str],
    action: str,
    street: str,
    board: list[str],
    seq: list[str],
    pot: float = 1.5,
) -> dict:
    return {
        "decision_id": f"hand1_dp{dp}",
        "action_taken": action,
        "felt_snapshot": {
            "street": street,
            "board_cards": board,
            "pot_size_bb": pot,
            "hero_position": pos,
            "hero_hole_cards": hole,
            "action_sequence": seq,
            "effective_stack_bb": 99.0,
        },
    }


def test_reconstructs_full_hand_timeline() -> None:
    """A 6-handed limped pot reconstructs one timeline with all seats + cards fixed."""
    rows = [
        _row(0, "UTG", ["Jd", "6c"], "fold", "preflop", [], []),
        _row(1, "MP", ["4d", "7s"], "fold", "preflop", [], ["UTG:fold"]),
        _row(2, "CO", ["Qs", "9s"], "call", "preflop", [], ["UTG:fold", "MP:fold"]),
        _row(3, "BTN", ["Ah", "Ad"], "call", "preflop", [], ["UTG:fold", "MP:fold", "CO:call"]),
        _row(4, "SB", ["Qd", "4s"], "fold", "preflop", [], ["UTG:fold", "MP:fold", "CO:call", "BTN:call"]),
        _row(5, "BB", ["Td", "Ks"], "check", "flop", ["3c", "3s", "7h"], []),
    ]
    steps = reconstruct_sim_replay(rows)

    assert len(steps) == 6, "one step per action (dp), in order"
    # Seats fixed across every step (god-view), all 6 positions present.
    seat_names = {v["name"] for v in steps[0]["seat_map"].values()}
    assert seat_names == {"UTG", "MP", "CO", "BTN", "SB", "BB"}
    for s in steps:
        assert {v["name"] for v in s["seat_map"].values()} == seat_names

    # Every seat's hole cards are known on EVERY step (not just its own dp).
    assert steps[0]["seat_cards"]["BTN"] == ["Ah", "Ad"]
    assert steps[0]["seat_cards"]["BB"] == ["Td", "Ks"]
    assert steps[5]["seat_cards"]["UTG"] == ["Jd", "6c"]

    # Actor advances with the action timeline.
    assert [s["actor"] for s in steps] == ["UTG", "MP", "CO", "BTN", "SB", "BB"]

    # Board builds per street (empty preflop, flop on the flop step).
    assert steps[0]["board"] == []
    assert steps[5]["board"] == ["3c", "3s", "7h"]

    # Button seat is stable; folds accumulate AFTER the folding step.
    assert steps[0]["button_seat"] is not None
    assert "UTG" not in steps[0]["folded"], "UTG still shown as actor on its own fold step"
    assert "UTG" in steps[1]["folded"], "UTG folded by the next step"
    assert "MP" in steps[2]["folded"]
    assert "CO" not in steps[5]["folded"], "CO called, never folds"


def _row_bet(
    dp: int, pos: str, action: str, street: str, pot: float, bet_before: float, board: list[str] | None = None
) -> dict:
    return {
        "decision_id": f"hand1_dp{dp}",
        "action_taken": action,
        "felt_snapshot": {
            "street": street,
            "board_cards": board or [],
            "pot_size_bb": pot,
            "hero_position": pos,
            "hero_hole_cards": ["2h", "2d"],
            "hero_bet_size_bb": bet_before,
            "action_sequence": [],
            "effective_stack_bb": 99.0,
        },
    }


def test_chips_and_pot_show_post_action_state() -> None:
    """felt_snapshot is pre-action; each step must show the actor's wager and the
    pot AFTER its action (delta = next dp's pot minus this dp's pot)."""
    flop = ["3c", "3s", "7h"]
    rows = [
        _row_bet(0, "UTG", "fold", "preflop", pot=1.5, bet_before=0.0),
        _row_bet(1, "BTN", "open_2_2bb", "preflop", pot=1.5, bet_before=0.0),
        _row_bet(2, "SB", "fold", "preflop", pot=3.7, bet_before=0.5),
        _row_bet(3, "BB", "call", "preflop", pot=3.7, bet_before=1.0),
        # hero_bet_size_bb is cumulative across the hand: 2.2 carried from preflop.
        _row_bet(4, "BB", "call", "flop", pot=4.9, bet_before=2.2, board=flop),
        _row_bet(5, "BTN", "raise_half_pot", "flop", pot=4.9, bet_before=2.2, board=flop),
        _row_bet(6, "BB", "fold", "flop", pot=7.35, bet_before=2.2, board=flop),
    ]
    steps = reconstruct_sim_replay(rows)

    # BTN's open shows its own 2.2bb on its own step, pot already includes it.
    assert steps[1]["bet_amount"] == 2.2
    assert steps[1]["pot"] == 3.7
    # pot_before is the pre-action pot (a decision view uses it to avoid leaking the action).
    assert steps[1]["pot_before"] == 1.5
    # SB's fold shows the dead blind, not a phantom bet.
    assert steps[2]["bet_amount"] == 0.5
    # BB's preflop call completes to the open size.
    assert steps[3]["bet_amount"] == 2.2
    assert steps[3]["pot"] == 4.9
    assert steps[3]["street_committed"]["BB"] == 2.2
    # Flop check: the cumulative preflop wager must NOT bleed into the new street.
    assert steps[4]["bet_amount"] == 0.0
    assert steps[4]["street_committed"]["BB"] == 0.0


def test_postflop_sim_vocab_translates_for_display() -> None:
    """SIM emits call-for-check and raise-for-bet; with no wager pending the
    display labels must read check / bet_*."""
    flop = ["3c", "3s", "7h"]
    rows = [
        _row_bet(0, "BB", "call", "flop", pot=4.9, bet_before=0.0, board=flop),
        _row_bet(1, "BTN", "raise_half_pot", "flop", pot=4.9, bet_before=0.0, board=flop),
        _row_bet(2, "BB", "call", "flop", pot=7.35, bet_before=0.0, board=flop),
    ]
    steps = reconstruct_sim_replay(rows)

    assert steps[0]["action_taken"] == "check", "call with nothing pending is a check"
    assert steps[0]["bet_amount"] == 0.0
    assert steps[1]["action_taken"] == "bet_50", "lead raise_* reconstructs to a sized bet (2.45/4.9=50%)"
    assert steps[1]["bet_amount"] == pytest.approx(2.45)
    assert steps[2]["action_taken"] == "call", "facing the bet it stays a call"
    # Last step has no next dp; a call's delta is the to-call amount.
    assert steps[2]["bet_amount"] == pytest.approx(2.45)


def test_verb_from_action_sequence_and_stacks_decrement() -> None:
    """action_taken can be a corpus-bucket mislabel (a 1bb limp tagged "open_2_2bb").
    The canonical action_sequence is the source of truth for the displayed verb, and
    each seat's stack decrements by its cumulative wager."""

    def r(dp: int, pos: str, taken: str, pot: float, herobet: float, seq: list[str]) -> dict:
        return {
            "decision_id": f"h_dp{dp}",
            "action_taken": taken,
            "felt_snapshot": {
                "street": "preflop",
                "board_cards": [],
                "pot_size_bb": pot,
                "hero_position": pos,
                "hero_hole_cards": ["Ah", "Kh"],
                "hero_bet_size_bb": herobet,
                "action_sequence": seq,
                "effective_stack_bb": 100.0,
            },
        }

    # UTG and BTN both limp (pot moves +1.0 each) but are tagged "open_2_2bb". A
    # non-dp SB fold appends between BTN's dp and BB's dp.
    rows = [
        r(0, "UTG", "open_2_2bb", 1.5, 0.0, []),
        r(1, "BTN", "open_2_2bb", 2.5, 0.0, ["UTG:call"]),
        r(2, "BB", "check", 3.5, 1.0, ["UTG:call", "BTN:call", "SB:fold"]),
    ]
    steps = reconstruct_sim_replay(rows)

    def seat(step: dict, name: str) -> float:
        return next(v["stack"] for v in step["seat_map"].values() if v["name"] == name)

    # Verb reflects the executed limp, not the open_2_2bb label; chips are the real 1.0.
    assert steps[0]["action_taken"] == "call"
    assert steps[0]["bet_amount"] == pytest.approx(1.0)
    # BTN's verb is matched by position even though SB:fold appended after it.
    assert steps[1]["action_taken"] == "call"

    # Stacks decrement by cumulative wager; un-acted seats keep the full stack.
    assert seat(steps[0], "UTG") == pytest.approx(99.0)
    assert seat(steps[0], "BTN") == pytest.approx(100.0)
    assert seat(steps[1], "BTN") == pytest.approx(99.0)


def test_preflop_raises_labeled_open_3bet_4bet() -> None:
    """Preflop voluntary raises read as poker semantics (open/3bet/4bet) by raise
    count, not the bet-agnostic raise_* vocab carried in the executed sequence."""

    def r(dp: int, pos: str, taken: str, pot: float, seq: list[str]) -> dict:
        return {
            "decision_id": f"h_dp{dp}",
            "action_taken": taken,
            "felt_snapshot": {
                "street": "preflop",
                "board_cards": [],
                "pot_size_bb": pot,
                "hero_position": pos,
                "hero_hole_cards": ["Ah", "Kh"],
                "hero_bet_size_bb": 0.0,
                "action_sequence": seq,
                "effective_stack_bb": 100.0,
            },
        }

    # UTG limps; BB raises (open); UTG re-raises (3bet); BB re-raises (4bet); UTG calls.
    rows = [
        r(0, "UTG", "call", 1.5, []),
        r(1, "BB", "open_3bb", 2.5, ["UTG:call"]),
        r(2, "UTG", "3bet_3x", 6.0, ["UTG:call", "BB:raise_half_pot"]),
        r(3, "BB", "4bet_2_5x", 14.0, ["UTG:call", "BB:raise_half_pot", "UTG:raise_pot"]),
        r(4, "UTG", "call", 18.0, ["UTG:call", "BB:raise_half_pot", "UTG:raise_pot", "BB:raise_pot"]),
    ]
    labels = [s["action_taken"] for s in reconstruct_sim_replay(rows)]
    assert labels == ["call", "open", "3bet", "4bet", "call"], labels


def test_postflop_bet_and_raise_show_sized_vocab() -> None:
    """Postflop bets/raises reconstruct the engine sized vocab (bet_25 / raise_min)
    from the actual chip delta, not the bet-agnostic raise_pot token RLCard records.
    Mirrors hand h994's turn: BB bets ~25% pot, UTG min-raises, BB calls."""
    turn = ["4c", "2d", "Qh", "As"]

    def r(dp: int, pos: str, taken: str, pot: float, herobet: float, seq: list[str]) -> dict:
        return {
            "decision_id": f"h_dp{dp}",
            "action_taken": taken,
            "felt_snapshot": {
                "street": "turn",
                "board_cards": turn,
                "pot_size_bb": pot,
                "hero_position": pos,
                "hero_hole_cards": ["Ah", "Kh"],
                "hero_bet_size_bb": herobet,
                "action_sequence": seq,
                "effective_stack_bb": 99.0,
            },
        }

    # The executed sequence tags both aggressive actions "raise_pot" (RLCard's
    # bet-agnostic vehicle); the real sizes are a 25%-pot bet and a min-raise.
    rows = [
        r(0, "BB", "bet_25", 4.5, 0.0, ["UTG:call", "/"]),
        r(1, "UTG", "raise_min", 5.5, 0.0, ["UTG:call", "/", "BB:raise_pot"]),
        r(2, "BB", "call", 7.5, 1.0, ["UTG:call", "/", "BB:raise_pot", "UTG:raise_pot"]),
    ]
    labels = [s["action_taken"] for s in reconstruct_sim_replay(rows)]
    assert labels == ["bet_25", "raise_min", "call"], labels


def test_terminal_aggressive_bet_shows_its_chips() -> None:
    """The LAST dp has no next dp, so pot_next is absent. A terminal aggressive
    bet must still render its real chip size, recovered from the sized token + the
    pre-action pot — not collapse to a zero delta."""
    flop = ["3c", "3s", "7h"]
    rows = [
        # BTN checks, then BB leads a 50%-pot bet as the final recorded action.
        _row_bet(0, "BTN", "call", "flop", pot=4.0, bet_before=0.0, board=flop),
        _row_bet(1, "BB", "bet_50", "flop", pot=4.0, bet_before=0.0, board=flop),
    ]
    steps = reconstruct_sim_replay(rows)

    assert steps[1]["action_taken"] == "bet_50"
    # 50% of the 4.0 pot = 2.0 chips, even with no next dp to diff the pot.
    assert steps[1]["bet_amount"] == pytest.approx(2.0)
    assert steps[1]["pot"] == pytest.approx(6.0)


def test_terminal_aggressive_raise_shows_its_chips() -> None:
    """A terminal raise (last dp, facing a bet) must recover its raise size from the
    raise vocab + the facing bet, not render zero chips."""
    flop = ["3c", "3s", "7h"]
    rows = [
        # BB bets 50% (2.0 into 4.0), then BTN min-raises as the final action.
        _row_bet(0, "BB", "bet_50", "flop", pot=4.0, bet_before=0.0, board=flop),
        # BTN faces the 2.0 bet (pot now 6.0); a min-raise is to 2x the facing bet.
        _row_bet(1, "BTN", "raise_min", "flop", pot=6.0, bet_before=0.0, board=flop),
    ]
    steps = reconstruct_sim_replay(rows)

    assert steps[1]["action_taken"] == "raise_min"
    # Facing 2.0, raise_min is to 4.0 total → 4.0 chips put in this street.
    assert steps[1]["bet_amount"] == pytest.approx(4.0)


def test_walk_hand_seats_all_six_including_bb() -> None:
    """A walk (everyone folds to the BB) gives the BB no dp and names it nowhere in
    the action history. The god-view must still seat all six — SIM is always 6-max —
    with the BB present on every step (its cards stay unknown). Mirrors hand h972."""

    def r(dp: int, pos: str, hole: list[str], seq: list[str]) -> dict:
        return {
            "decision_id": f"h_dp{dp}",
            "action_taken": "fold",
            "felt_snapshot": {
                "street": "preflop",
                "board_cards": [],
                "pot_size_bb": 1.5,
                "hero_position": pos,
                "hero_hole_cards": hole,
                "action_sequence": seq,
                "effective_stack_bb": 100.0,
            },
        }

    rows = [
        r(0, "UTG", ["6s", "7c"], []),
        r(1, "MP", ["2h", "3s"], ["UTG:fold"]),
        r(2, "CO", ["Td", "6c"], ["UTG:fold", "MP:fold"]),
        r(3, "BTN", ["6h", "2s"], ["UTG:fold", "MP:fold", "CO:fold"]),
        r(4, "SB", ["Ts", "3c"], ["UTG:fold", "MP:fold", "CO:fold", "BTN:fold"]),
    ]
    steps = reconstruct_sim_replay(rows)

    for s in steps:
        names = {v["name"] for v in s["seat_map"].values()}
        assert names == {"UTG", "MP", "CO", "BTN", "SB", "BB"}, names


def test_empty_rows() -> None:
    assert reconstruct_sim_replay([]) == []


def test_orders_by_dp_index_not_lexical() -> None:
    """dp10 must come after dp2 (numeric, not lexical)."""
    rows = [
        _row(10, "BB", ["2h", "2d"], "check", "river", ["3c", "3s", "7h", "3h", "4c"], []),
        _row(2, "CO", ["Qs", "9s"], "call", "preflop", [], []),
    ]
    steps = reconstruct_sim_replay(rows)
    assert [s["actor"] for s in steps] == ["CO", "BB"], "dp2 before dp10"
