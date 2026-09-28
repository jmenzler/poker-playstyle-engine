from __future__ import annotations

from src.study.probe import _facing_bet_from_spot, _filter_illegal_actions


def test_facing_bet_trailing_bet():
    assert _facing_bet_from_spot({"action_sequence": ["BB:check", "BTN:bet_50"]}) is True


def test_facing_bet_trailing_raise():
    assert _facing_bet_from_spot({"action_sequence": ["BB:bet_50", "BTN:raise_3x"]}) is True


def test_facing_bet_trailing_check_is_false():
    assert _facing_bet_from_spot({"action_sequence": ["BTN:check"]}) is False


def test_facing_bet_trailing_call_is_false():
    # a call closes the street → hero is first to act next street → no live bet
    assert _facing_bet_from_spot({"action_sequence": ["BTN:bet_50", "BB:call"]}) is False


def test_facing_bet_empty_sequence_is_false():
    assert _facing_bet_from_spot({"action_sequence": []}) is False


def test_drop_fold_when_checked_to():
    # the reported bug: KK checked-to on the river blended to fold 100%.
    # all-illegal blend → fall back to the passive legal action (check), never fold.
    out = _filter_illegal_actions({"fold": 1.0}, facing_bet=False)
    assert out == {"check": 1.0}


def test_all_illegal_facing_bet_falls_back_to_fold():
    out = _filter_illegal_actions({"check": 1.0}, facing_bet=True)
    assert out == {"fold": 1.0}


def test_renormalize_after_dropping_illegal_fold():
    out = _filter_illegal_actions({"fold": 0.5, "check": 0.3, "bet_75": 0.2}, facing_bet=False)
    assert "fold" not in out
    assert abs(sum(out.values()) - 1.0) < 1e-6
    assert out["check"] > out["bet_75"]


def test_drop_check_when_facing_bet():
    out = _filter_illegal_actions({"check": 0.4, "fold": 0.3, "call": 0.3}, facing_bet=True)
    assert "check" not in out
    assert abs(sum(out.values()) - 1.0) < 1e-6


def test_allin_legal_both_states():
    assert "allin" in _filter_illegal_actions({"allin": 0.5, "check": 0.5}, facing_bet=True)
    assert "allin" in _filter_illegal_actions({"allin": 0.5, "fold": 0.5}, facing_bet=False)
