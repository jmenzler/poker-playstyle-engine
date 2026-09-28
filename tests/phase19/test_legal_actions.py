from __future__ import annotations

import pytest


def test_derive_legal_actions_check_to():
    from src._errors import ValidationError  # noqa: F401
    from src.study.gaps import _derive_legal_actions

    felt = {"hero_facing_bet_bb": 0, "effective_stack_bb": 50.0, "street": "flop"}
    actions = _derive_legal_actions(felt)
    assert "check" in actions
    assert "fold" not in actions
    assert "call" not in actions
    assert "bet_50" in actions
    assert "allin" in actions
    # postflop check-to must NOT offer preflop open sizes
    assert "open_2_2bb" not in actions
    assert "open_3bb" not in actions


def test_derive_legal_actions_facing_bet():
    from src.study.gaps import _derive_legal_actions

    felt = {"hero_facing_bet_bb": 5.0, "effective_stack_bb": 50.0, "street": "turn"}
    actions = _derive_legal_actions(felt)
    assert "fold" in actions
    assert "call" in actions
    assert "check" not in actions
    assert "bet_50" not in actions
    assert "allin" in actions
    assert "raise_pot" in actions
    # postflop facing-bet must NOT offer preflop 3bet/4bet sizes
    assert "3bet_3x" not in actions
    assert "3bet_4x" not in actions
    assert "4bet_2_5x" not in actions


def test_derive_legal_actions_preflop_street_aware():
    from src.study.gaps import _derive_legal_actions

    facing = _derive_legal_actions({"hero_facing_bet_bb": 3.0, "street": "preflop"})
    assert "3bet_3x" in facing
    assert "fold" in facing and "call" in facing
    assert "raise_pot" not in facing
    assert "bet_50" not in facing
    check_to = _derive_legal_actions({"hero_facing_bet_bb": 0, "street": "preflop"})
    assert "open_3bb" in check_to
    assert "bet_50" not in check_to


def test_validate_action_dist_rejects_bad_keys():
    from src._errors import ValidationError
    from src.study.gaps import _validate_action_dist

    with pytest.raises(ValidationError):
        _validate_action_dist({"nonsense": 1.0})


def test_validate_action_dist_rejects_negative():
    from src._errors import ValidationError
    from src.study.gaps import _validate_action_dist

    with pytest.raises(ValidationError):
        _validate_action_dist({"check": 1.5, "bet_50": -0.5})


def test_validate_action_dist_rejects_bad_sum():
    from src._errors import ValidationError
    from src.study.gaps import _validate_action_dist

    with pytest.raises(ValidationError):
        _validate_action_dist({"check": 0.5})
