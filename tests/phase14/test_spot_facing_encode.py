"""Spot encoding must derive facing from the action_sequence trailing bet token.

The frontend sends action_sequence but never hero_facing_bet_bb, so _gamestate_from_spot
defaulted it to 0.0 -> every spot encoded as checked-to -> facing-bet probes retrieved
checked-to neighbors and collapsed to fold. The trailing token carries the facing bet;
derive hero_facing_bet_bb from it (bridging named bet_half_pot and frontend numeric bet_50).
"""

from __future__ import annotations

from tools.build_embedding import _gamestate_from_spot


def _spot(seq: list[str]) -> dict:
    return {
        "street": "flop",
        "hero_hole": ["Kh", "Ks"],
        "board": ["7h", "2d", "9c"],
        "hero_position": "BTN",
        "action_sequence": seq,
    }


def test_checked_to_spot_has_no_facing_bet() -> None:
    gs = _gamestate_from_spot(_spot(["CO:open_2.5", "BTN:call", "CO:check"]))
    assert gs.hero_facing_bet_bb == 0.0


def test_facing_bet_spot_derives_facing_from_trailing_token() -> None:
    gs = _gamestate_from_spot(_spot(["CO:open_2.5", "BTN:call", "CO:bet_pot"]))
    assert gs.hero_facing_bet_bb > 0.0


def test_named_and_numeric_bet_vocab_agree() -> None:
    pot = _gamestate_from_spot(_spot(["CO:bet_pot"])).hero_facing_bet_bb
    half_named = _gamestate_from_spot(_spot(["CO:bet_half_pot"])).hero_facing_bet_bb
    half_numeric = _gamestate_from_spot(_spot(["CO:bet_50"])).hero_facing_bet_bb
    assert pot > half_named > 0.0
    assert abs(half_numeric - half_named) < 1e-6


def test_trailing_raise_is_facing_bet() -> None:
    gs = _gamestate_from_spot(_spot(["CO:open_2.5", "BTN:bet_half_pot", "CO:raise_3x"]))
    assert gs.hero_facing_bet_bb > 0.0


def test_explicit_hero_facing_bet_bb_is_respected() -> None:
    gs = _gamestate_from_spot({**_spot(["CO:check"]), "hero_facing_bet_bb": 4.0})
    assert gs.hero_facing_bet_bb == 4.0
