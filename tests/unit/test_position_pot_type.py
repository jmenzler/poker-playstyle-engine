"""Shared preflop pot_type helper — build and serve must agree on the label.

Regression guard for the train/serve skew where extract_decisions stamped a
hand-final pot_type on every preflop decision while the serve encoder computed
it per-decision. ~46% of preflop DPs were mislabeled (Session 5).
"""

from __future__ import annotations

from tools.position import preflop_pot_type


def _seq(*actions: str) -> list[dict]:
    """Build an action list with the build-side ("action") shape."""
    return [{"action": a} for a in actions]


def test_no_raises_is_limp() -> None:
    assert preflop_pot_type(_seq("fold", "fold")) == "limp"
    assert preflop_pot_type([]) == "limp"


def test_single_raise_is_srp() -> None:
    # MP opens, CO folds — hero faces a single open.
    assert preflop_pot_type(_seq("fold", "raise", "fold")) == "srp"


def test_two_raises_is_3bet() -> None:
    # MP open, BTN 3bet — hero (SB) faces a 3bet.
    assert preflop_pot_type(_seq("fold", "raise", "fold", "raise")) == "3bet"


def test_three_raises_is_4bet() -> None:
    assert preflop_pot_type(_seq("raise", "raise", "raise")) == "4bet"


def test_four_or_more_raises_is_5bet_plus() -> None:
    assert preflop_pot_type(_seq("raise", "raise", "raise", "raise")) == "5bet+"
    assert preflop_pot_type(_seq("raise", "raise", "raise", "raise", "raise")) == "5bet+"


def test_allin_counts_as_raise() -> None:
    # Matches serve-side _infer_pot_type which counts allin as a raise.
    assert preflop_pot_type(_seq("raise", "allin")) == "3bet"


def test_calls_and_folds_do_not_count() -> None:
    assert preflop_pot_type(_seq("call", "call", "fold")) == "limp"


def test_serve_token_shape_pos_key() -> None:
    # Serve passes the parsed-token shape; only "action" is read, so either works.
    assert preflop_pot_type([{"pos": "MP", "action": "raise"}]) == "srp"
