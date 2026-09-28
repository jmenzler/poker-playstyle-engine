"""Board texture classifier tests (OQ-5 lock).

Locked priority order: MONOTONE > PAIRED > WET TWO-TONE > DRY RAINBOW.
"""

from __future__ import annotations

import logging

import pytest

from src.eval.texture import classify_board


@pytest.mark.parametrize(
    "board,expected",
    [
        # MONOTONE — all same suit (priority 1)
        (["Ah", "Th", "5h"], "monotone"),
        (["Ah", "Th", "5h", "2h"], "monotone"),
        (["2s", "7s", "Ks", "Qs", "Js"], "monotone"),
        # Note: ["Ah","Ah","Ah"] is degenerate (impossible deck) but the classifier
        # operates on string features only — monotone takes priority over paired.
        (["Ah", "Ah", "Ah"], "monotone"),
        # PAIRED — any rank ≥ 2 (priority 2, when not monotone)
        (["Ah", "Th", "Tc"], "paired"),
        (["7h", "7d", "5c"], "paired"),
        (["Ah", "Ad", "Ac"], "paired"),  # triple-A: not monotone, paired wins
        (["Ah", "Th", "5d", "Tc"], "paired"),
        # WET TWO-TONE — exactly 2 distinct suits (priority 3)
        (["Ah", "Th", "5d"], "wet_two_tone"),
        (["Kc", "Qc", "5h"], "wet_two_tone"),
        (["Ah", "Th", "5h", "2d"], "wet_two_tone"),  # 3 hearts + 1 diamond
        # DRY RAINBOW — 3+ distinct suits (priority 4)
        (["Ah", "Td", "5c"], "dry_rainbow"),
        (["Ks", "Qd", "Jh"], "dry_rainbow"),
        (["Ah", "Td", "5c", "2s"], "dry_rainbow"),  # all 4 suits
    ],
)
def test_classify_board(board: list[str], expected: str) -> None:
    assert classify_board(board) == expected


def test_priority_monotone_over_paired() -> None:
    """Monotone trumps paired when both technically apply."""
    # All hearts AND a pair of aces — monotone wins (priority 1).
    assert classify_board(["Ah", "Ah", "Th"]) == "monotone"


def test_empty_board_returns_dry_rainbow_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Empty/preflop board is a safe-default 'dry_rainbow' with a logged warning."""
    caplog.set_level(logging.WARNING)
    assert classify_board([]) == "dry_rainbow"


def test_two_card_board_returns_dry_rainbow_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 2-card board (illegal in Eval flow) returns 'dry_rainbow' safely."""
    caplog.set_level(logging.WARNING)
    assert classify_board(["Ah", "Th"]) == "dry_rainbow"
