"""Per-seat opponent profiles in run_match (heterogeneous tables)."""

from __future__ import annotations

import os

import pytest


class _StubEngine:
    def decide(self, gs):
        return ("call", False, 0.0, [], "stub_cluster")


def test_hero_position_heads_up():
    """HU: the button (dealer) posts SB and acts first; the other seat is BB."""
    from src.sim.adapter import _hero_position

    assert _hero_position(0, 0, n_players=2) == "SB"  # hero on the button
    assert _hero_position(1, 0, n_players=2) == "BB"
    assert _hero_position(1, 1, n_players=2) == "SB"  # button rotated to seat 1
    assert _hero_position(0, 1, n_players=2) == "BB"


def test_hero_position_6max_unchanged():
    """6-max ring is unchanged: dealer=BTN, +1=SB, +2=BB, +3=UTG, +4=MP, +5=CO."""
    from src.sim.adapter import _hero_position

    expected = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"}
    for pid, pos in expected.items():
        assert _hero_position(pid, 0) == pos
        assert _hero_position(pid, 0, n_players=6) == pos


def test_opponents_wrong_length_raises():
    """Validation fires before RLCard env construction (no rlcard needed)."""
    from src.eval.run_match import run_match

    with pytest.raises(ValueError, match="needs 1 entries"):
        run_match(opponents=["random", "random"], table_size=2, hands=1, persist=False, _engine=_StubEngine())


def test_opponents_unknown_raises():
    from src.eval.run_match import run_match

    with pytest.raises(KeyError):
        run_match(
            opponents=["not-a-real-opponent"], table_size=2, hands=1, persist=False, _engine=_StubEngine()
        )


@pytest.mark.skipif("CI" in os.environ, reason="full RLCard run; integration only")
def test_per_seat_label_and_runs():
    """Per-seat opponents run end-to-end; result.opponent is the joined composition."""
    pytest.importorskip("rlcard")
    from src.eval.run_match import run_match

    r = run_match(opponents=["random"], table_size=2, hands=20, seed=42, persist=False, _engine=_StubEngine())
    assert r.opponent == "random"


@pytest.mark.skipif("CI" in os.environ, reason="full RLCard run; integration only")
def test_per_seat_heterogeneous_6max_label():
    pytest.importorskip("rlcard")
    from src.eval.run_match import run_match

    seats = ["random", "always-call", "tight-passive", "LAG-profile", "TAG-profile"]
    r = run_match(opponents=seats, table_size=6, hands=10, seed=1, persist=False, _engine=_StubEngine())
    assert r.opponent == "+".join(seats)
