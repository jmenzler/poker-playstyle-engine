"""Unit tests for src/eval/result.py — MatchResult struct + persistence helpers.

Persist/load round-trip is folded into the integration-test sections of
tests/phase6/test_eval_run_match.py (Task 3). These tests verify the struct
shape + module-level export contract.
"""

from __future__ import annotations

from src.eval.result import MatchResult, load_match, persist_match


def test_match_result_constructs_with_all_required_fields():
    r = MatchResult(
        match_id="some-uuid",
        opponent="random",
        hands=100,
        seed=42,
        bb_per_100=0.0,
        ci_low=-1.0,
        ci_high=1.0,
        per_street={"preflop": 0.0, "flop": 0.0, "turn": 0.0, "river": 0.0},
        per_texture={
            "dry_rainbow": 0.0,
            "wet_two_tone": 0.0,
            "paired": 0.0,
            "monotone": 0.0,
        },
        top5_profitable=[],
        top5_leaky=[],
        status="inconclusive",
    )
    assert r.opponent == "random"
    assert r.bb_per_100 == 0.0
    assert r.per_street["flop"] == 0.0
    assert r.top5_profitable == []
    assert r.status == "inconclusive"
    # Optional fields default to None
    assert r.engine_version is None
    assert r.started_at is None
    assert r.finished_at is None


def test_match_result_is_frozen():
    """MatchResult is a frozen msgspec.Struct."""
    r = MatchResult(
        match_id="x",
        opponent="random",
        hands=1,
        seed=1,
        bb_per_100=0.0,
        ci_low=0.0,
        ci_high=0.0,
        per_street={},
        per_texture={},
        top5_profitable=[],
        top5_leaky=[],
        status="inconclusive",
    )
    try:
        r.hands = 2  # type: ignore[misc]
    except (AttributeError, TypeError):
        return
    raise AssertionError("MatchResult must be frozen")


def test_module_exposes_persist_and_load():
    assert callable(persist_match)
    assert callable(load_match)
