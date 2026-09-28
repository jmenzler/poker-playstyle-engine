"""Unit + integration tests for src/eval/run_match.py.

Fast tests run in every CI invocation; the two RLCard-gated integration
tests are skipped under CI (real env required + minutes-scale).

The fast suite covers Blockers 2 / 3 / 4 via structural assertions:
    - Blocker 2: ``_bb_per_chip`` helper does NOT exist; inline
      ``payoffs[0] / 2.0`` IS present.
    - Blocker 3: EngineRecorder + DecisionTrace are wired in (real
      cluster attribution surface — verified via inspect.getsource).
    - Blocker 4: ``_per_street_attribution`` is the per-street EV path;
      proportional to ``hero_chips_in``, NOT equal-split.
"""

from __future__ import annotations

import inspect
import os

import pytest

from src.eval.run_match import (
    _STREETS,
    _engine_version,
    _per_street_attribution,
    verdict,
)
from src.eval.trace import RecordedDecision


def test_streets_constant_has_four_values():
    """Per-street EV always carries four buckets."""
    assert _STREETS == ("preflop", "flop", "turn", "river")


def test_verdict_won_when_ci_low_above_zero():
    assert verdict(0.5, 2.0) == "won"


def test_verdict_lost_when_ci_high_below_zero():
    assert verdict(-2.0, -0.5) == "lost"


def test_verdict_inconclusive_when_ci_straddles_zero():
    assert verdict(-1.0, 1.0) == "inconclusive"


def test_verdict_regression_when_loss_against_prev_engine():
    assert verdict(-2.0, -0.5, is_regression_test=True) == "regression"


def test_verdict_regression_flag_irrelevant_when_won():
    """is_regression_test only matters for the lost branch."""
    assert verdict(0.5, 2.0, is_regression_test=True) == "won"


def test_engine_version_returns_str_or_none():
    v = _engine_version()
    assert v is None or isinstance(v, str)


def test_engine_version_reads_pyproject_version_when_available():
    """If pyproject.toml [project].version exists, _engine_version returns it."""
    v = _engine_version()
    # Repo's pyproject.toml has version="0.1.0" — verifies real read, not stub.
    assert v == "0.1.0"


def test_no_bb_per_chip_helper_dead_code():
    """Blocker 2: `_bb_per_chip` helper must NOT exist (was dead+broken)."""
    from src.eval import run_match as rm

    src = inspect.getsource(rm)
    assert "_bb_per_chip" not in src, "dead+broken `_bb_per_chip` helper must be deleted (Blocker 2)"


def test_inline_chip_to_bb_conversion_present():
    """Blocker 2 / Pitfall 2: inline `payoffs[0] / 2.0` must be present."""
    from src.eval import run_match as rm

    src = inspect.getsource(rm)
    assert "payoffs[0] / 2.0" in src or "payoffs[0]/2.0" in src, (
        "inline `payoffs[0] / 2.0` (Pitfall 2) must be present"
    )


def test_engine_recorder_and_decision_trace_wired_in():
    """Blocker 3: run_match must use EngineRecorder + DecisionTrace."""
    from src.eval import run_match as rm

    src = inspect.getsource(rm)
    assert "EngineRecorder" in src, "EngineRecorder must be wired in (Blocker 3)"
    assert "DecisionTrace" in src, "DecisionTrace must be wired in (Blocker 3)"


def test_per_street_attribution_uses_real_function_not_equal_split():
    """Blocker 4: per-street EV must come from _per_street_attribution()."""
    from src.eval import run_match as rm

    src = inspect.getsource(rm)
    assert "_per_street_attribution" in src, "Blocker 4 path missing"


def test_per_street_attribution_proportional_to_chips():
    """Blocker 4: per-street EV is proportional to hero_chips_in per street."""
    decisions = [
        RecordedDecision(
            hand_id=0,
            street="preflop",
            cluster_key="ck_p",
            hero_action="call",
            hero_chips_in=2.0,
        ),
        RecordedDecision(
            hand_id=0,
            street="flop",
            cluster_key="ck_f",
            hero_action="bet_50",
            hero_chips_in=8.0,
        ),
    ]
    out = _per_street_attribution(bb_delta=10.0, hand_decisions=decisions)
    assert out["preflop"] == 2.0  # 20% of 10
    assert out["flop"] == 8.0  # 80% of 10
    assert out["turn"] == 0.0
    assert out["river"] == 0.0


def test_per_street_attribution_edge_case_all_checks():
    """Hero only checked: bb_delta attributed to last street observed."""
    decisions = [
        RecordedDecision(
            hand_id=0,
            street="preflop",
            cluster_key="ck_p",
            hero_action="check",
            hero_chips_in=0.0,
        ),
        RecordedDecision(
            hand_id=0,
            street="flop",
            cluster_key="ck_f",
            hero_action="check",
            hero_chips_in=0.0,
        ),
    ]
    out = _per_street_attribution(bb_delta=-3.0, hand_decisions=decisions)
    assert out["flop"] == -3.0
    assert out["preflop"] == 0.0


def test_per_street_attribution_no_decisions_attributes_to_preflop():
    """Safety: empty decision list maps bb_delta to preflop bucket."""
    out = _per_street_attribution(bb_delta=5.0, hand_decisions=[])
    assert out["preflop"] == 5.0
    assert out["flop"] == 0.0


def test_per_street_attribution_returns_all_four_keys():
    """Output dict always has preflop/flop/turn/river keys."""
    out = _per_street_attribution(bb_delta=0.0, hand_decisions=[])
    assert set(out.keys()) == {"preflop", "flop", "turn", "river"}


def test_unknown_opponent_raises_keyerror():
    """run_match raises KeyError pointing to REGISTRY when opponent unknown."""
    from src.eval.run_match import run_match

    with pytest.raises(KeyError, match="unknown"):
        run_match("not_a_real_opponent")


def test_run_match_exports_verdict_and_run_match():
    """Module surface contract: verdict + run_match are exported."""
    from src.eval import run_match as rm

    assert hasattr(rm, "verdict")
    assert hasattr(rm, "run_match")
    assert callable(rm.verdict)
    assert callable(rm.run_match)


def test_engine_agent_propagates_engine_failure_not_check_call():
    """ERR-01: a broken engine / empty KB (NoStrategyError) must NOT be masked
    as a silent 'check_call' — it surfaces so a degraded engine can't score."""
    from src._errors import NoStrategyError
    from src.eval.run_match import _engine_agent
    from src.eval.trace import DecisionTrace, EngineRecorder

    class _BrokenEngine:
        def decide(self, gs):
            raise NoStrategyError("empty KB")

    recorder = EngineRecorder(_BrokenEngine(), DecisionTrace())
    recorder.start_hand(0)
    agent = _engine_agent(recorder, env=None)
    state = {
        "raw_obs": {
            "stage": "preflop",
            "hand": ["SA", "HK"],
            "public_cards": [],
            "all_chips": [1, 2],
            "stakes": [99, 98],
            "pot": 3,
            "current_player": 0,
            "legal_actions": [0, 1],
        },
        "legal_actions": {0: None, 1: None},
        "action_record": [],
    }
    with pytest.raises(NoStrategyError):
        agent.step(state)


# -------------------------- RLCard-gated integration -------------------------
# Skip under CI; require rlcard installed locally. Two tests exercise
# determinism (Test 5) and cluster attribution (Blocker 3 end-to-end).


@pytest.mark.skipif("CI" in os.environ, reason="full RLCard run; integration only")
def test_run_match_deterministic():
    """Two runs with same seed produce identical bb_per_100."""
    pytest.importorskip("rlcard")
    from src.eval.run_match import run_match

    class _StubEngine:
        def decide(self, gs):
            return ("call", False, 0.0, [], "stub_cluster")

    r1 = run_match("random", hands=50, seed=42, persist=False, _engine=_StubEngine())
    r2 = run_match("random", hands=50, seed=42, persist=False, _engine=_StubEngine())
    assert r1.bb_per_100 == r2.bb_per_100


@pytest.mark.skipif("CI" in os.environ, reason="full RLCard run; integration only")
def test_top5_attribution_from_known_winner_cluster():
    """Blocker 3: top5 lists derived from real trace, NOT empty stubs.

    A stub engine that always returns a fixed cluster_key 'KNOWN_WINNER_CLUSTER'.
    A short match should produce that cluster_key in either top5_profitable or
    top5_leaky (depending on whether the engine net-won or net-lost) — proving
    the attribution is wired, not an empty list.
    """
    pytest.importorskip("rlcard")
    from src.eval.run_match import run_match

    class _StubAlwaysCall:
        def decide(self, gs):
            return ("call", False, 0.0, [], "KNOWN_WINNER_CLUSTER")

    result = run_match("random", hands=20, seed=42, persist=False, _engine=_StubAlwaysCall())
    all_clusters = [p["cluster_key"] for p in result.top5_profitable] + [
        item["cluster_key"] for item in result.top5_leaky
    ]
    assert "KNOWN_WINNER_CLUSTER" in all_clusters, (
        "EngineRecorder must surface the cluster_key engine actually visited"
    )


def test_run_match_accepts_opponent_engine_for_head_to_head():
    """opponent_engine puts a pinned engine on seat 1 via _engine_agent (not a baseline)."""
    import inspect

    from src.eval import run_match as rm

    assert "opponent_engine" in inspect.signature(rm.run_match).parameters
    src = inspect.getsource(rm.run_match)
    assert "EngineRecorder(opponent_engine" in src
    assert "_engine_agent(villain_recorder" in src


@pytest.mark.skipif("CI" in os.environ, reason="full RLCard run; integration only")
def test_head_to_head_runs_two_pinned_engine_seats():
    """opponent_engine drives seat 1 as an engine; the match runs and scores the hero."""
    pytest.importorskip("rlcard")
    from src.eval.run_match import run_match

    class _StubAlwaysCall:
        def decide(self, gs):
            return ("call", False, 0.0, [], "C")

    result = run_match(
        opponent_engine=_StubAlwaysCall(),
        opponent_label="v_old",
        hands=10,
        seed=42,
        persist=False,
        _engine=_StubAlwaysCall(),
    )
    assert isinstance(result.bb_per_100, float)
