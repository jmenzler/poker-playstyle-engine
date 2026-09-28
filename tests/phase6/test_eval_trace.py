"""Unit tests for src/eval/trace.py.

Closes Blocker 3 of Plan 06-05b: cluster attribution is implemented properly
(via EngineRecorder + DecisionTrace), NOT empty lists.

Tests are pure — no DB, no RLCard, no subprocess. Just structural verification
of RecordedDecision / DecisionTrace / EngineRecorder.
"""

from __future__ import annotations

from src.eval.trace import DecisionTrace, EngineRecorder, RecordedDecision


def test_recorded_decision_constructs():
    rd = RecordedDecision(
        hand_id=0,
        street="flop",
        cluster_key="ck1",
        hero_action="bet_50",
        hero_chips_in=50.0,
    )
    assert rd.cluster_key == "ck1"
    assert rd.street == "flop"
    assert rd.hero_chips_in == 50.0
    assert rd.hand_id == 0
    assert rd.hero_action == "bet_50"


def test_recorded_decision_is_frozen():
    """RecordedDecision must be a frozen msgspec.Struct — no in-place mutation."""
    import msgspec  # noqa: F401 — confirms msgspec is the chosen lib

    rd = RecordedDecision(hand_id=0, street="preflop", cluster_key="ck", hero_action="call")
    # msgspec.Struct(frozen=True) raises AttributeError on attribute assignment.
    try:
        rd.hand_id = 1  # type: ignore[misc]
    except (AttributeError, TypeError):
        return
    raise AssertionError("RecordedDecision must be frozen")


def test_decision_trace_append_and_iter():
    trace = DecisionTrace()
    trace.append(
        RecordedDecision(
            hand_id=0,
            street="preflop",
            cluster_key="ck0",
            hero_action="call",
            hero_chips_in=2.0,
        )
    )
    trace.append(
        RecordedDecision(
            hand_id=0,
            street="flop",
            cluster_key="ck1",
            hero_action="bet_50",
            hero_chips_in=50.0,
        )
    )
    assert len(trace) == 2
    decisions = list(trace)
    assert decisions[0].cluster_key == "ck0"
    assert decisions[1].cluster_key == "ck1"


def test_engine_recorder_wraps_decide_returns_same_result():
    """EngineRecorder.decide returns identical tuple to wrapped engine."""
    fake_engine = type("E", (), {"decide": lambda self, gs: ("bet_50", False, 0.1, [0.0] * 80)})()
    trace = DecisionTrace()
    recorder = EngineRecorder(fake_engine, trace)
    recorder.start_hand(7)
    recorder.note_street("flop")
    gs = {"cluster_key": "ck_X"}
    result = recorder.decide(gs)
    assert result == ("bet_50", False, 0.1, [0.0] * 80)


def test_engine_recorder_records_decision_with_correct_fields():
    """After recorder.decide() the trace contains a RecordedDecision with state-derived cluster_key."""
    fake_engine = type("E", (), {"decide": lambda self, gs: ("bet_50", False, 0.1, [0.0] * 80)})()
    trace = DecisionTrace()
    recorder = EngineRecorder(fake_engine, trace)
    recorder.start_hand(7)
    recorder.note_street("flop")
    gs = {"cluster_key": "ck_X"}
    recorder.decide(gs)
    assert len(trace) == 1
    rd = next(iter(trace))
    assert rd.hand_id == 7
    assert rd.street == "flop"
    assert rd.cluster_key == "ck_X"
    assert rd.hero_action == "bet_50"
    # Coarse weight for bet_50 (no my_chips in state)
    assert rd.hero_chips_in == 0.5


def test_engine_recorder_start_hand_resets_state():
    """start_hand resets hand_id and street to preflop."""
    fake_engine = type("E", (), {"decide": lambda self, gs: ("call", False, 0.0, [])})()
    trace = DecisionTrace()
    recorder = EngineRecorder(fake_engine, trace)
    recorder.start_hand(3)
    recorder.note_street("river")
    recorder.start_hand(4)
    # After start_hand(4), street resets to preflop
    recorder.decide({"cluster_key": "ck"})
    rd = list(trace)[-1]
    assert rd.hand_id == 4
    assert rd.street == "preflop"


def test_engine_recorder_extracts_cluster_key_from_5tuple_result():
    """If wrapped engine returns 5-tuple, EngineRecorder uses index 4 as cluster_key."""
    fake_engine = type(
        "E",
        (),
        {"decide": lambda self, gs: ("call", False, 0.0, [], "ENGINE_CLUSTER")},
    )()
    trace = DecisionTrace()
    recorder = EngineRecorder(fake_engine, trace)
    recorder.start_hand(0)
    recorder.decide({"cluster_key": "STATE_CLUSTER"})
    rd = next(iter(trace))
    # 5-tuple result wins over state attribute (engine is the source of truth)
    assert rd.cluster_key == "ENGINE_CLUSTER"


def test_engine_recorder_falls_back_to_check_on_nostrategy():
    """Zero-coverage spot (NoStrategyError), not facing a bet → fall back to check, counted."""
    from src._errors import NoStrategyError

    class _NoStrat:
        def decide_with_encoding(self, gs):
            raise NoStrategyError("zero neighbors: collection=postflop_decisions")

    trace = DecisionTrace()
    recorder = EngineRecorder(_NoStrat(), trace)
    recorder.start_hand(1)
    recorder.note_street("turn")
    # dict game_state has no hero_facing_bet_bb attr → getattr default 0 → check.
    result = recorder.decide({"cluster_key": "ignored"})
    assert result == "check"
    assert recorder.nostrategy_fallbacks == 1
    rd = next(iter(trace))
    assert rd.hero_action == "check"
    assert rd.cluster_key == "nostrategy"  # _canon absent on the stub → sentinel


def test_engine_recorder_falls_back_to_fold_when_facing_bet():
    """Zero-coverage spot while facing a bet → fold (never fold for free applies to check)."""
    import types

    from src._errors import NoStrategyError

    class _NoStrat:
        def decide_with_encoding(self, gs):
            raise NoStrategyError("zero neighbors")

    trace = DecisionTrace()
    recorder = EngineRecorder(_NoStrat(), trace)
    recorder.start_hand(2)
    result = recorder.decide(types.SimpleNamespace(hero_facing_bet_bb=5.0))
    assert result == "fold"
    assert recorder.nostrategy_fallbacks == 1


def test_aggregate_proportional_to_chips_in():
    """Hand-0 has two decisions; bb_delta=+10 split proportional to chips."""
    trace = DecisionTrace()
    trace.append(
        RecordedDecision(
            hand_id=0,
            street="preflop",
            cluster_key="ck_pre",
            hero_action="call",
            hero_chips_in=2.0,
        )
    )
    trace.append(
        RecordedDecision(
            hand_id=0,
            street="flop",
            cluster_key="ck_flop",
            hero_action="bet_50",
            hero_chips_in=8.0,
        )
    )
    payoffs = {0: 10.0}
    agg = trace.aggregate_by_cluster(payoffs)
    # 2 chips out of 10 total = 20% → ck_pre gets 2.0 bb
    # 8 chips out of 10 total = 80% → ck_flop gets 8.0 bb
    assert agg["ck_pre"]["bb_total"] == 2.0
    assert agg["ck_flop"]["bb_total"] == 8.0
    assert agg["ck_pre"]["n_decisions"] == 1
    assert agg["ck_flop"]["n_decisions"] == 1


def test_aggregate_equal_split_when_no_chips():
    """All check/fold decisions: equal split fallback."""
    trace = DecisionTrace()
    trace.append(
        RecordedDecision(
            hand_id=0,
            street="preflop",
            cluster_key="ck_a",
            hero_action="check",
            hero_chips_in=0.0,
        )
    )
    trace.append(
        RecordedDecision(
            hand_id=0,
            street="flop",
            cluster_key="ck_b",
            hero_action="check",
            hero_chips_in=0.0,
        )
    )
    agg = trace.aggregate_by_cluster({0: -4.0})
    assert agg["ck_a"]["bb_total"] == -2.0
    assert agg["ck_b"]["bb_total"] == -2.0


def test_top5_profitable_and_leaky():
    """A short match: one cluster wins, one loses, one neutral.

    top5_profitable returns the winner (positive bb_total only);
    top5_leaky returns the loser (negative bb_total only);
    neutral cluster excluded from both.
    """
    trace = DecisionTrace()
    # Hand 0: ck_winner takes all +5 bb
    trace.append(
        RecordedDecision(
            hand_id=0,
            street="flop",
            cluster_key="ck_winner",
            hero_action="bet_50",
            hero_chips_in=5.0,
        )
    )
    # Hand 1: ck_loser takes all -3 bb
    trace.append(
        RecordedDecision(
            hand_id=1,
            street="flop",
            cluster_key="ck_loser",
            hero_action="bet_50",
            hero_chips_in=5.0,
        )
    )
    # Hand 2: ck_neutral nets 0
    trace.append(
        RecordedDecision(
            hand_id=2,
            street="flop",
            cluster_key="ck_neutral",
            hero_action="check",
            hero_chips_in=0.0,
        )
    )
    payoffs = {0: 5.0, 1: -3.0, 2: 0.0}
    profitable = trace.top5_profitable(payoffs)
    leaky = trace.top5_leaky(payoffs)
    assert any(p["cluster_key"] == "ck_winner" for p in profitable)
    assert any(item["cluster_key"] == "ck_loser" for item in leaky)
    assert profitable[0]["cluster_key"] == "ck_winner"  # highest bb_total first
    assert leaky[0]["cluster_key"] == "ck_loser"  # most-negative first
    # ck_neutral (bb_total == 0) is in neither list
    assert all(p["cluster_key"] != "ck_neutral" for p in profitable)
    assert all(item["cluster_key"] != "ck_neutral" for item in leaky)


def test_top5_lists_capped_at_5():
    """top5_profitable / top5_leaky return at most 5 entries."""
    trace = DecisionTrace()
    payoffs = {}
    for i in range(10):
        trace.append(
            RecordedDecision(
                hand_id=i,
                street="flop",
                cluster_key=f"ck_pos_{i}",
                hero_action="bet_50",
                hero_chips_in=1.0,
            )
        )
        payoffs[i] = float(i + 1)  # all positive
    profitable = trace.top5_profitable(payoffs)
    assert len(profitable) == 5
    # Highest bb_total first → ck_pos_9 (payoff 10) on top
    assert profitable[0]["cluster_key"] == "ck_pos_9"


def test_top5_structure_keys():
    """Each top5 item is dict with keys cluster_key, n_decisions, bb_total."""
    trace = DecisionTrace()
    trace.append(
        RecordedDecision(
            hand_id=0,
            street="flop",
            cluster_key="ck1",
            hero_action="bet_50",
            hero_chips_in=1.0,
        )
    )
    profitable = trace.top5_profitable({0: 5.0})
    assert len(profitable) == 1
    rec = profitable[0]
    assert set(rec.keys()) == {"cluster_key", "n_decisions", "bb_total"}
    assert isinstance(rec["cluster_key"], str)
    assert isinstance(rec["n_decisions"], int)
    assert isinstance(rec["bb_total"], float)


def test_engine_recorder_chips_in_for_check_fold_is_zero():
    """Action 'check' or 'fold' records hero_chips_in == 0.0 (no chip commitment)."""
    fake_engine_check = type("E", (), {"decide": lambda self, gs: ("check", False, 0.0, [])})()
    trace = DecisionTrace()
    recorder = EngineRecorder(fake_engine_check, trace)
    recorder.start_hand(0)
    recorder.decide({"cluster_key": "ck"})
    assert next(iter(trace)).hero_chips_in == 0.0

    fake_engine_fold = type("E", (), {"decide": lambda self, gs: ("fold", False, 0.0, [])})()
    trace2 = DecisionTrace()
    recorder2 = EngineRecorder(fake_engine_fold, trace2)
    recorder2.start_hand(0)
    recorder2.decide({"cluster_key": "ck"})
    assert next(iter(trace2)).hero_chips_in == 0.0


# --------------------- real-engine shape (production path) -------------------
# The production KNNDecisionEngine.decide() returns a BARE STRING and the real
# GameState has no cluster_key. The recorder must derive cluster_key from
# decide_with_encoding(gs).enc.hard_filter via the canonical sorted-join.


class _FrozenGameState:
    """Stand-in for the real frozen GameState struct: no cluster_key attribute."""

    __slots__ = ("street",)

    def __init__(self, street: str = "preflop") -> None:
        self.street = street


class _EncResult:
    """Minimal EncodeResult stand-in carrying a hard_filter dict."""

    def __init__(self, hard_filter: dict) -> None:
        self.hard_filter = hard_filter


class _RealishEngine:
    """Mimics KNNDecisionEngine: decide() -> bare string, decide_with_encoding -> 4-tuple."""

    def __init__(self, action: str, hard_filter: dict) -> None:
        self._action = action
        self._hf = hard_filter

    def decide(self, gs):
        action, _flag, _dist, _enc = self.decide_with_encoding(gs)
        return action

    def decide_with_encoding(self, gs):
        return self._action, False, 0.12, _EncResult(self._hf)


def test_recorder_derives_cluster_key_from_hard_filter_for_real_engine():
    """Production path: cluster_key comes from enc.hard_filter, not 'unknown'."""
    hf = {
        "street_class": "preflop",
        "pot_type": "srp",
        "hero_pos_rel": "IP",
        "n_players_active": 2,
    }
    engine = _RealishEngine("open_2_2bb", hf)
    trace = DecisionTrace()
    recorder = EngineRecorder(engine, trace)
    recorder.start_hand(0)
    recorder.decide(_FrozenGameState("preflop"))
    rd = next(iter(trace))
    # Canonical sorted-join — identical to sim.harness._cluster_key_from_filter.
    assert rd.cluster_key == ("hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=preflop")
    assert rd.cluster_key != "unknown"
    assert rd.hero_action == "open_2_2bb"


def test_recorder_returns_bare_string_for_real_engine():
    """Recorder preserves the real decide() contract: returns the bare action string."""
    engine = _RealishEngine("call", {"street_class": "preflop"})
    trace = DecisionTrace()
    recorder = EngineRecorder(engine, trace)
    recorder.start_hand(0)
    result = recorder.decide(_FrozenGameState("preflop"))
    assert result == "call"


def test_recorder_counts_unknown_cluster_keys():
    """When hard_filter is empty the cluster_key resolves 'unknown' and is counted."""
    engine = _RealishEngine("call", {})
    trace = DecisionTrace()
    recorder = EngineRecorder(engine, trace)
    recorder.start_hand(0)
    recorder.decide(_FrozenGameState("preflop"))
    assert recorder.unknown_cluster_count == 1
    assert next(iter(trace)).cluster_key == "unknown"


def test_action_weight_covers_preflop_sized_raises():
    """_ACTION_WEIGHT must carry the preflop sized-raise verbs (not fall back to 1.0)."""
    from src.eval.trace import _ACTION_WEIGHT

    for verb in ("open_2_2bb", "open_3bb", "3bet_3x", "3bet_4x", "4bet_2_5x"):
        assert verb in _ACTION_WEIGHT, f"{verb} missing from _ACTION_WEIGHT"
    # A 4bet commits more chips than a 2.2bb open.
    assert _ACTION_WEIGHT["4bet_2_5x"] > _ACTION_WEIGHT["open_2_2bb"]
    assert _ACTION_WEIGHT["3bet_4x"] > _ACTION_WEIGHT["3bet_3x"]


def test_recorder_chips_in_uses_sized_raise_weight():
    """A sized-raise action records its table weight, not the 1.0 fallback."""
    engine = _RealishEngine("4bet_2_5x", {"street_class": "preflop"})
    trace = DecisionTrace()
    recorder = EngineRecorder(engine, trace)
    recorder.start_hand(0)
    recorder.decide(_FrozenGameState("preflop"))
    from src.eval.trace import _ACTION_WEIGHT

    assert next(iter(trace)).hero_chips_in == _ACTION_WEIGHT["4bet_2_5x"]
