"""Tests for per-decision latency buffer in src/sim/harness.py (METR-03).

Verifies that run_record_session collects per-decision timing via
time.perf_counter() and exposes it in summary['latency_buffer'].
Also verifies SIM-02 byte-identical determinism is unaffected.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._errors import NoStrategyError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_encode_result(hard_filter=None, embedding=None):
    """Build a minimal EncodeResult-like MagicMock."""
    enc = MagicMock()
    enc.hard_filter = hard_filter or {"street_class": "postflop", "pot_type": "srp"}
    enc.embedding = embedding or [0.1, 0.2, 0.3]
    return enc


def _make_engine(enc_result=None, raises=False):
    """Build a mock engine whose decide_with_encoding returns (action, flagged_sparse, max_neighbor_distance, enc) or raises."""
    engine = MagicMock()
    engine._canon = MagicMock()
    engine._canon.encode.return_value = _make_encode_result()

    if raises:
        engine.decide_with_encoding.side_effect = NoStrategyError("no strategy")
    else:
        enc = enc_result or _make_encode_result()
        engine.decide_with_encoding.return_value = ("fold", False, None, enc)

    return engine


def _make_adapter(n_decisions_per_hand=3, n_hands=1):
    """Build a mock adapter that yields n_decisions_per_hand decisions per hand."""
    adapter = MagicMock()
    adapter._env = MagicMock()

    # has_more() returns True for each decision then False
    call_count = {"n": 0}

    def has_more_side_effect():
        # Each hand: True n_decisions_per_hand times, then False
        hand_calls = call_count["n"] % (n_decisions_per_hand + 1)
        call_count["n"] += 1
        return hand_calls < n_decisions_per_hand

    adapter.has_more.side_effect = has_more_side_effect
    adapter.next_game_state.return_value = MagicMock()
    adapter.current_legal_actions.return_value = ["fold", "call"]
    adapter.map_to_rlcard_action.return_value = 0
    return adapter


def _make_writer():
    """Build a minimal mock ObservationWriter."""
    writer = MagicMock()
    return writer


# ---------------------------------------------------------------------------
# Test 1: latency_buffer exists in summary and is a list of floats
# ---------------------------------------------------------------------------


def test_run_record_session_returns_latency_buffer_in_summary():
    """summary['latency_buffer'] is a list of floats after run_record_session."""
    from src.sim.harness import run_record_session

    engine = _make_engine()
    adapter = _make_adapter(n_decisions_per_hand=3, n_hands=1)
    writer = _make_writer()

    summary = run_record_session(adapter, engine, writer, session_seed=42, n_hands=1)

    assert "latency_buffer" in summary, "summary must contain 'latency_buffer'"
    buf = summary["latency_buffer"]
    assert isinstance(buf, list), f"latency_buffer must be list, got {type(buf)}"
    assert len(buf) > 0, "latency_buffer must be non-empty after at least one decision"
    for v in buf:
        assert isinstance(v, float), f"each latency entry must be float, got {type(v)}"


# ---------------------------------------------------------------------------
# Test 2: len(latency_buffer) == total_decisions
# ---------------------------------------------------------------------------


def test_latency_buffer_length_equals_total_decisions():
    """len(summary['latency_buffer']) == summary['total_decisions']."""
    from src.sim.harness import run_record_session

    engine = _make_engine()
    adapter = _make_adapter(n_decisions_per_hand=5, n_hands=2)
    writer = _make_writer()

    summary = run_record_session(adapter, engine, writer, session_seed=7, n_hands=2)

    assert len(summary["latency_buffer"]) == summary["total_decisions"], (
        f"latency_buffer length {len(summary['latency_buffer'])} != "
        f"total_decisions {summary['total_decisions']}"
    )


# ---------------------------------------------------------------------------
# Test 3: each latency is positive (wall-clock)
# ---------------------------------------------------------------------------


def test_latency_values_are_positive():
    """Each value in latency_buffer is >= 0 (wall-clock elapsed time)."""
    from src.sim.harness import run_record_session

    engine = _make_engine()
    adapter = _make_adapter(n_decisions_per_hand=3, n_hands=1)
    writer = _make_writer()

    summary = run_record_session(adapter, engine, writer, session_seed=99, n_hands=1)
    buf = summary["latency_buffer"]
    assert len(buf) > 0
    for i, v in enumerate(buf):
        assert v >= 0, f"latency_buffer[{i}] = {v} is negative"


# ---------------------------------------------------------------------------
# Test 4: NoStrategyError fallback still appends latency entry
# ---------------------------------------------------------------------------


def test_nostrategy_fallback_still_records_latency():
    """When engine.decide_with_encoding raises NoStrategyError, latency is still recorded."""
    from src.sim.harness import run_record_session

    engine = _make_engine(raises=True)  # always raises NoStrategyError
    adapter = _make_adapter(n_decisions_per_hand=2, n_hands=1)
    writer = _make_writer()

    summary = run_record_session(adapter, engine, writer, session_seed=1, n_hands=1)

    assert "latency_buffer" in summary
    buf = summary["latency_buffer"]
    # All decisions attempted — even failed ones should have a latency entry
    assert len(buf) == summary["total_decisions"], (
        "NoStrategyError fallbacks must still produce a latency entry"
    )
    assert summary["nostrategy_fallbacks"] > 0, "expected at least one fallback"
    for v in buf:
        assert v >= 0


# ---------------------------------------------------------------------------
# Test 5: SIM-02 determinism — same seed produces identical checksum
# ---------------------------------------------------------------------------


def _compute_recorded_checksum(calls) -> str:
    """SHA256 over recorded (cluster_key, embedding, action_taken) tuples."""
    h = hashlib.sha256()
    for call in calls:
        kwargs = call.kwargs if call.kwargs else {}
        args = call.args if call.args else ()

        # Support both positional and keyword invocation of writer.record()
        cluster_key = kwargs.get("cluster_key", args[0] if len(args) > 0 else "")
        embedding = kwargs.get("embedding", args[1] if len(args) > 1 else [])
        action_taken = kwargs.get("action_taken", args[2] if len(args) > 2 else "")

        h.update(str(cluster_key).encode())
        h.update(b"|")
        h.update(",".join(f"{x:.6f}" for x in embedding).encode())
        h.update(b"|")
        h.update(str(action_taken).encode())
        h.update(b"\n")
    return h.hexdigest()


def test_sim02_determinism_unchanged_by_latency_buffer():
    """Same seed produces byte-identical (cluster_key, embedding, action) sequences.

    Verifies that adding latency_buffer to the summary dict does NOT alter the
    SIM-02 byte-identical observation contract (timing is wall-clock, not part
    of any checksummed field).
    """
    from src.sim.harness import run_record_session

    engine = _make_engine()
    writer_a = _make_writer()
    writer_b = _make_writer()
    adapter_a = _make_adapter(n_decisions_per_hand=3, n_hands=2)
    adapter_b = _make_adapter(n_decisions_per_hand=3, n_hands=2)

    run_record_session(adapter_a, engine, writer_a, session_seed=1234, n_hands=2)
    run_record_session(adapter_b, engine, writer_b, session_seed=1234, n_hands=2)

    checksum_a = _compute_recorded_checksum(writer_a.record.call_args_list)
    checksum_b = _compute_recorded_checksum(writer_b.record.call_args_list)

    assert checksum_a == checksum_b, (
        f"SIM-02 violated: checksums differ between two same-seed runs. "
        f"A={checksum_a[:12]}... B={checksum_b[:12]}..."
    )
