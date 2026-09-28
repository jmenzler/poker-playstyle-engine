"""Phase 6 OQ-1 resolution tests for src/solver/postflop_cli.py.

Plan: 06-02. Verifies:
- PostflopCliBackend.solve() parses solve_mode JSON stdout and returns SolverResult.
- _map_solver_to_canonical snaps BET/RAISE chip-ratios to the project's 15-action vocab.
- SolverParseError raised on malformed JSON, unknown labels, length mismatches, and
  bucket-snap > 20% off.
- NoStrategyError raised on subprocess non-zero exit (preserved Phase 4 behavior).
- Integration test (gated by @pytest.mark.integration + binary presence) verifies
  end-to-end solve on a real postflop-cli binary.

Mirrors the subprocess-mock pattern from tests/phase5/test_patch_engine.py and the
binary-existence skip pattern from tests/integration/test_postflop_cli_integration.py.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src._errors import NoStrategyError, SolverParseError
from src.decision_engine.blending import CANONICAL_ACTIONS
from src.solver.postflop_cli import (
    _BET_RATIO_BUCKETS,
    _RAISE_RATIO_BUCKETS,
    _SNAP_TOLERANCE,
    PostflopCliBackend,
    SolverSpot,
    _map_solver_to_canonical,
    _snap_to_bucket,
)

# ----------------------------------------------------------------------------
# Module-level constants — locked bucket tables (OQ-1)
# ----------------------------------------------------------------------------


def test_snap_tolerance_locked_at_20_percent() -> None:
    """OQ-1 lock: snap tolerance is 20%."""
    assert _SNAP_TOLERANCE == 0.20


def test_bet_ratio_buckets_locked() -> None:
    """OQ-1 lock: BET ratio buckets map ratio -> canonical bucket name."""
    expected_ratios = [0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.50]
    actual_ratios = [r for r, _ in _BET_RATIO_BUCKETS]
    assert actual_ratios == expected_ratios
    # All bucket names must be in the canonical vocab
    for _, name in _BET_RATIO_BUCKETS:
        assert name in CANONICAL_ACTIONS, f"bucket {name!r} not in CANONICAL_ACTIONS"


def test_raise_ratio_buckets_locked() -> None:
    """OQ-1 lock: RAISE ratio buckets map ratio -> canonical bucket name."""
    expected_ratios = [1.00, 2.50, 3.00, 4.00]
    actual_ratios = [r for r, _ in _RAISE_RATIO_BUCKETS]
    assert actual_ratios == expected_ratios
    for _, name in _RAISE_RATIO_BUCKETS:
        assert name in CANONICAL_ACTIONS, f"bucket {name!r} not in CANONICAL_ACTIONS"


# ----------------------------------------------------------------------------
# _snap_to_bucket
# ----------------------------------------------------------------------------


def test_snap_to_bucket_exact_match() -> None:
    """Exact-ratio match snaps to the matching bucket."""
    assert _snap_to_bucket(0.50, _BET_RATIO_BUCKETS, action_label="BET 50") == "bet_50"
    assert _snap_to_bucket(0.33, _BET_RATIO_BUCKETS, action_label="BET 33") == "bet_33"


def test_snap_to_bucket_overbet_catch_all() -> None:
    """ratio >= _BET_OVERBET_FLOOR (1.80) maps to bet_overbet (catch-all, no tolerance check)."""
    assert _snap_to_bucket(2.0, _BET_RATIO_BUCKETS, action_label="BET 200") == "bet_overbet"
    assert _snap_to_bucket(5.0, _BET_RATIO_BUCKETS, action_label="BET 500") == "bet_overbet"
    assert _snap_to_bucket(100.0, _BET_RATIO_BUCKETS, action_label="BET 10000") == "bet_overbet"


def test_snap_to_bucket_dead_zone_maps_to_overbet() -> None:
    """The 1.80-2.0 gap (>20% off bet_150, below the old 2.0 floor) maps to bet_overbet.

    Regression: BET 1661 / pot 900 = ratio 1.846 used to reject the whole node.
    """
    assert _snap_to_bucket(1.846, _BET_RATIO_BUCKETS, action_label="BET 1661") == "bet_overbet"
    assert _snap_to_bucket(1.80, _BET_RATIO_BUCKETS, action_label="BET 180") == "bet_overbet"
    # Just below the floor still snaps to bet_150 (|1.50-1.79|/1.50 = 0.193 < 0.20).
    assert _snap_to_bucket(1.79, _BET_RATIO_BUCKETS, action_label="BET 179") == "bet_150"


def test_snap_to_bucket_within_tolerance() -> None:
    """Ratio within 20% of nearest bucket snaps without error."""
    # 0.47 -> nearest is 0.50; rel_err = |0.50 - 0.47| / 0.50 = 0.06 (6%) — within 20%.
    assert _snap_to_bucket(0.47, _BET_RATIO_BUCKETS, action_label="BET 47") == "bet_50"
    # 0.40 -> nearest is 0.33; rel_err = |0.33 - 0.40| / 0.33 ≈ 0.21 (21%) — JUST out.
    # Use 0.38 instead: nearest is 0.33; rel_err = |0.33 - 0.38| / 0.33 ≈ 0.15 — within.
    assert _snap_to_bucket(0.38, _BET_RATIO_BUCKETS, action_label="BET 38") == "bet_33"


def test_snap_to_bucket_interior_gap_snaps_nearest() -> None:
    """Interior BET ratios in the bucket gaps snap to nearest, never reject.

    The vocab is intentionally coarse; geometric solver sizing ('e') routinely
    lands between buckets (e.g. 0.40-pot, 0.62-pot). These must snap, not raise.
    """
    # 0.40 -> nearest 0.33 (rel 0.21) — was rejected by the old 20% gate.
    assert _snap_to_bucket(0.40, _BET_RATIO_BUCKETS, action_label="BET 40") == "bet_33"
    # 0.62 -> nearest 0.50 (rel 0.24) — was rejected; now snaps.
    assert _snap_to_bucket(0.62, _BET_RATIO_BUCKETS, action_label="BET 62") == "bet_50"
    # 1.21 -> nearest 1.0 (rel 0.21) — interior gap below the overbet floor.
    assert _snap_to_bucket(1.21, _BET_RATIO_BUCKETS, action_label="BET 121") == "bet_100"


# ----------------------------------------------------------------------------
# _map_solver_to_canonical
# ----------------------------------------------------------------------------


def test_map_check_only() -> None:
    """Single CHECK action passes through to 'check' bucket."""
    out = _map_solver_to_canonical(["CHECK"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)
    assert out == {"check": 1.0}


def test_map_fold_only() -> None:
    """Single FOLD action passes through to 'fold' bucket."""
    out = _map_solver_to_canonical(["FOLD"], [1.0], pot_at_decision=100, prev_bet_at_decision=50)
    assert out == {"fold": 1.0}


def test_map_call_only() -> None:
    """Single CALL action passes through to 'call' bucket."""
    out = _map_solver_to_canonical(["CALL"], [1.0], pot_at_decision=100, prev_bet_at_decision=50)
    assert out == {"call": 1.0}


def test_map_bet_50_exact_ratio() -> None:
    """BET 50 in pot=100 → ratio 0.50 → bet_50 bucket exact."""
    out = _map_solver_to_canonical(["BET 50"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)
    assert out == {"bet_50": 1.0}


def test_map_bet_47_snaps_to_50() -> None:
    """BET 47 / pot 100 → ratio 0.47 → snap to bet_50 (6% off, within tolerance)."""
    out = _map_solver_to_canonical(["BET 47"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)
    assert out == {"bet_50": 1.0}


def test_map_bet_10_floors_to_bet_25() -> None:
    """BET 10 / pot 100 → ratio 0.10, below the bet_25 floor → bet_25 catch-all
    (deep-pot probes are real solver output; the vocab floor is the closest label)."""
    out = _map_solver_to_canonical(["BET 10"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)
    assert out == {"bet_25": 1.0}


def test_map_allin() -> None:
    """ALLIN <chips> always maps to 'allin' (no tolerance check)."""
    out = _map_solver_to_canonical(["ALLIN 500"], [1.0], pot_at_decision=100, prev_bet_at_decision=50)
    assert out == {"allin": 1.0}


def test_map_merged_actions_sum_frequencies() -> None:
    """Two BET actions that snap to the same bucket sum their frequencies.

    Three-action input keeps the total at 1.0 so we exercise the merge
    path without triggering defensive renormalization.
    """
    # BET 47 -> 0.47 -> bet_50; BET 53 -> 0.53 -> bet_50.
    out = _map_solver_to_canonical(
        ["CHECK", "BET 47", "BET 53"],
        [0.4, 0.3, 0.3],
        pot_at_decision=100,
        prev_bet_at_decision=0,
    )
    # bet_50 must absorb the two BETs (0.3 + 0.3 = 0.6).
    assert "bet_50" in out
    assert out["bet_50"] == pytest.approx(0.6)
    assert out["check"] == pytest.approx(0.4)
    # And no separate bet_47 / bet_53 leaked through.
    assert sum(out.values()) == pytest.approx(1.0, abs=0.001)
    assert len(out) == 2


def test_map_normalization_when_freqs_sum_off() -> None:
    """Mismatched-sum frequencies are renormalized so the output sums to ~1.0."""
    # Solver returns [0.4, 0.4] (sums to 0.8) — should renormalize.
    out = _map_solver_to_canonical(
        ["CHECK", "BET 50"],
        [0.4, 0.4],
        pot_at_decision=100,
        prev_bet_at_decision=0,
    )
    total = sum(out.values())
    assert total == pytest.approx(1.0, abs=0.001)


def test_map_length_mismatch_raises() -> None:
    """actions_root len != aggregate_freq_root len → SolverParseError."""
    with pytest.raises(SolverParseError, match=r"len"):
        _map_solver_to_canonical(
            ["CHECK", "BET 50"],
            [1.0],
            pot_at_decision=100,
            prev_bet_at_decision=0,
        )


def test_map_unknown_label_raises() -> None:
    """Unknown solver label → SolverParseError."""
    with pytest.raises(SolverParseError, match=r"unknown"):
        _map_solver_to_canonical(
            ["NUKE_THE_POT"],
            [1.0],
            pot_at_decision=100,
            prev_bet_at_decision=0,
        )


def test_map_realistic_three_action_solver_output() -> None:
    """Realistic CHECK / BET 33 / BET 66 mix sums to 1.0 with all canonical keys."""
    out = _map_solver_to_canonical(
        ["CHECK", "BET 33", "BET 66"],
        [0.4, 0.3, 0.3],
        pot_at_decision=100,
        prev_bet_at_decision=0,
    )
    # BET 33 -> 0.33 -> bet_33
    # BET 66 -> 0.66 -> nearest 0.75 (12% off) -> bet_75 (within tolerance)
    assert out["check"] == pytest.approx(0.4)
    assert out["bet_33"] == pytest.approx(0.3)
    assert "bet_75" in out
    assert sum(out.values()) == pytest.approx(1.0, abs=0.001)


def test_map_raise_min_exact() -> None:
    """RAISE TO 2x prev_bet → increment ratio 1.0 → raise_min bucket."""
    out = _map_solver_to_canonical(
        ["RAISE 100"], [1.0], pot_at_decision=100, prev_bet_at_decision=50, villain_bet_total=50
    )
    assert out == {"raise_min": 1.0}


def test_map_raise_huge_ratio_catches_all_to_raise_pot() -> None:
    """An increment ratio past the top bucket maps to raise_pot, no tolerance error."""
    out = _map_solver_to_canonical(
        ["RAISE 500"], [1.0], pot_at_decision=100, prev_bet_at_decision=50, villain_bet_total=50
    )
    assert out == {"raise_pot": 1.0}


def test_map_bet_below_floor_catches_all_to_bet_25() -> None:
    """A sub-25%-pot probe (deep 4bet pots) floors to bet_25, no tolerance error."""
    out = _map_solver_to_canonical(["BET 16"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)
    assert out == {"bet_25": 1.0}


def test_map_bet_interior_gap_snaps_without_tolerance_error() -> None:
    """BET ratios in the inter-bucket gaps snap to nearest; the coarse vocab
    approximates geometric solver sizing instead of rejecting the node."""
    # BET 40 / pot 100 -> 0.40 -> bet_33 (rel 0.21 off, old gate rejected this).
    out = _map_solver_to_canonical(["BET 40"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)
    assert out == {"bet_33": 1.0}
    # BET 62 / pot 100 -> 0.62 -> bet_50 (rel 0.24 off).
    out = _map_solver_to_canonical(["BET 62"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)
    assert out == {"bet_50": 1.0}


def test_map_bet_negative_chips_fails_loud() -> None:
    """A negative BET chip amount is garbage, not a snap — fail loud."""
    with pytest.raises(SolverParseError, match=r"BET -50"):
        _map_solver_to_canonical(["BET -50"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)


def test_map_raise_gap_ratio_snaps_to_nearest_without_tolerance_error() -> None:
    """Ratio 1.5 sits in the raise-bucket gap (1.0 vs 2.5, both >20% off) — the
    sparse raise buckets snap nearest instead of erroring."""
    out = _map_solver_to_canonical(
        ["RAISE 125"], [1.0], pot_at_decision=100, prev_bet_at_decision=50, villain_bet_total=50
    )
    assert out == {"raise_min": 1.0}


def test_map_raise_with_no_prev_bet_fails_loud() -> None:
    """A RAISE label at a node with prev_bet 0 is a contract violation, not a snap."""
    with pytest.raises(SolverParseError, match="prev_bet 0"):
        _map_solver_to_canonical(["RAISE 100"], [1.0], pot_at_decision=100, prev_bet_at_decision=0)


def test_map_raise_increment_uses_villain_total_when_hero_has_wagered() -> None:
    """Re-raise node: hero wagered 50, villain TO 150 (to_call 100); RAISE TO 400
    → increment 250 / to_call 100 = 2.5 → raise_2_5x."""
    out = _map_solver_to_canonical(
        ["RAISE 400"], [1.0], pot_at_decision=300, prev_bet_at_decision=100, villain_bet_total=150
    )
    assert out == {"raise_2_5x": 1.0}


# ----------------------------------------------------------------------------
# PostflopCliBackend.solve() — unit (mocked subprocess)
# ----------------------------------------------------------------------------


@pytest.fixture
def fake_spot() -> SolverSpot:
    """Minimal SolverSpot for unit tests; pot=100, no prev_bet."""
    return SolverSpot(
        pot=100,
        effective_stack=500,
        board=["Ah", "7c", "2d"],
        range_ip="AKo,AKs",
        range_oop="22+",
    )


def test_solve_parses_valid_stdout(fake_spot: SolverSpot) -> None:
    """Mock subprocess returns valid solve_mode JSON → SolverResult populated."""
    fake_output = {
        "exploitability_pct": 0.5,
        "time_ms": 1234,
        "actions_root": ["CHECK", "BET 50", "BET 100"],
        "aggregate_freq_root": [0.4, 0.3, 0.3],
        "actions_ip_after_check": [],
        "aggregate_freq_ip_after_check": [],
    }
    fake_result = subprocess.CompletedProcess(
        args=["postflop-cli"],
        returncode=0,
        stdout=json.dumps(fake_output),
        stderr="",
    )
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=fake_result):
        result = backend.solve(fake_spot)
    assert result.exploitability_pct == 0.5
    assert result.solve_time_ms == 1234
    # CHECK -> check; BET 50 -> bet_50 (0.5 ratio exact); BET 100 -> bet_100 (1.0 ratio exact)
    assert result.action_dist["check"] == pytest.approx(0.4)
    assert result.action_dist["bet_50"] == pytest.approx(0.3)
    assert result.action_dist["bet_100"] == pytest.approx(0.3)
    assert sum(result.action_dist.values()) == pytest.approx(1.0, abs=0.001)


def test_solve_nonzero_exit_raises_nostrategy(fake_spot: SolverSpot) -> None:
    """Subprocess nonzero exit → NoStrategyError with stderr in message."""
    fake_result = subprocess.CompletedProcess(
        args=["postflop-cli"],
        returncode=7,
        stdout="",
        stderr="convergence failed in 500 iters",
    )
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=fake_result):
        with pytest.raises(NoStrategyError, match=r"convergence"):
            backend.solve(fake_spot)


def test_solve_malformed_json_raises_parseerror(fake_spot: SolverSpot) -> None:
    """Malformed JSON stdout → SolverParseError wrapping JSONDecodeError."""
    fake_result = subprocess.CompletedProcess(
        args=["postflop-cli"],
        returncode=0,
        stdout="this is not json {{{",
        stderr="",
    )
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=fake_result):
        with pytest.raises(SolverParseError, match=r"not valid JSON"):
            backend.solve(fake_spot)


def test_solve_no_longer_raises_notimplemented(fake_spot: SolverSpot) -> None:
    """Phase 6 contract: solve() must not raise NotImplementedError."""
    fake_output = {
        "exploitability_pct": 0.5,
        "time_ms": 1,
        "actions_root": ["CHECK"],
        "aggregate_freq_root": [1.0],
        "actions_ip_after_check": [],
        "aggregate_freq_ip_after_check": [],
    }
    fake_result = subprocess.CompletedProcess(
        args=["postflop-cli"], returncode=0, stdout=json.dumps(fake_output), stderr=""
    )
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=fake_result):
        # Must NOT raise NotImplementedError; should return SolverResult.
        result = backend.solve(fake_spot)
    assert result.action_dist == {"check": 1.0}


# ----------------------------------------------------------------------------
# Integration test — gated by binary presence + integration marker
# ----------------------------------------------------------------------------


@pytest.mark.integration
def test_real_postflop_cli_solve_returns_normalized_dist() -> None:
    """PC-only: real binary roundtrip; action_dist sums to 1.0 ± 0.001.

    Skipped on Mac dev where the binary isn't present. Run on PC with
    `POSTFLOP_CLI_BIN` env var or the default ~/postflop-cli path.
    """
    binary = Path(
        os.environ.get(
            "POSTFLOP_CLI_BIN",
            "~/postflop-cli/target/release/postflop-cli",
        )
    ).expanduser()
    if not binary.exists():
        pytest.skip(f"postflop-cli binary not found: {binary} — run on PC")

    backend = PostflopCliBackend(binary_path=binary)
    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo",
        range_oop="AA:0.5,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,K9s,K8s,K7s,K6s",
        max_iterations=10,
        target_exploitability_pct=5.0,
    )
    result = backend.solve(spot, timeout_s=120.0)
    assert result.solve_time_ms > 0
    assert 0.0 <= result.exploitability_pct <= 100.0
    assert sum(result.action_dist.values()) == pytest.approx(1.0, abs=0.001)
