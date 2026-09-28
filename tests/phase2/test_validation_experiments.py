"""Validation experiment tests for tools/validation_experiments.py.

Created as stubs during Phase 2 Plan 03 (test infra scaffolding).
Activated during Phase 2 Plan 07 (validation runner scaffold).

Each test asserts the SHAPE of the VerdictDict returned by the corresponding
run_test_N_* function. Data-quality assertions (actual PASS/FAIL thresholds)
are a runtime concern handled by Plan 09 on PC.

Integration mark: tests needing a live Milvus client are marked with
@pytest.mark.integration; synthetic probe tests use mocks only.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from tools.validation_experiments import (
    CRITICAL_TESTS,
    MITIGATION_TESTS,
    run_test_1_monotone_twin,
    run_test_2_made_vs_draw,
    run_test_3_position_sensitivity,
    run_test_4_spr_sensitivity,
    run_test_5_knn_consistency,
    run_test_6_bimodal_66_vs_kq,
    run_test_7_betting_sequence,
    run_test_8_ehs_baseline_beat,
)

# ─── Shared helpers ──────────────────────────────────────────────────────────

EXPECTED_VERDICT_KEYS = {"n", "name", "threshold", "observed", "verdict", "mitigation"}
VALID_VERDICTS = {"PASS", "FAIL", "PENDING"}


def _assert_verdict_shape(result: dict, expected_n: int) -> None:
    """Assert VerdictDict has the required keys and valid values."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    missing = EXPECTED_VERDICT_KEYS - result.keys()
    assert not missing, f"VerdictDict missing keys: {missing}"
    assert result["n"] == expected_n, f"Expected n={expected_n}, got n={result['n']}"
    assert isinstance(result["name"], str) and result["name"], "name must be non-empty str"
    assert isinstance(result["threshold"], str) and result["threshold"], "threshold must be non-empty str"
    assert isinstance(result["observed"], str), "observed must be str"
    assert result["verdict"] in VALID_VERDICTS, f"verdict must be one of {VALID_VERDICTS}"
    assert isinstance(result["mitigation"], str), "mitigation must be str"


# ─── Synthetic probes (no live Milvus needed) ────────────────────────────────


def test_monotone_twin() -> None:
    """CRITICAL Test 1: shape assertion for run_test_1_monotone_twin."""
    result = run_test_1_monotone_twin(client=None)
    _assert_verdict_shape(result, expected_n=1)


def test_made_vs_draw() -> None:
    """CRITICAL Test 2: shape assertion for run_test_2_made_vs_draw."""
    result = run_test_2_made_vs_draw(client=None)
    _assert_verdict_shape(result, expected_n=2)


def test_position_sensitivity() -> None:
    """Test 3 (mitigation-eligible): shape assertion for run_test_3_position_sensitivity."""
    result = run_test_3_position_sensitivity(client=None)
    _assert_verdict_shape(result, expected_n=3)


def test_spr_sensitivity() -> None:
    """Test 4 (mitigation-eligible): shape assertion for run_test_4_spr_sensitivity."""
    result = run_test_4_spr_sensitivity(client=None)
    _assert_verdict_shape(result, expected_n=4)


def test_bimodal_66_vs_kq() -> None:
    """Test 6 (mitigation-eligible): shape assertion for run_test_6_bimodal_66_vs_kq."""
    result = run_test_6_bimodal_66_vs_kq(client=None)
    _assert_verdict_shape(result, expected_n=6)


def test_betting_sequence_sensitivity() -> None:
    """Test 7 (mitigation-eligible): shape assertion for run_test_7_betting_sequence."""
    result = run_test_7_betting_sequence(client=None)
    _assert_verdict_shape(result, expected_n=7)


# ─── kNN-dependent tests (require live Milvus) ───────────────────────────────


@pytest.mark.integration
def test_knn_consistency(milvus_uri, milvus_token) -> None:
    """CRITICAL Test 5: shape assertion for run_test_5_knn_consistency (integration)."""
    mock_client = MagicMock()
    result = run_test_5_knn_consistency(client=mock_client)
    _assert_verdict_shape(result, expected_n=5)


@pytest.mark.integration
def test_ehs_baseline_beat(milvus_uri, milvus_token) -> None:
    """CRITICAL Test 8: shape assertion for run_test_8_ehs_baseline_beat (integration)."""
    mock_client = MagicMock()
    result = run_test_8_ehs_baseline_beat(client=mock_client)
    _assert_verdict_shape(result, expected_n=8)


# ─── Classification constants ─────────────────────────────────────────────────


def test_critical_and_mitigation_sets() -> None:
    """CRITICAL_TESTS and MITIGATION_TESTS have correct members and are disjoint."""
    assert CRITICAL_TESTS == frozenset({1, 2, 5, 8}), f"CRITICAL_TESTS wrong: {CRITICAL_TESTS}"
    assert MITIGATION_TESTS == frozenset({3, 4, 6, 7}), f"MITIGATION_TESTS wrong: {MITIGATION_TESTS}"
    assert CRITICAL_TESTS.isdisjoint(MITIGATION_TESTS), "CRITICAL and MITIGATION sets overlap"


# ─── Gate logic: critical failures block (exit != 0) ─────────────────────────


def test_critical_tests_block_on_fail() -> None:
    """Process exits non-zero if any of tests {1,2,5,8} fails (CONTEXT.md Decision 3B).

    Patches SYNTHETIC_TESTS[0] to return verdict=FAIL (test 1 is critical),
    then calls run_all + exit-code logic in a subprocess to verify exit != 0.
    Patching SYNTHETIC_TESTS[0] directly (not run_test_1_monotone_twin) because
    run_all iterates the list by reference, not by attribute lookup.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, '.'); "
                "import tools.validation_experiments as ve; "
                # Patch SYNTHETIC_TESTS[0] to a function returning FAIL for test 1
                "ve.SYNTHETIC_TESTS[0] = lambda **kw: {'n': 1, 'name': 'T1', "
                "'threshold': 't', 'observed': 'x', 'verdict': 'FAIL', 'mitigation': 'n/a'}; "
                # run_all with workers=1 (no pool, avoids pickling issues)
                "results = ve.run_all(client=None, workers=1); "
                "from pathlib import Path; "
                "ve.write_report(results, Path('/tmp/ve_test_stub.md')); "
                "crit_fails = [r for r in results if r['n'] in ve.CRITICAL_TESTS and r['verdict'] == 'FAIL']; "
                "sys.exit(2 if crit_fails else 0)"
            ),
        ],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0, (
        f"Expected non-zero exit on critical test FAIL, got {result.returncode}.\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
