"""Phase 9 / eval suite orchestrator — pass/fail aggregation (unit)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_loo_result(passed: bool) -> MagicMock:
    r = MagicMock()
    r.passed = passed
    r.tier1_tvd = 0.05
    r.tier1_top1 = 0.80
    r.n_clusters = 5
    r.coverage_ok = True
    r.regression_delta = 0.0
    r.last_run_tvd = 0.05
    return r


def _make_expl_result(exploitability_mean: float = 2.5) -> MagicMock:
    r = MagicMock()
    r.exploitability_mean = exploitability_mean
    r.n_solved_clusters = 3
    r.coverage_pct = 0.8
    r.report_only = True
    return r


def _make_match_result(ci_low: float, ci_high: float, status: str = "won") -> MagicMock:
    r = MagicMock()
    r.ci_low = ci_low
    r.ci_high = ci_high
    r.status = status
    r.bb_per_100 = 10.0
    r.opponent = "random"
    r.hands = 100
    r.seed = 42
    r.match_id = "test-id"
    return r


def test_suite_pass_fail():
    """Suite FAILs if Tier-1 LOO regresses OR any match-play bb/100 CI <= 0 (D-09-15)."""
    from src.eval.suite import EvalSuiteResult, run_suite

    # --- Case 1: Tier-1 fails → suite fails ---
    with (
        patch("src.eval.suite.run_loo_gate") as mock_loo,
        patch("src.eval.suite.aggregate_exploitability") as mock_expl,
        patch("src.eval.suite.run_match") as mock_match,
        patch("src.eval.suite.load_eval_config") as mock_cfg,
        patch("src.eval.suite._persist_suite_metrics"),
        patch("src.eval.suite._write_artifact"),
    ):
        mock_cfg.return_value = MagicMock(
            match_hands_hu=10,
            match_hands_6max=10,
            tvd_floor=0.15,
            tvd_regression_max_delta=0.05,
        )
        mock_loo.return_value = _make_loo_result(passed=False)
        mock_expl.return_value = _make_expl_result()
        mock_match.return_value = _make_match_result(ci_low=1.0, ci_high=5.0, status="won")

        result = run_suite(hands=10, seed=42, _tsdb_conn=None, _milvus=None)
        assert isinstance(result, EvalSuiteResult)
        assert result.tier1_pass is False
        assert result.suite_pass is False

    # --- Case 2: Any match CI <= 0 → suite fails ---
    with (
        patch("src.eval.suite.run_loo_gate") as mock_loo,
        patch("src.eval.suite.aggregate_exploitability") as mock_expl,
        patch("src.eval.suite.run_match") as mock_match,
        patch("src.eval.suite.load_eval_config") as mock_cfg,
        patch("src.eval.suite._persist_suite_metrics"),
        patch("src.eval.suite._write_artifact"),
    ):
        mock_cfg.return_value = MagicMock(
            match_hands_hu=10,
            match_hands_6max=10,
            tvd_floor=0.15,
            tvd_regression_max_delta=0.05,
        )
        mock_loo.return_value = _make_loo_result(passed=True)
        mock_expl.return_value = _make_expl_result()
        # First match is losing (ci_low <= 0) → match_pass=False
        mock_match.return_value = _make_match_result(ci_low=-2.0, ci_high=1.0, status="inconclusive")

        result = run_suite(hands=10, seed=42, _tsdb_conn=None, _milvus=None)
        assert result.tier1_pass is True
        assert result.match_pass is False
        assert result.suite_pass is False

    # --- Case 3: Tier-2 high exploitability alone → suite still PASS (report-only) ---
    with (
        patch("src.eval.suite.run_loo_gate") as mock_loo,
        patch("src.eval.suite.aggregate_exploitability") as mock_expl,
        patch("src.eval.suite.run_match") as mock_match,
        patch("src.eval.suite.load_eval_config") as mock_cfg,
        patch("src.eval.suite._persist_suite_metrics"),
        patch("src.eval.suite._write_artifact"),
    ):
        mock_cfg.return_value = MagicMock(
            match_hands_hu=10,
            match_hands_6max=10,
            tvd_floor=0.15,
            tvd_regression_max_delta=0.05,
        )
        mock_loo.return_value = _make_loo_result(passed=True)
        mock_expl.return_value = _make_expl_result(exploitability_mean=99.0)  # very high
        mock_match.return_value = _make_match_result(ci_low=1.0, ci_high=5.0, status="won")

        result = run_suite(hands=10, seed=42, _tsdb_conn=None, _milvus=None)
        assert result.tier2_pass is True  # always True
        assert result.suite_pass is True  # tier-2 doesn't gate

    # --- Case 4: All pass → suite pass ---
    with (
        patch("src.eval.suite.run_loo_gate") as mock_loo,
        patch("src.eval.suite.aggregate_exploitability") as mock_expl,
        patch("src.eval.suite.run_match") as mock_match,
        patch("src.eval.suite.load_eval_config") as mock_cfg,
        patch("src.eval.suite._persist_suite_metrics"),
        patch("src.eval.suite._write_artifact"),
    ):
        mock_cfg.return_value = MagicMock(
            match_hands_hu=10,
            match_hands_6max=10,
            tvd_floor=0.15,
            tvd_regression_max_delta=0.05,
        )
        mock_loo.return_value = _make_loo_result(passed=True)
        mock_expl.return_value = _make_expl_result()
        mock_match.return_value = _make_match_result(ci_low=1.0, ci_high=5.0, status="won")

        result = run_suite(hands=10, seed=42, _tsdb_conn=None, _milvus=None)
        assert result.tier1_pass is True
        assert result.match_pass is True
        assert result.suite_pass is True


def _failing_conn() -> MagicMock:
    """A TSDB conn stub whose transaction() raises, simulating a metrics-write failure."""
    conn = MagicMock()
    conn.transaction.side_effect = RuntimeError("db down")
    return conn


def test_persist_suite_metrics_returns_false_on_write_failure():
    from datetime import UTC, datetime

    from src.eval.suite import _persist_suite_metrics

    ok = _persist_suite_metrics(
        suite_session_id="s1",
        run_start=datetime(2026, 1, 1, tzinfo=UTC),
        tvd_loo=0.05,
        exploitability_pct=2.5,
        _tsdb_conn=_failing_conn(),
    )
    assert ok is False


def test_suite_flags_metrics_not_persisted_on_write_failure():
    """A swallowed metrics-write failure must surface as metrics_persisted=False, not as suite_pass alone."""
    from src.eval.suite import run_suite

    with (
        patch("src.eval.suite.run_loo_gate") as mock_loo,
        patch("src.eval.suite.aggregate_exploitability") as mock_expl,
        patch("src.eval.suite.run_match") as mock_match,
        patch("src.eval.suite.load_eval_config") as mock_cfg,
        patch("src.eval.suite._write_artifact"),
    ):
        mock_cfg.return_value = MagicMock(
            match_hands_hu=10,
            match_hands_6max=10,
            tvd_floor=0.15,
            tvd_regression_max_delta=0.05,
        )
        mock_loo.return_value = _make_loo_result(passed=True)
        mock_expl.return_value = _make_expl_result()
        mock_match.return_value = _make_match_result(ci_low=1.0, ci_high=5.0, status="won")

        result = run_suite(hands=10, seed=42, _tsdb_conn=_failing_conn(), _milvus=None)
        assert result.suite_pass is True
        assert result.metrics_persisted is False
