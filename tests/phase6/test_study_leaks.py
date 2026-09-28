"""Unit tests for src/study/leaks.py and src/study/leaks_coverage.py.

Uses MagicMock connections — no live DB needed. Integration tests under
@pytest.mark.integration live in tests/integration/.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.study.leaks import _bucket_from_cluster_key, rank_leaks
from src.study.leaks_coverage import rank_coverage_gaps

_STRATEGY_DESC = [("cluster_key",), ("xbar",), ("n_sessions",), ("source",)]
_OBS_COUNT_DESC = [("cluster_key",), ("n_obs",)]
_COVERAGE_DESC = [
    ("obs_id",),
    ("cluster_key",),
    ("ts",),
    ("session_id",),
    ("action_taken",),
    ("max_neighbor_distance",),
]


def _mock_conn(
    strategy_rows=(),
    obs_rows=(),
    coverage_rows=(),
):
    """Create a mock psycopg connection that dispatches by SQL content.

    The implementations issue three SQL shapes:

    * ``rank_leaks`` strategy branch: ``SELECT m.cluster_key, AVG(m.value)...``
      (returns ``strategy_rows``, columns = _STRATEGY_DESC).
    * ``rank_leaks`` strategy branch (second query): ``SELECT cluster_key, COUNT(*)
      FROM observations GROUP BY cluster_key`` (returns ``obs_rows``,
      columns = _OBS_COUNT_DESC).
    * ``rank_coverage_gaps``: ``SELECT obs_id, cluster_key, ts ... FROM
      observations WHERE flagged_sparse = TRUE`` (returns ``coverage_rows``,
      columns = _COVERAGE_DESC).

    Dispatching by SQL content (instead of call order) lets one mock serve
    both the ``leak_type='coverage'`` path (1 cursor call) and the
    ``leak_type='both'`` path (3 cursor calls) without the test having to
    pre-compute the call count.
    """
    conn = MagicMock()

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        # Defaults pre-execute (avoid AttributeError if a query never runs).
        cm.description = _STRATEGY_DESC
        cm.fetchall.return_value = []
        cm.fetchone.return_value = None
        cm.rowcount = 0

        def execute(sql, params=None):
            # Lower-case both sides for robustness against whitespace/casing diffs.
            s = " ".join(sql.split()).lower()
            if "flagged_sparse" in s:
                cm.description = _COVERAGE_DESC
                cm.fetchall.return_value = list(coverage_rows)
            elif "from observations" in s and "count(*)" in s:
                cm.description = _OBS_COUNT_DESC
                cm.fetchall.return_value = list(obs_rows)
            elif "metric_name = 'ev_loss'" in s or "ev_loss" in s:
                cm.description = _STRATEGY_DESC
                cm.fetchall.return_value = list(strategy_rows)
            return None

        cm.execute.side_effect = execute
        return cm

    conn.cursor.side_effect = cursor_factory
    conn.transaction.return_value.__enter__ = lambda s: s
    conn.transaction.return_value.__exit__ = lambda *a: False
    return conn


def test_bucket_parses_cluster_key():
    ck = "street_class=flop|pot_type=srp|hero_pos_rel=ip|n_players=2"
    assert _bucket_from_cluster_key(ck) == ("flop", "srp")


def test_bucket_unknown_keys_default():
    assert _bucket_from_cluster_key("garbage") == ("?", "?")


def test_rank_leaks_both_returns_keys():
    conn = _mock_conn(
        strategy_rows=[
            ("street_class=flop|pot_type=srp|xx", 0.5, 10, "autoloop"),
            ("street_class=flop|pot_type=srp|yy", 0.3, 10, "manual"),
        ],
        obs_rows=[
            ("street_class=flop|pot_type=srp|xx", 50),
            ("street_class=flop|pot_type=srp|yy", 100),
        ],
        coverage_rows=[],
    )
    result = rank_leaks(leak_type="both", min_n=20, limit=10, _tsdb_conn=conn)
    assert set(result.keys()) >= {"coverage", "strategy", "bucket_stats"}


def test_rank_leaks_strategy_only():
    conn = _mock_conn(
        strategy_rows=[("street_class=flop|pot_type=srp|xx", 0.5, 10, "autoloop")],
        obs_rows=[("street_class=flop|pot_type=srp|xx", 50)],
    )
    result = rank_leaks(leak_type="strategy", _tsdb_conn=conn)
    assert result["coverage"] == []
    assert len(result["strategy"]) == 1
    row = result["strategy"][0]
    assert set(row.keys()) >= {
        "cluster_key",
        "ev_loss",
        "ci_low",
        "ci_high",
        "n_obs",
        "source",
        "stage",
        "shrunk",
        "score",
    }
    assert row["stage"] == "A"  # source='autoloop' -> stage A


def test_rank_leaks_coverage_only():
    conn = _mock_conn(
        strategy_rows=[],
        obs_rows=[],
        coverage_rows=[
            ("obs-1", "ck-1", "2026-05-19T00:00:00Z", "sess-1", "fold", 0.95),
        ],
    )
    result = rank_leaks(leak_type="coverage", _tsdb_conn=conn)
    assert result["strategy"] == []
    assert len(result["coverage"]) == 1
    assert result["coverage"][0]["cluster_key"] == "ck-1"


def test_strategy_excludes_low_n():
    conn = _mock_conn(
        strategy_rows=[("street_class=flop|pot_type=srp|xx", 0.5, 10, "autoloop")],
        obs_rows=[("street_class=flop|pot_type=srp|xx", 5)],  # < min_n=20
    )
    result = rank_leaks(leak_type="strategy", min_n=20, _tsdb_conn=conn)
    assert result["strategy"] == []


def test_strategy_sorted_by_score_desc():
    conn = _mock_conn(
        strategy_rows=[
            ("street_class=flop|pot_type=srp|aaa", 0.2, 10, "autoloop"),
            ("street_class=flop|pot_type=srp|bbb", 0.8, 10, "autoloop"),
            ("street_class=flop|pot_type=srp|ccc", 0.5, 10, "autoloop"),
        ],
        obs_rows=[
            ("street_class=flop|pot_type=srp|aaa", 100),
            ("street_class=flop|pot_type=srp|bbb", 100),
            ("street_class=flop|pot_type=srp|ccc", 100),
        ],
    )
    result = rank_leaks(leak_type="strategy", _tsdb_conn=conn)
    scores = [r["score"] for r in result["strategy"]]
    assert scores == sorted(scores, reverse=True)


def test_solver_source_marks_stage_b():
    conn = _mock_conn(
        strategy_rows=[("street_class=flop|pot_type=srp|xx", 0.5, 10, "solver")],
        obs_rows=[("street_class=flop|pot_type=srp|xx", 50)],
    )
    result = rank_leaks(leak_type="strategy", _tsdb_conn=conn)
    assert result["strategy"][0]["stage"] == "B"


def test_solver_verify_source_marks_stage_b():
    conn = _mock_conn(
        strategy_rows=[("street_class=flop|pot_type=srp|xx", 0.5, 10, "solver_verify")],
        obs_rows=[("street_class=flop|pot_type=srp|xx", 50)],
    )
    result = rank_leaks(leak_type="strategy", _tsdb_conn=conn)
    assert result["strategy"][0]["stage"] == "B"


def test_rank_coverage_gaps_returns_expected_columns():
    conn = _mock_conn(
        strategy_rows=[],
        obs_rows=[],
        coverage_rows=[
            ("obs-1", "ck-1", "2026-05-19T00:00:00Z", "sess-1", "fold", 0.95),
            ("obs-2", "ck-2", "2026-05-19T01:00:00Z", "sess-1", "call", 0.82),
        ],
    )
    rows = rank_coverage_gaps(limit=10, _tsdb_conn=conn)
    assert len(rows) == 2
    expected = {"obs_id", "cluster_key", "ts", "session_id", "action_taken", "max_neighbor_distance"}
    assert set(rows[0].keys()) >= expected


def test_rank_leaks_limit_respected():
    """Top-N limit caps the strategy result list."""
    conn = _mock_conn(
        strategy_rows=[(f"street_class=flop|pot_type=srp|k{i}", 0.5, 10, "autoloop") for i in range(50)],
        obs_rows=[(f"street_class=flop|pot_type=srp|k{i}", 100) for i in range(50)],
    )
    result = rank_leaks(leak_type="strategy", min_n=20, limit=5, _tsdb_conn=conn)
    assert len(result["strategy"]) == 5
