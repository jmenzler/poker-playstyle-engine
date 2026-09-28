"""tests/phase5/test_leak_detector.py — Unit tests for leak_detector.py (Plan 06).

Requirements covered:
- LOOP-03: candidate ranking and filtering — rank by ev_loss x log(n_obs);
           skip source='seed' nodes; min_observations guard
"""

from __future__ import annotations

import math
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Helper: build a mock TSDB connection for leak_detector tests.
#
# select_patch_candidates():
#   1. fetchall() — returns leak_rows (ev_loss query results)
#   2. For each leak row:
#       a. fetchone() — returns obs_count row (n_obs)
#       b. fetchone() — returns active_node row
#          (uuid_str, source_str, locked_bool, is_suppressed_bool) or None
#
# So fetchone.side_effect is a flat list of length 2*len(leak_rows):
#   [obs_count_for_row0, active_row_for_row0, obs_count_for_row1, ...]
#
# Active-row shape was extended from 2-tuple to 4-tuple in Phase 7 / 07-03
# (_ACTIVE_NODE_QUERY now returns locked_from_autoloop + is_suppressed). The
# happy paths exercised in this file always pass locked=False, is_suppressed=False.
# ---------------------------------------------------------------------------


def _make_leak_detector_tsdb(
    leak_rows: list[tuple],
    fetchone_side_effect: list,
) -> MagicMock:
    """Return a MagicMock psycopg connection for leak_detector tests.

    Args:
        leak_rows: rows returned by the ev_loss fetchall() query.
            Each row: (cluster_key, ev_loss_value)
        fetchone_side_effect: flat list of fetchone() return values.
            Order: [obs_count_row0, active_row_row0, obs_count_row1, ...]
            obs_count_row: (int,) — total observation count
            active_row: (uuid_str, source_str, locked_bool, is_suppressed_bool) or None
    """
    conn = MagicMock()
    mock_cur = MagicMock()

    mock_cur.fetchall.return_value = leak_rows
    mock_cur.fetchone.side_effect = fetchone_side_effect

    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.transaction.return_value.__enter__ = MagicMock(return_value=conn)
    conn.transaction.return_value.__exit__ = MagicMock(return_value=False)

    return conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_filters_below_tau_leak(autoloop_config: MagicMock) -> None:
    """SQL WHERE already filters; returning [] from fetchall means no candidates."""
    # leak_rows empty — the ev_loss query WHERE clause applies tau_leak in SQL
    tsdb = _make_leak_detector_tsdb(leak_rows=[], fetchone_side_effect=[])

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert result == []


def test_filters_below_min_observations(autoloop_config: MagicMock) -> None:
    """Cluster with n_obs < min_observations is filtered out."""
    # autoloop_config has min_observations=5 in conftest
    leak_rows = [("ck_low_obs", 0.5)]
    # obs count = 2 (< 5 min_observations); active node query never reached
    fetchone_side_effect = [
        (2,),  # obs count for ck_low_obs
        None,  # active node — won't be reached but must be in list for safety
    ]
    tsdb = _make_leak_detector_tsdb(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert result == []


def test_filters_seed_source_cluster(autoloop_config: MagicMock) -> None:
    """Cluster with active node source='seed' is filtered out (PTCH-04 pre-check)."""
    seed_uuid = str(uuid.uuid4())
    leak_rows = [("ck_seed", 0.5)]
    fetchone_side_effect = [
        (20,),  # obs count passes min_observations
        (seed_uuid, "seed", False, False),  # active node has source='seed' -> skip
    ]
    tsdb = _make_leak_detector_tsdb(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert result == []


def test_passes_when_no_active_node(autoloop_config: MagicMock) -> None:
    """When no active node exists, candidate passes with prev_node_id=None."""
    leak_rows = [("ck_no_node", 0.5)]
    fetchone_side_effect = [
        (20,),  # obs count passes
        None,  # no active node
    ]
    tsdb = _make_leak_detector_tsdb(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import ClusterCandidate, select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert len(result) == 1
    candidate = result[0]
    assert isinstance(candidate, ClusterCandidate)
    assert candidate.cluster_key == "ck_no_node"
    assert candidate.prev_node_id is None


def test_passes_when_active_node_non_seed(autoloop_config: MagicMock) -> None:
    """When active node has source!='seed', candidate passes with prev_node_id set."""
    node_id = str(uuid.uuid4())
    leak_rows = [("ck_sim_node", 0.6)]
    fetchone_side_effect = [
        (20,),  # obs count passes
        (node_id, "sim", False, False),  # active node with source='sim' (not seed)
    ]
    tsdb = _make_leak_detector_tsdb(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert len(result) == 1
    candidate = result[0]
    assert candidate.cluster_key == "ck_sim_node"
    assert candidate.prev_node_id == uuid.UUID(node_id)


def test_ranks_by_score_descending(autoloop_config: MagicMock) -> None:
    """Three clusters ranked by score = ev_loss * log(max(n_obs, 2)) descending."""
    # Configure autoloop_config to allow all 3 candidates
    from src._config import AutoLoopConfig

    cfg = AutoLoopConfig(
        enabled=True,
        tau_leak=0.1,
        min_observations=5,
        max_patches_per_run=10,
    )

    # cluster_a: ev_loss=0.8, n_obs=10 -> score = 0.8 * log(10) ~ 1.842
    # cluster_b: ev_loss=1.0, n_obs=5  -> score = 1.0 * log(5)  ~ 1.609
    # cluster_c: ev_loss=0.5, n_obs=30 -> score = 0.5 * log(30) ~ 1.700
    # Expected order: a (1.842) > c (1.700) > b (1.609)
    leak_rows = [
        ("cluster_a", 0.8),
        ("cluster_b", 1.0),
        ("cluster_c", 0.5),
    ]
    fetchone_side_effect = [
        (10,),
        None,  # cluster_a: 10 obs, no active node
        (5,),
        None,  # cluster_b: 5 obs, no active node
        (30,),
        None,  # cluster_c: 30 obs, no active node
    ]
    tsdb = _make_leak_detector_tsdb(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(session_id="s1", config=cfg, _tsdb_conn=tsdb)
    assert len(result) == 3
    assert result[0].cluster_key == "cluster_a"
    assert result[1].cluster_key == "cluster_c"
    assert result[2].cluster_key == "cluster_b"
    # Verify scores are actually descending
    assert result[0].score > result[1].score > result[2].score


def test_truncates_to_max_patches_per_run(autoloop_config: MagicMock) -> None:
    """10 candidates pass all filters; max_patches_per_run=5 truncates to 5."""
    from src._config import AutoLoopConfig

    cfg = AutoLoopConfig(
        enabled=True,
        tau_leak=0.1,
        min_observations=5,
        max_patches_per_run=5,
    )

    n = 10
    leak_rows = [(f"ck_{i}", 0.5 + i * 0.1) for i in range(n)]
    # Each row: obs_count=20, no active node
    fetchone_side_effect: list = []
    for _ in range(n):
        fetchone_side_effect.append((20,))
        fetchone_side_effect.append(None)

    tsdb = _make_leak_detector_tsdb(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(session_id="s1", config=cfg, _tsdb_conn=tsdb)
    assert len(result) == 5
    # First element should have the highest score
    for i in range(len(result) - 1):
        assert result[i].score >= result[i + 1].score


def test_no_candidates_returns_empty_list(autoloop_config: MagicMock) -> None:
    """When EV loss query returns empty, result is an empty list."""
    tsdb = _make_leak_detector_tsdb(leak_rows=[], fetchone_side_effect=[])

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert result == []


def test_score_formula_with_min_n_obs(autoloop_config: MagicMock) -> None:
    """score = ev_loss * log(max(n_obs, 2)); verified numerically."""
    ev = 0.5
    n_obs = 20
    expected_score = ev * math.log(20)  # log(max(20, 2)) = log(20)

    leak_rows = [("ck_formula", ev)]
    fetchone_side_effect = [
        (n_obs,),  # obs count
        None,  # no active node
    ]
    tsdb = _make_leak_detector_tsdb(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert len(result) == 1
    assert abs(result[0].score - expected_score) < 1e-9
