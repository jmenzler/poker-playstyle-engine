"""Phase 7 / 07-03 — leak_detector honors locked_from_autoloop + leak_suppressions.

Closes BLOCKER 2 / INTG-02 (LOOP-01 + PTCH-01 operator invariant). The current
`_ACTIVE_NODE_QUERY` returns a 2-tuple `(node_id, source)` and ignores the
operator-controlled `locked_from_autoloop` column (migration 011) and active
`leak_suppressions` row (migration 012). After 07-03 the query returns a 4-tuple
`(node_id, source, locked, is_suppressed)` and `select_patch_candidates` skips
clusters whose active node is locked OR whose cluster_key has an active
suppression row.

These three tests pin the new contract; they fail against the current source
(ValueError unpacking 2-tuple → 4-tuple) and pass after Task 2.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src._config import AutoLoopConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Helper: build a mock TSDB connection for leak_detector tests (4-tuple shape).
#
# select_patch_candidates():
#   1. fetchall() — returns leak_rows (ev_loss query results)
#   2. For each leak row:
#       a. fetchone() — obs_count row: (int,)
#       b. fetchone() — active_node row: (uuid_str, source, locked, is_suppressed) or None
#
# fetchone.side_effect is a flat list of length 2*len(leak_rows):
#   [obs_count_for_row0, active_row_for_row0, obs_count_for_row1, ...]
# ---------------------------------------------------------------------------


def _make_leak_detector_tsdb_v2(
    leak_rows: list[tuple],
    fetchone_side_effect: list,
) -> MagicMock:
    """Return a MagicMock psycopg connection for leak_detector tests (4-tuple shape).

    Mirrors `tests/phase5/test_leak_detector.py::_make_leak_detector_tsdb` but the
    active_row entries in `fetchone_side_effect` are 4-tuples:
        (uuid_str, source_str, locked_bool, is_suppressed_bool)
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


@pytest.fixture
def autoloop_config() -> AutoLoopConfig:
    """Local copy of the phase5 fixture — keeps phase 7 tests self-contained."""
    return AutoLoopConfig(
        enabled=True,
        tau_leak=0.3,
        min_observations=5,
        max_patches_per_run=1,
        n_hands_validation=10,
        bootstrap_resamples=10,
        ci_alpha=0.05,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_skips_locked(autoloop_config: AutoLoopConfig) -> None:
    """Active node with locked_from_autoloop=TRUE → cluster skipped."""
    node_id = str(uuid.uuid4())
    leak_rows = [("ck_locked", 0.5)]
    fetchone_side_effect = [
        (20,),  # obs count passes min_observations
        (node_id, "sim", True, False),  # active node locked from autoloop
    ]
    tsdb = _make_leak_detector_tsdb_v2(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert result == []


def test_skips_suppressed(autoloop_config: AutoLoopConfig) -> None:
    """Cluster with active leak_suppressions row → cluster skipped."""
    node_id = str(uuid.uuid4())
    leak_rows = [("ck_suppressed", 0.5)]
    fetchone_side_effect = [
        (20,),
        (node_id, "sim", False, True),  # active suppression for this cluster
    ]
    tsdb = _make_leak_detector_tsdb_v2(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert result == []


def test_passes_when_unlocked_unsuppressed(autoloop_config: AutoLoopConfig) -> None:
    """Active node, source!='seed', not locked, not suppressed → cluster returned."""
    node_id = str(uuid.uuid4())
    leak_rows = [("ck_ok", 0.6)]
    fetchone_side_effect = [
        (20,),
        (node_id, "sim", False, False),
    ]
    tsdb = _make_leak_detector_tsdb_v2(leak_rows, fetchone_side_effect)

    from src.autoloop.leak_detector import ClusterCandidate, select_patch_candidates

    result = select_patch_candidates(
        session_id="s1",
        config=autoloop_config,
        _tsdb_conn=tsdb,
    )
    assert len(result) == 1
    candidate = result[0]
    assert isinstance(candidate, ClusterCandidate)
    assert candidate.cluster_key == "ck_ok"
    assert candidate.prev_node_id == uuid.UUID(node_id)
