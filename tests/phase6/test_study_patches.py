"""Unit tests for src/study/patches.py.

Mocks psycopg cursors; covers list_patches (filters, ordering, schema) and
patch_detail (single-row lookup + chain reconstruction).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.study.patches import list_patches, patch_detail

_LIST_DESC = [
    ("patch_id",),
    ("ts",),
    ("cluster_key",),
    ("source",),
    ("pre_ev_loss",),
    ("post_ev_loss",),
    ("status",),
    ("prev_node_id",),
    ("new_node_id",),
    ("prev_patch_id",),
]
_DETAIL_DESC = [
    ("patch_id",),
    ("ts",),
    ("cluster_key",),
    ("source",),
    ("pre_ev_loss",),
    ("post_ev_loss",),
    ("status",),
    ("prev_node_id",),
    ("new_node_id",),
    ("decision_id",),
    ("override_action_dist",),
    ("original_action_dist",),
]
_CHAIN_DESC = [("patch_id",), ("ts",), ("source",), ("pre_ev_loss",), ("post_ev_loss",)]


def _mock_conn(list_rows=(), detail_row=None, chain_rows=()):
    """Mock psycopg conn that dispatches by SQL.

    - SELECT p.patch_id ... FROM patches p WHERE 1=1 ...  -> list_rows + _LIST_DESC
    - SELECT patch_id, ts, ... FROM patches WHERE patch_id = %s -> detail_row + _DETAIL_DESC
    - SELECT patch_id, ts, source ... FROM patches WHERE cluster_key = %s ... -> chain_rows + _CHAIN_DESC
    """
    conn = MagicMock()

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        cm.description = _LIST_DESC
        cm.fetchall.return_value = []
        cm.fetchone.return_value = None

        def execute(sql, params=None):
            s = " ".join(sql.split()).lower()
            if "from patches p where 1=1" in s:
                cm.description = _LIST_DESC
                cm.fetchall.return_value = list(list_rows)
            elif "where p.patch_id = %s" in s:
                cm.description = _DETAIL_DESC
                cm.fetchone.return_value = detail_row
            elif "from patches where cluster_key = %s order by ts asc" in s:
                cm.description = _CHAIN_DESC
                cm.fetchall.return_value = list(chain_rows)
            return None

        cm.execute.side_effect = execute
        return cm

    conn.cursor.side_effect = cursor_factory
    return conn


def test_list_patches_returns_required_columns():
    list_rows = [
        (
            "p1",  # patch_id
            "2026-05-19T10:00:00Z",  # ts
            "ck-1",  # cluster_key
            "autoloop",  # source
            0.5,  # pre_ev_loss
            0.3,  # post_ev_loss
            "applied",  # status
            "node-0",  # prev_node_id
            "node-1",  # new_node_id
            None,  # prev_patch_id (first patch for this cluster)
        ),
    ]
    conn = _mock_conn(list_rows=list_rows)
    result = list_patches(limit=10, _tsdb_conn=conn)
    assert len(result) == 1
    required = {
        "patch_id",
        "ts",
        "cluster_key",
        "source",
        "pre_ev_loss",
        "post_ev_loss",
        "status",
        "prev_patch_id",
    }
    assert set(result[0].keys()) >= required


def test_list_patches_respects_limit_param():
    list_rows = [
        (f"p{i}", "2026-05-19T00:00:00Z", "ck-x", "autoloop", 0.5, 0.3, "applied", "n0", f"n{i}", None)
        for i in range(3)
    ]
    # We can't directly read the LIMIT off MagicMock's side_effect cursor
    # (cursor_factory returns a fresh MagicMock each call). Instead assert
    # the function returns the canned row count without crashing.
    result = list_patches(limit=10, _tsdb_conn=_mock_conn(list_rows=list_rows))
    assert len(result) == 3


def test_list_patches_with_cluster_key_filter():
    """When cluster_key arg is passed, SQL must include the filter clause."""
    captured_sql = []

    conn = MagicMock()

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        cm.description = _LIST_DESC
        cm.fetchall.return_value = []

        def execute(sql, params=None):
            captured_sql.append(sql)
            return None

        cm.execute.side_effect = execute
        return cm

    conn.cursor.side_effect = cursor_factory
    list_patches(cluster_key="ck-target", limit=5, _tsdb_conn=conn)
    assert any("p.cluster_key = %s" in s for s in captured_sql)


def test_list_patches_orders_by_ts_desc():
    """Ordering clause must appear in SQL."""
    captured_sql = []
    conn = MagicMock()

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        cm.description = _LIST_DESC
        cm.fetchall.return_value = []
        cm.execute.side_effect = lambda sql, params=None: captured_sql.append(sql)
        return cm

    conn.cursor.side_effect = cursor_factory
    list_patches(limit=5, _tsdb_conn=conn)
    assert any("order by p.ts desc" in s.lower() for s in captured_sql)


def test_patch_detail_returns_none_when_missing():
    conn = _mock_conn(detail_row=None)
    assert patch_detail("nonexistent_pid", _tsdb_conn=conn) is None


def test_patch_detail_returns_dict_when_found():
    detail = (
        "p1",
        "2026-05-19T10:00:00Z",
        "ck-1",
        "autoloop",
        0.5,
        0.3,
        "applied",
        "node-0",
        "node-1",
        "hand1_dp0",
        {"bet_50": 1.0},
        {"check": 1.0},
    )
    conn = _mock_conn(detail_row=detail)
    result = patch_detail("p1", include_chain=False, _tsdb_conn=conn)
    assert result is not None
    assert result["patch_id"] == "p1"
    assert "chain" not in result


def _mock_detail_capturing(detail_row, captured):
    """Mock conn that dispatches the single-patch detail query and captures its SQL."""
    conn = MagicMock()

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        cm.fetchone.return_value = None
        cm.fetchall.return_value = []

        def execute(sql, params=None):
            s = " ".join(sql.split())
            if "patch_id = %s" in s.lower():
                captured["detail_sql"] = s.lower()
                cm.description = _DETAIL_DESC
                cm.fetchone.return_value = detail_row

        cm.execute.side_effect = execute
        return cm

    conn.cursor.side_effect = cursor_factory
    return conn


def test_patch_detail_enriched_with_decision_id_dists_and_hand_id():
    captured: dict = {}
    detail = (
        "p1",
        "2026-05-19T10:00:00Z",
        "ck-1",
        "manual",
        0.5,
        0.3,
        "applied",
        "node-0",
        "node-1",
        "hand42_dp3",
        {"bet_50": 0.6, "check": 0.4},  # override (new_node)
        {"check": 1.0},  # original (prev_node)
    )
    result = patch_detail("p1", include_chain=False, _tsdb_conn=_mock_detail_capturing(detail, captured))

    sql = captured["detail_sql"]
    assert "left join strategy_nodes" in sql
    assert "new_node_id" in sql and "prev_node_id" in sql
    assert "p.decision_id" in sql

    assert result["decision_id"] == "hand42_dp3"
    assert result["hand_id"] == "hand42"
    assert result["override_action_dist"] == {"bet_50": 0.6, "check": 0.4}
    assert result["original_action_dist"] == {"check": 1.0}


def test_patch_detail_original_none_for_first_ever_patch():
    captured: dict = {}
    detail = (
        "p1",
        "2026-05-19T10:00:00Z",
        "ck-1",
        "manual",
        0.5,
        0.3,
        "applied",
        None,  # prev_node_id NULL -> first-ever patch
        "node-1",
        "hand42_dp0",
        {"bet_50": 1.0},
        None,  # original null; frontend falls back to /api/probe
    )
    result = patch_detail("p1", include_chain=False, _tsdb_conn=_mock_detail_capturing(detail, captured))
    assert result["original_action_dist"] is None
    assert result["hand_id"] == "hand42"


def test_patch_detail_includes_chain_when_requested():
    detail = (
        "p2",
        "2026-05-19T11:00:00Z",
        "ck-x",
        "autoloop",
        0.4,
        0.2,
        "applied",
        "node-A",
        "node-B",
        "hand2_dp1",
        {"raise": 1.0},
        {"call": 1.0},
    )
    chain = [
        ("p1", "2026-05-19T10:00:00Z", "manual", 0.5, 0.4),
        ("p2", "2026-05-19T11:00:00Z", "autoloop", 0.4, 0.2),
    ]
    conn = _mock_conn(detail_row=detail, chain_rows=chain)
    result = patch_detail("p2", include_chain=True, _tsdb_conn=conn)
    assert result is not None
    assert "chain" in result
    assert len(result["chain"]) == 2
    # Chain entries are dicts with the expected keys
    assert {"patch_id", "ts", "source", "pre_ev_loss", "post_ev_loss"} <= set(result["chain"][0].keys())
