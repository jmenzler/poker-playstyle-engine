"""Regression test for D-13b: probe._knn_neighbors() must z-score normalize the  long-ok
embedding before passing it to the Milvus client.search() call.

The Milvus corpus was upserted with z-score normalized embeddings (tools/upsert_milvus.py).
Passing a raw [0,1] min-maxed embedding makes COSINE distance meaningless — silent
retrieval corruption.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.study.probe import _knn_neighbors


def _raw_embedding(dim: int = 10) -> list[float]:
    """Return a raw [0,1] normalized embedding (all values in [0,1])."""
    return [i / (dim - 1) for i in range(dim)]


def _make_mock_conn(embedding: list[float]) -> MagicMock:
    """Mock psycopg conn: first query (strategy_nodes embedding) returns the row;
    subsequent in-loop lookups return None so the neighbor takes the fallback path
    (the loop's strategy_node lookup expects a 2-tuple, not this embedding row)."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    calls = {"n": 0}

    def _fetchone():
        calls["n"] += 1
        return (embedding,) if calls["n"] == 1 else None

    cursor.fetchone.side_effect = _fetchone
    conn.cursor.return_value = cursor
    return conn


def _make_mock_client() -> tuple[MagicMock, list]:
    """Return (client, captured_data) where captured_data collects search() data= args."""
    captured: list = []
    client = MagicMock()

    def _search(**kwargs):
        captured.append(kwargs.get("data", []))
        hit = SimpleNamespace(
            entity=SimpleNamespace(get=lambda k, d=None: "other_cluster" if k == "cluster_key" else d),
            distance=0.1,
        )
        return [[hit]]

    client.search.side_effect = _search
    return client, captured


def _stub_manifest(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, std, weights) arrays with non-trivial values."""
    return (
        np.array([0.5] * dim),
        np.array([0.2] * dim),
        np.array([1.0] * dim),
    )


def test_probe_zscore_normalizes_before_search() -> None:
    """D-13b: vector passed to client.search() must be z-score normalized.  long-ok

    A raw [0,1] embedding under mean=0.5, std=0.2, weights=1.0 maps to [-2.5, 2.5]
    — all outside [0,1]. We assert the searched vector differs from raw input and
    contains at least one value outside [0,1].
    """
    dim = 10
    raw = _raw_embedding(dim)
    conn = _make_mock_conn(raw)
    client, captured = _make_mock_client()

    cluster_key = "street_class=preflop|pot_type=srp|hero_pos_rel=IP|n_players=2"
    mean, std, weights = _stub_manifest(dim)

    with patch("src.study.probe._load_zscore_manifest", return_value=(mean, std, weights), create=True):
        _knn_neighbors(conn, client, cluster_key, k=5)

    assert captured, "client.search() was never called — _knn_neighbors returned early"

    searched_vector = captured[0][0]

    assert searched_vector != raw, (
        "client.search() received the RAW embedding unchanged — z-score was NOT applied"
    )
    assert any(v < 0.0 or v > 1.0 for v in searched_vector), (
        "All searched values are in [0,1] — z-score normalization was not applied "
        f"(searched_vector[:5]={searched_vector[:5]})"
    )
