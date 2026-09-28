"""Unit tests for src/study/similar.py.

Mocks both psycopg connection and pymilvus MilvusClient — no live DB.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.study.similar import find_similar


def _mock_tsdb(embedding=None, n_obs_per_cluster=42, action_dist=None):
    """Mock psycopg connection.

    Returns ``embedding`` from the strategy_nodes lookup, ``n_obs_per_cluster``
    for every observations COUNT, and ``action_dist`` for every action_dist
    lookup.
    """
    conn = MagicMock()

    def cursor_factory():
        cm = MagicMock()
        cm.__enter__ = lambda self: cm
        cm.__exit__ = lambda *a: False
        cm.description = []
        cm.fetchall.return_value = []
        cm.fetchone.return_value = None

        def execute(sql, params=None):
            s = " ".join(sql.split()).lower()
            if "select embedding from strategy_nodes" in s:
                cm.fetchone.return_value = (embedding,) if embedding is not None else None
            elif "count(*) from observations" in s:
                cm.fetchone.return_value = (n_obs_per_cluster,)
            elif "select action_dist from strategy_nodes" in s:
                cm.fetchone.return_value = (action_dist,) if action_dist is not None else None
            return None

        cm.execute.side_effect = execute
        return cm

    conn.cursor.side_effect = cursor_factory
    return conn


def _hit(cluster_key, distance, source="autoloop"):
    """Build a pymilvus-style hit object.

    pymilvus 3.x hits expose ``hit.distance`` and ``hit.entity.get(field)``.
    Use a MagicMock for ``entity`` so we can attach an arbitrary ``.get``
    method (real ``dict.get`` is read-only on built-in dict instances).
    """
    hit = MagicMock()
    hit.distance = distance
    fields = {"cluster_key": cluster_key, "source": source}
    entity = MagicMock()
    entity.get = lambda k, default=None: fields.get(k, default)
    hit.entity = entity
    return hit


def _mock_milvus(hits):
    """Mock pymilvus MilvusClient where ``search`` returns ``[hits]``."""
    client = MagicMock()
    client.search.return_value = [hits]
    return client


def test_find_similar_returns_expected_row_shape():
    source_ck = "street_class=flop|pot_type=srp|x"
    embedding = [0.1] * 80
    hits = [
        _hit("street_class=flop|pot_type=srp|a", 0.05),
        _hit("street_class=flop|pot_type=srp|b", 0.10),
        _hit("street_class=flop|pot_type=srp|c", 0.15),
    ]
    conn = _mock_tsdb(embedding=embedding)
    client = _mock_milvus(hits)

    result = find_similar(source_ck, k=3, _tsdb_conn=conn, _milvus=client)
    assert len(result) == 3
    expected = {"rank", "cluster_key", "distance", "source", "n_obs"}
    assert set(result[0].keys()) >= expected
    # Ranks are 1-indexed and contiguous.
    assert [r["rank"] for r in result] == [1, 2, 3]


def test_find_similar_dispatches_postflop_collection():
    source_ck = "street_class=flop|pot_type=srp|x"
    embedding = [0.1] * 80
    hits = [_hit("street_class=flop|pot_type=srp|a", 0.05)]
    conn = _mock_tsdb(embedding=embedding)
    client = _mock_milvus(hits)

    find_similar(source_ck, k=1, _tsdb_conn=conn, _milvus=client)
    args = client.search.call_args
    assert args.kwargs.get("collection_name") == "postflop_decisions"


def test_find_similar_dispatches_preflop_collection():
    source_ck = "street_class=preflop|pot_type=open|x"
    embedding = [0.1] * 32
    hits = [_hit("street_class=preflop|pot_type=open|a", 0.05)]
    conn = _mock_tsdb(embedding=embedding)
    client = _mock_milvus(hits)

    find_similar(source_ck, k=1, _tsdb_conn=conn, _milvus=client)
    args = client.search.call_args
    assert args.kwargs.get("collection_name") == "preflop_decisions"


def test_find_similar_skips_source_cluster_in_results():
    source_ck = "street_class=flop|pot_type=srp|self"
    embedding = [0.1] * 80
    # Source cluster comes back as first hit (distance 0) — must be filtered.
    hits = [
        _hit(source_ck, 0.0),
        _hit("street_class=flop|pot_type=srp|a", 0.05),
        _hit("street_class=flop|pot_type=srp|b", 0.10),
    ]
    conn = _mock_tsdb(embedding=embedding)
    client = _mock_milvus(hits)

    result = find_similar(source_ck, k=2, _tsdb_conn=conn, _milvus=client)
    assert all(r["cluster_key"] != source_ck for r in result)
    assert len(result) == 2


def test_find_similar_returns_empty_when_no_embedding():
    """If strategy_nodes lookup returns None, find_similar yields []."""
    conn = _mock_tsdb(embedding=None)
    client = _mock_milvus([])

    result = find_similar("nonexistent_ck", k=5, _tsdb_conn=conn, _milvus=client)
    assert result == []


def test_find_similar_includes_action_dist_when_requested():
    source_ck = "street_class=flop|pot_type=srp|x"
    embedding = [0.1] * 80
    hits = [_hit("street_class=flop|pot_type=srp|a", 0.05)]
    canned_dist = {"fold": 0.3, "call": 0.4, "bet_50": 0.3}
    conn = _mock_tsdb(embedding=embedding, action_dist=canned_dist)
    client = _mock_milvus(hits)

    result = find_similar(source_ck, k=1, include_action_dist=True, _tsdb_conn=conn, _milvus=client)
    assert "action_dist" in result[0]
    assert result[0]["action_dist"] == canned_dist


def test_find_similar_excludes_action_dist_by_default():
    source_ck = "street_class=flop|pot_type=srp|x"
    embedding = [0.1] * 80
    hits = [_hit("street_class=flop|pot_type=srp|a", 0.05)]
    conn = _mock_tsdb(embedding=embedding)
    client = _mock_milvus(hits)

    result = find_similar(source_ck, k=1, _tsdb_conn=conn, _milvus=client)
    assert "action_dist" not in result[0]


def test_find_similar_truncates_to_k_when_more_hits():
    source_ck = "street_class=flop|pot_type=srp|x"
    embedding = [0.1] * 80
    # 7 hits returned by Milvus when k=3 (since we ask for k+1=4 internally,
    # extra hits MUST be truncated).
    hits = [_hit(f"street_class=flop|pot_type=srp|c{i}", 0.05 + i * 0.01) for i in range(7)]
    conn = _mock_tsdb(embedding=embedding)
    client = _mock_milvus(hits)

    result = find_similar(source_ck, k=3, _tsdb_conn=conn, _milvus=client)
    assert len(result) == 3
