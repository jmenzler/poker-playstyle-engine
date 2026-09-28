"""Regression: _knn_neighbors must request only output_fields the collection has.

The corpus 13-field schema has no cluster_key field; requesting it makes Milvus reject
the whole search (code=65535), so kNN silently returned []. Reconstruct from scalars.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.study.probe as probe_mod

_SPOT = {
    "street": "preflop",
    "hero_hole": ["Ah", "Kh"],
    "hero_position": "BB",
    "action_sequence": ["BTN:open_2.5", "SB:fold"],
}

_SCHEMA_NO_CK = [
    "decision_id",
    "embedding",
    "street_class",
    "pot_type",
    "hero_pos_rel",
    "n_players_active",
    "hero_action_type",
    "gto_score",
]
_SCHEMA_WITH_CK = [*_SCHEMA_NO_CK, "cluster_key"]


def _run_knn(field_names: list[str]) -> list[str]:
    """Run _knn_neighbors against a mock collection whose schema = field_names.
    Returns the output_fields list actually passed to client.search."""
    probe_mod._SCHEMA_FIELD_CACHE.clear()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    client = MagicMock()
    client.describe_collection.return_value = {"fields": [{"name": n} for n in field_names]}
    client.search.return_value = [[]]
    with patch.object(probe_mod, "_encode_embedding_for_spot", return_value=np.zeros(34)):
        probe_mod._knn_neighbors(
            conn,
            client,
            "street_class=preflop|pot_type=srp|hero_pos_rel=IP|n_players_active=2",
            spot=_SPOT,
            k=5,
        )
    assert client.search.called
    return client.search.call_args.kwargs["output_fields"]


def test_cluster_key_omitted_when_absent_from_schema():
    of = _run_knn(_SCHEMA_NO_CK)
    assert "cluster_key" not in of
    assert "decision_id" in of and "hero_action_type" in of


def test_cluster_key_requested_when_present_in_schema():
    of = _run_knn(_SCHEMA_WITH_CK)
    assert "cluster_key" in of


def test_knn_search_passes_engine_hard_filter():
    """kNN must hard-partition (active + cluster_key components), mirroring the engine,
    not do a global COSINE search across mismatched pot_types/streets."""
    probe_mod._SCHEMA_FIELD_CACHE.clear()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    client = MagicMock()
    client.describe_collection.return_value = {"fields": [{"name": n} for n in _SCHEMA_NO_CK]}
    client.search.return_value = [[]]
    with patch.object(probe_mod, "_encode_embedding_for_spot", return_value=np.zeros(34)):
        probe_mod._knn_neighbors(
            conn,
            client,
            "street_class=preflop|pot_type=srp|hero_pos_rel=IP|n_players_active=2",
            spot=_SPOT,
            k=5,
        )
    f = client.search.call_args.kwargs["filter"]
    assert f.startswith("active == True")
    assert 'pot_type == "srp"' in f
    assert 'street_class == "preflop"' in f
    assert 'hero_pos_rel == "IP"' in f
    assert "n_players_active == 2" in f and 'n_players_active == "2"' not in f  # int unquoted
    assert 'street == "preflop"' in f  # spot street band


def test_describe_failure_falls_back_without_cluster_key():
    """If describe_collection throws, search must still run (sans cluster_key)."""
    probe_mod._SCHEMA_FIELD_CACHE.clear()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    client = MagicMock()
    client.describe_collection.side_effect = RuntimeError("milvus down")
    client.search.return_value = [[]]
    with patch.object(probe_mod, "_encode_embedding_for_spot", return_value=np.zeros(34)):
        probe_mod._knn_neighbors(
            conn,
            client,
            "street_class=preflop|pot_type=srp|hero_pos_rel=IP|n_players_active=2",
            spot=_SPOT,
            k=5,
        )
    assert client.search.called
    assert "cluster_key" not in client.search.call_args.kwargs["output_fields"]


def test_neighbor_hand_id_derived_from_decision_id():
    """A corpus hit's hand_id (for replay) comes from its decision_id, not observations."""
    probe_mod._SCHEMA_FIELD_CACHE.clear()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    client = MagicMock()
    client.describe_collection.return_value = {"fields": [{"name": n} for n in _SCHEMA_NO_CK]}
    entity = SimpleNamespace(
        get=lambda k, d=None: {"decision_id": "RC1001_dp3", "hero_action_type": "bet"}.get(k, d)
    )
    client.search.return_value = [[SimpleNamespace(entity=entity, distance=0.4)]]
    with patch.object(probe_mod, "_encode_embedding_for_spot", return_value=np.zeros(80)):
        out = probe_mod._knn_neighbors(
            conn,
            client,
            "street_class=postflop|pot_type=4bet|hero_pos_rel=OOP|n_players_active=2",
            spot={"street": "flop"},
            k=5,
        )
    assert len(out) == 1
    assert out[0]["hand_id"] == "RC1001"
    assert out[0]["decision_id"] == "RC1001_dp3"


_ENGINE_SNAP_SCHEMA = [
    "decision_id",
    "hero_action_type",
    "confidence",
    "gto_score",
    "hero_action_size_pot_frac",
    "raise_ratio",
    "hero_action_allin",
]


def test_engine_output_fields_keeps_snap_when_present():
    from src.decision_engine import engine as engine_mod

    engine_mod._ENGINE_SCHEMA_FIELDS.clear()
    client = MagicMock()
    client.describe_collection.return_value = {"fields": [{"name": n} for n in _ENGINE_SNAP_SCHEMA]}
    of = engine_mod._engine_output_fields(client, "postflop_decisions")
    assert "hero_action_size_pot_frac" in of and "raise_ratio" in of and "hero_action_allin" in of


def test_engine_output_fields_drops_snap_when_collection_lacks_it():
    """A pre-snap collection genuinely missing the size fields is legit: drop them, no raise."""
    from src.decision_engine import engine as engine_mod

    engine_mod._ENGINE_SCHEMA_FIELDS.clear()
    pre_snap = ["decision_id", "hero_action_type", "confidence", "gto_score"]
    client = MagicMock()
    client.describe_collection.return_value = {"fields": [{"name": n} for n in pre_snap]}
    of = engine_mod._engine_output_fields(client, "preflop_decisions")
    assert "hero_action_size_pot_frac" not in of
    assert "decision_id" in of and "hero_action_type" in of


def test_engine_output_fields_raises_on_describe_rpc_error():
    """A transient describe RPC fault must NOT silently drop snap fields (sizing -> defaults)."""
    from src.decision_engine import engine as engine_mod

    engine_mod._ENGINE_SCHEMA_FIELDS.clear()
    client = MagicMock()
    client.describe_collection.side_effect = RuntimeError("milvus down")
    with pytest.raises(RuntimeError, match="milvus down"):
        engine_mod._engine_output_fields(client, "postflop_decisions")
