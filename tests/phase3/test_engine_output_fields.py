from __future__ import annotations

from unittest.mock import MagicMock

from src.decision_engine import engine as eng


def _client_with_fields(field_names: list[str]) -> MagicMock:
    client = MagicMock()
    client.describe_collection.return_value = {"fields": [{"name": n} for n in field_names]}
    return client


def test_action_dist_kept_when_collection_has_it() -> None:
    eng._ENGINE_SCHEMA_FIELDS.clear()
    client = _client_with_fields(
        ["decision_id", "hero_action_type", "confidence", "gto_score", "action_dist"]
    )
    fields = eng._engine_output_fields(client, "postflop_decisions")
    assert "action_dist" in fields


def test_action_dist_dropped_when_collection_lacks_it() -> None:
    eng._ENGINE_SCHEMA_FIELDS.clear()
    client = _client_with_fields(["decision_id", "hero_action_type", "confidence", "gto_score"])
    fields = eng._engine_output_fields(client, "postflop_decisions")
    assert "action_dist" not in fields
