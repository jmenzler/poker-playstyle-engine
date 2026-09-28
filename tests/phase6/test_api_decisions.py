"""Tests for src/api/decisions.py — scalar filter allowlist + expr building.

The Milvus client is injected via a FastAPI dependency override, so these run
mock-free against the real app object without a live Milvus.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _FakeMilvus:
    """Records the filter expr it was queried with; returns no rows."""

    def __init__(self) -> None:
        self.last_filter: str | None = None

    def query(self, *, collection_name, filter, limit, offset, output_fields):
        self.last_filter = filter
        return []

    def get_collection_stats(self, *, collection_name):
        return {"row_count": 0}


@pytest.fixture
def fake_milvus() -> _FakeMilvus:
    return _FakeMilvus()


@pytest.fixture
def client(fake_milvus: _FakeMilvus) -> TestClient:
    from src.api.deps import get_milvus
    from src.api.main import app

    app.dependency_overrides[get_milvus] = lambda: fake_milvus
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_milvus, None)


def test_valid_filters_build_expr(client: TestClient, fake_milvus: _FakeMilvus) -> None:
    r = client.get(
        "/api/decisions/sample",
        params={"street": "flop", "pot_type": "srp", "hero_pos_rel": "IP"},
    )
    assert r.status_code == 200
    expr = fake_milvus.last_filter
    assert "street == 'flop'" in expr
    assert "pot_type == 'srp'" in expr
    assert "hero_pos_rel == 'IP'" in expr


@pytest.mark.parametrize(
    "field,value",
    [
        ("street", "preflop'; drop"),
        ("street", "midnight"),
        ("pot_type", "6bet"),
        ("hero_pos_rel", "BTN"),
        ("hero_action_type", "rm -rf"),
    ],
)
def test_unknown_filter_value_rejected(
    client: TestClient, fake_milvus: _FakeMilvus, field: str, value: str
) -> None:
    """Out-of-vocabulary params are rejected with 422 before reaching Milvus."""
    r = client.get("/api/decisions/sample", params={field: value})
    assert r.status_code == 422
    assert fake_milvus.last_filter is None  # never queried Milvus


def test_injection_attempt_via_filter_value_rejected(client: TestClient, fake_milvus: _FakeMilvus) -> None:
    """A quote-bearing payload that _quote alone would only escape is rejected."""
    r = client.get(
        "/api/decisions/sample",
        params={"hero_action_type": "fold' or active == true or 'x' == 'x"},
    )
    assert r.status_code == 422
    assert fake_milvus.last_filter is None


@pytest.mark.parametrize("value", ["bet_50", "raise", "bet", "call", "allin"])
def test_hero_action_type_accepts_stored_vocab(
    value: str, client: TestClient, fake_milvus: _FakeMilvus
) -> None:
    """hero_action_type is a mixed field: canonical buckets (solver/patch rows) AND
    raw verbs like 'raise'/'bet' (HH-ingest rows). Both must be accepted."""
    r = client.get("/api/decisions/sample", params={"hero_action_type": value})
    assert r.status_code == 200
    assert f"hero_action_type == '{value}'" in fake_milvus.last_filter
