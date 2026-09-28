"""Contract tests for src/api/edit_node.py — Task 2 mutation router.

Covers:
- POST /api/edit-node valid body → backend invocation + 200 response.
- POST /api/edit-node ValidationError from backend → mapped to HTTP 400.
- POST /api/edit-node missing body fields → 422 (Pydantic validation).
- POST /api/edit-node/lock toggles locked_from_autoloop and returns count.
- POST /api/edit-node/lock 404 when set_lock_from_autoloop returns 0 rows.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src._errors import ValidationError


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient with DB deps stubbed (see test_api_leaks.py for rationale)."""
    from src.api.deps import get_milvus, get_tsdb
    from src.api.main import app

    def _stub_tsdb():
        yield "STUB_TSDB_CONN"

    def _stub_milvus():
        return "STUB_MILVUS_CLIENT"

    app.dependency_overrides[get_tsdb] = _stub_tsdb
    app.dependency_overrides[get_milvus] = _stub_milvus
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


# --- /api/edit-node -----------------------------------------------------------


def test_edit_node_happy_path(client: TestClient) -> None:
    canned = {
        "patch_id": "p1",
        "new_node_id": "n2",
        "source": "manual",
        "reason": "test",
    }
    with patch("src.api.edit_node.edit_node", return_value=canned):
        r = client.post(
            "/api/edit-node",
            json={
                "cluster_key": "ck1",
                "action_dist": {"fold": 0.3, "call": 0.7},
                "reason": "test",
            },
        )
    assert r.status_code == 200, r.text
    assert r.json() == canned


def test_edit_node_forwards_action_dist(client: TestClient) -> None:
    captured: dict = {}

    def fake(cluster_key, action_dist, **kwargs):
        captured["cluster_key"] = cluster_key
        captured["action_dist"] = action_dist
        captured["reason"] = kwargs.get("reason", "")
        return {"patch_id": "p1", "new_node_id": "n2", "source": "manual", "reason": ""}

    with patch("src.api.edit_node.edit_node", side_effect=fake):
        r = client.post(
            "/api/edit-node",
            json={"cluster_key": "ck1", "action_dist": {"fold": 1.0}, "reason": "r"},
        )
    assert r.status_code == 200, r.text
    assert captured["cluster_key"] == "ck1"
    assert captured["action_dist"] == {"fold": 1.0}
    assert captured["reason"] == "r"


def test_edit_node_validation_error_returns_400(client: TestClient) -> None:
    with patch("src.api.edit_node.edit_node", side_effect=ValidationError("bad sum")):
        r = client.post(
            "/api/edit-node",
            json={"cluster_key": "ck1", "action_dist": {"fold": 0.5}},
        )
    assert r.status_code == 400, r.text
    assert "bad sum" in r.json()["detail"]


def test_edit_node_missing_cluster_key_returns_422(client: TestClient) -> None:
    r = client.post("/api/edit-node", json={"action_dist": {"fold": 1.0}})
    assert r.status_code == 422


def test_edit_node_missing_action_dist_returns_422(client: TestClient) -> None:
    r = client.post("/api/edit-node", json={"cluster_key": "ck1"})
    assert r.status_code == 422


# --- /api/edit-node/lock ------------------------------------------------------


def test_lock_returns_200_with_updated_count(client: TestClient) -> None:
    with patch("src.api.edit_node.set_lock_from_autoloop", return_value=1):
        r = client.post("/api/edit-node/lock", json={"cluster_key": "ck1", "locked": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"cluster_key": "ck1", "locked": True, "n_updated": 1}


def test_lock_returns_404_when_no_rows_updated(client: TestClient) -> None:
    with patch("src.api.edit_node.set_lock_from_autoloop", return_value=0):
        r = client.post("/api/edit-node/lock", json={"cluster_key": "unknown", "locked": True})
    assert r.status_code == 404, r.text


def test_lock_unlock(client: TestClient) -> None:
    """Verify locked=False flows through (unlock path)."""
    captured: dict = {}

    def fake(cluster_key, locked, **kwargs):
        captured["cluster_key"] = cluster_key
        captured["locked"] = locked
        return 1

    with patch("src.api.edit_node.set_lock_from_autoloop", side_effect=fake):
        r = client.post("/api/edit-node/lock", json={"cluster_key": "ck1", "locked": False})
    assert r.status_code == 200, r.text
    assert captured["locked"] is False
