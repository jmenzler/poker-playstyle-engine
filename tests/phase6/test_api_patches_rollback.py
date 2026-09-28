"""Unit tests for POST /api/patches/{patch_id}/rollback endpoint.

Mocks rollback_patch so tests run without any DB connections.
Four cases covered: happy path (200), 404, 409, 422.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.cli.rollback import RollbackResult


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient with DB deps stubbed (same pattern as test_api_leaks.py)."""
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


VALID_PATCH_ID = "12345678-1234-5678-1234-567812345678"
ROLLBACK_PATCH_ID = "87654321-4321-8765-4321-876543218765"

CANNED_RESULT = RollbackResult(
    rollback_patch_id=ROLLBACK_PATCH_ID,
    restored_node_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    source="rolled_back",
    ts="2026-05-19T12:00:00+00:00",
)


def test_rollback_happy_path_200(client: TestClient) -> None:
    """POST /{id}/rollback returns 200 with rollback receipt on success."""
    with patch("src.api.patches.rollback_patch", return_value=CANNED_RESULT):
        r = client.post(f"/api/patches/{VALID_PATCH_ID}/rollback")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rollback_patch_id"] == ROLLBACK_PATCH_ID
    assert body["restored_node_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert body["source"] == "rolled_back"
    assert body["ts"] == "2026-05-19T12:00:00+00:00"


def test_rollback_404_when_not_found(client: TestClient) -> None:
    """POST /{id}/rollback returns 404 when patch_id is unknown."""
    with patch(
        "src.api.patches.rollback_patch",
        side_effect=LookupError(f"patch_id {VALID_PATCH_ID} not found in patches table"),
    ):
        r = client.post(f"/api/patches/{VALID_PATCH_ID}/rollback")
    assert r.status_code == 404, r.text
    assert "not found" in r.json()["detail"].lower()


def test_rollback_409_when_already_rolled_back(client: TestClient) -> None:
    """POST /{id}/rollback returns 409 when patch was already rolled back."""
    with patch(
        "src.api.patches.rollback_patch",
        side_effect=RuntimeError(f"patch {VALID_PATCH_ID} already rolled back"),
    ):
        r = client.post(f"/api/patches/{VALID_PATCH_ID}/rollback")
    assert r.status_code == 409, r.text
    assert "already rolled back" in r.json()["detail"].lower()


def test_rollback_422_on_invalid_patch_id_format(client: TestClient) -> None:
    """POST /not-a-uuid/rollback returns 422 for invalid UUID format."""
    r = client.post("/api/patches/not-a-valid-uuid/rollback")
    assert r.status_code == 422, r.text
    assert "invalid patch_id format" in r.json()["detail"].lower()
