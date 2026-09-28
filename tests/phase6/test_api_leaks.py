"""Contract tests for the read routers under src/api/ — Task 2.

Each router wraps exactly one backend function from src/study/*. Tests
monkeypatch the backend function and assert the router:
1. Forwards parameters with Pydantic validation (Query / Path / BaseModel).
2. Returns the backend's return value verbatim (D-09 single-source-of-truth).
3. Maps query-string validation errors to 422 (FastAPI default).

Backends are patched at the router-module level (the router does
``from src.study.leaks import rank_leaks``), so the monkeypatch target is
``src.api.leaks.rank_leaks``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient with DB deps stubbed.

    The router unit tests mock the backend function (rank_leaks etc.) so the
    actual conn/milvus values are never used — but FastAPI still resolves the
    Depends. Override get_tsdb / get_milvus to return harmless sentinels so
    the test environment doesn't need TSDB_PASSWORD set.
    """
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


# --- /api/leaks --------------------------------------------------------------


def test_get_leaks_returns_200_with_expected_shape(client: TestClient) -> None:
    canned = {"coverage": [], "strategy": [], "bucket_stats": {}}
    with patch("src.api.leaks.rank_leaks", return_value=canned):
        r = client.get("/api/leaks?type=both&min_n=20&limit=10")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == canned


def test_get_leaks_forwards_query_params(client: TestClient) -> None:
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"coverage": [], "strategy": [], "bucket_stats": {}}

    with patch("src.api.leaks.rank_leaks", side_effect=fake):
        r = client.get("/api/leaks?type=strategy&min_n=50&limit=5")
    assert r.status_code == 200, r.text
    assert captured["leak_type"] == "strategy"
    assert captured["min_n"] == 50
    assert captured["limit"] == 5


def test_get_leaks_validates_min_n_lower_bound(client: TestClient) -> None:
    r = client.get("/api/leaks?min_n=0")
    assert r.status_code == 422


def test_get_leaks_validates_type_pattern(client: TestClient) -> None:
    r = client.get("/api/leaks?type=bogus")
    assert r.status_code == 422


def test_get_leaks_default_params(client: TestClient) -> None:
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"coverage": [], "strategy": [], "bucket_stats": {}}

    with patch("src.api.leaks.rank_leaks", side_effect=fake):
        r = client.get("/api/leaks")
    assert r.status_code == 200, r.text
    assert captured["leak_type"] == "both"
    assert captured["min_n"] == 20
    assert captured["limit"] == 20


# --- /api/similar/{cluster_key} ---------------------------------------------


def test_get_similar_returns_200_with_list(client: TestClient) -> None:
    canned = [{"rank": 1, "cluster_key": "ck2", "distance": 0.1, "source": "kNN", "n_obs": 50}]
    with patch("src.api.similar.find_similar", return_value=canned):
        r = client.get("/api/similar/cluster_a?k=5")
    assert r.status_code == 200, r.text
    assert r.json() == canned


def test_get_similar_forwards_cluster_key_and_k(client: TestClient) -> None:
    captured: dict = {}

    def fake(cluster_key, **kwargs):
        captured["cluster_key"] = cluster_key
        captured.update(kwargs)
        return []

    with patch("src.api.similar.find_similar", side_effect=fake):
        r = client.get("/api/similar/cluster_a?k=3")
    assert r.status_code == 200, r.text
    assert captured["cluster_key"] == "cluster_a"
    assert captured["k"] == 3
    # API contract: always include action_dist in JSON output (D-13).
    assert captured["include_action_dist"] is True


# --- /api/probe ---------------------------------------------------------------


def test_get_probe_by_cluster_key(client: TestClient) -> None:
    canned = {
        "engine_response": {"source": "kNN", "action_dist": {"fold": 0.3, "call": 0.7}},
        "knn_neighbors": [],
        "solver_truth": None,
    }
    with patch("src.api.probe.probe_by_cluster_key", return_value=canned):
        r = client.get("/api/probe?cluster_key=ck1&k=3")
    assert r.status_code == 200, r.text
    assert r.json()["engine_response"]["source"] == "kNN"


def test_post_probe_by_spot(client: TestClient) -> None:
    canned = {"engine_response": {}, "knn_neighbors": [], "solver_truth": None}
    with patch("src.api.probe.probe_by_spot", return_value=canned):
        r = client.post(
            "/api/probe/spot",
            json={"spot": {"street": "flop", "pot": 100}, "k": 5},
        )
    assert r.status_code == 200, r.text
    assert "engine_response" in r.json()


# --- /api/patches -------------------------------------------------------------


def test_get_patches_returns_list(client: TestClient) -> None:
    canned = [{"patch_id": "abc", "cluster_key": "ck1", "source": "manual"}]
    with patch("src.api.patches.list_patches", return_value=canned):
        r = client.get("/api/patches?limit=10")
    assert r.status_code == 200, r.text
    assert r.json() == canned


def test_get_patch_detail_returns_404_when_missing(client: TestClient) -> None:
    with patch("src.api.patches.patch_detail", return_value=None):
        r = client.get("/api/patches/does-not-exist")
    assert r.status_code == 404


def test_get_patch_detail_returns_200(client: TestClient) -> None:
    canned = {"patch_id": "abc", "cluster_key": "ck1", "source": "manual", "chain": []}
    with patch("src.api.patches.patch_detail", return_value=canned):
        r = client.get("/api/patches/abc")
    assert r.status_code == 200, r.text
    assert r.json()["patch_id"] == "abc"


# --- /api/dashboard -----------------------------------------------------------


def test_get_dashboard(client: TestClient) -> None:
    canned = {
        "loop_health": {"verdict": "improving"},
        "ev_loss_trend": [],
        "recent_activity": [],
        "health": {"db": "ok"},
        "session": None,
        "kb_growth": {"total_nodes": 0},
    }
    with patch("src.api.dashboard.dashboard_snapshot", return_value=canned):
        r = client.get("/api/dashboard")
    assert r.status_code == 200, r.text
    assert r.json()["loop_health"]["verdict"] == "improving"


# --- /api/hands ---------------------------------------------------------------


def test_get_hands_list(client: TestClient) -> None:
    canned = [{"obs_id": "o1", "session_id": "s1", "cluster_key": "ck1"}]
    with patch("src.api.hands.list_hands", return_value=canned):
        r = client.get("/api/hands?session_id=s1&limit=20")
    assert r.status_code == 200, r.text
    assert r.json() == canned


def test_get_hands_replay(client: TestClient) -> None:
    canned = [{"obs_id": "o1", "ts": "2026-05-19T00:00:00Z"}]
    with patch("src.api.hands.get_replay", return_value=canned):
        r = client.get("/api/hands/o1/replay")
    assert r.status_code == 200, r.text
    assert r.json() == canned


def test_get_hands_validates_limit(client: TestClient) -> None:
    r = client.get("/api/hands?limit=0")
    assert r.status_code == 422


# --- /api/ingest --------------------------------------------------------------


def test_post_ingest_incremental(client: TestClient) -> None:
    canned = {
        "hands_processed": 42,
        "hands_skipped": {},
        "elapsed_s": 1.2,
        "watermark_before": 100,
        "watermark_after": 142,
    }
    with patch("src.api.ingest.ingest_incremental", return_value=canned):
        r = client.post("/api/ingest", json={"rebuild": False})
    assert r.status_code == 200, r.text
    assert r.json()["hands_processed"] == 42


def test_post_ingest_rebuild_calls_rebuild_function(client: TestClient) -> None:
    canned = {"hands_processed": 999, "rebuild": True}
    with patch("src.api.ingest.ingest_rebuild", return_value=canned) as fn:
        r = client.post("/api/ingest", json={"rebuild": True})
    assert r.status_code == 200, r.text
    fn.assert_called_once()


# --- /api/suppressions --------------------------------------------------------


def test_post_suppression(client: TestClient) -> None:
    with patch("src.api.suppressions.suppress", return_value="sid-42"):
        r = client.post(
            "/api/suppressions",
            json={"cluster_key": "ck1", "reason": "intentional"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["suppression_id"] == "sid-42"
    assert body["cluster_key"] == "ck1"
    assert body["active"] is True


def test_delete_suppression(client: TestClient) -> None:
    with patch("src.api.suppressions.unsuppress", return_value=2):
        r = client.delete("/api/suppressions/ck1")
    assert r.status_code == 200, r.text
    assert r.json() == {"cluster_key": "ck1", "n_deactivated": 2}


def test_get_suppression_status(client: TestClient) -> None:
    with patch("src.api.suppressions.is_suppressed", return_value=True):
        r = client.get("/api/suppressions/ck1")
    assert r.status_code == 200, r.text
    assert r.json() == {"cluster_key": "ck1", "active": True}
