from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_gap_ordering():
    """GET /api/gaps returns gaps in max_neighbor_distance DESC order."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.gaps import router

    from src.api.deps import get_milvus, get_tsdb

    rows = [
        {"decision_id": "h1_dp0", "max_neighbor_distance": 0.9, "hand_id": "h1"},
        {"decision_id": "h2_dp0", "max_neighbor_distance": 0.5, "hand_id": "h2"},
        {"decision_id": "h3_dp0", "max_neighbor_distance": 0.3, "hand_id": "h3"},
    ]

    app = FastAPI()
    app.include_router(router)
    mock_conn = MagicMock()
    app.dependency_overrides[get_tsdb] = lambda: mock_conn
    app.dependency_overrides[get_milvus] = lambda: MagicMock()

    with patch("src.api.gaps.list_gaps", return_value=rows):
        client = TestClient(app)
        resp = client.get("/api/gaps")

    assert resp.status_code == 200
    data = resp.json()
    distances = [r["max_neighbor_distance"] for r in data]
    assert distances == sorted(distances, reverse=True)


def test_multiway_send_to_solver_blocked():
    """POST /api/gaps/{decision_id}/send-to-solver returns 400 for multiway gaps."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.gaps import router

    from src._errors import ValidationError
    from src.api.deps import get_milvus, get_tsdb

    app = FastAPI()
    app.include_router(router)
    mock_conn = MagicMock()
    app.dependency_overrides[get_tsdb] = lambda: mock_conn
    app.dependency_overrides[get_milvus] = lambda: MagicMock()

    with patch(
        "src.api.gaps.send_to_solver",
        side_effect=ValidationError("solver is HU-only; multiway gaps cannot be sent to solver (D-17)"),
    ):
        client = TestClient(app)
        resp = client.post("/api/gaps/h1_dp0/send-to-solver")

    assert resp.status_code == 400
