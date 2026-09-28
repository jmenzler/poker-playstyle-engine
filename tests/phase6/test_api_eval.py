"""Contract tests for SSE-bearing routers — Task 3.

Covers src/api/verify.py + src/api/eval.py:
- POST /api/verify returns a job_id.
- POST /api/verify/cancel/{job_id} cancels a running task.
- POST /api/eval/run returns job_id; unknown opponent → 400.
- GET  /api/eval/leaderboard returns rows from matches table.
- GET  /api/eval/matches with filters returns rows.
- src/api/eval.py validates opponent against BASELINE_REGISTRY.

SSE streaming is exercised by a structural test (EventSourceResponse import)
since the streaming-loop behavior is hard to unit-test cleanly with TestClient
sync — Plan-07 verification on the PC will hit the live SSE streams.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient with DB deps stubbed (see test_api_leaks.py)."""
    from src.api.deps import get_milvus, get_tsdb
    from src.api.main import app

    def _stub_tsdb():
        # Provide a MagicMock so .cursor() etc. don't crash if the router
        # actually touches the conn before the patched backend can short-
        # circuit. Tests that need specific cursor returns will patch
        # individually.
        mock = MagicMock()
        yield mock

    def _stub_milvus():
        return MagicMock()

    app.dependency_overrides[get_tsdb] = _stub_tsdb
    app.dependency_overrides[get_milvus] = _stub_milvus
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


# --- /api/verify -------------------------------------------------------------


def test_post_verify_returns_job_id(client: TestClient) -> None:
    """POST /api/verify creates a Job and returns job_id + status."""
    with patch("src.api.verify._run_verify") as fake_run:
        # _run_verify is an async coroutine; let it return immediately.
        async def _noop(*args, **kwargs):
            return None

        fake_run.side_effect = _noop
        r = client.post("/api/verify", json={"cluster_key": "ck1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["status"] in ("queued", "running")
    assert body["cluster_key"] == "ck1"


def test_post_verify_cancel_returns_404_for_missing_job(client: TestClient) -> None:
    r = client.post("/api/verify/cancel/nope-not-real")
    assert r.status_code == 404


def test_post_verify_cancel_requests_best_effort(client: TestClient) -> None:
    """Cancel is best-effort: it flags intent and cancels the task, but does NOT
    report 'cancelled' synchronously — the worker flips status once the
    uninterruptible executor thread actually exits.
    """
    from src.api import jobs as api_jobs

    job = api_jobs.registry.create(kind="verify")

    # MagicMock mimics the asyncio.Task surface the cancel handler uses; the
    # TestClient sync test has no running loop to build a real Task.
    mock_task = MagicMock()
    mock_task.done.return_value = False
    job.task = mock_task

    r = client.post(f"/api/verify/cancel/{job.job_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job.job_id
    assert body["cancel_requested"] is True
    assert body["status"] != "cancelled"  # not a status lie while work may run
    assert job.metadata["cancel_requested"] is True
    mock_task.cancel.assert_called_once()


def test_verify_router_uses_event_source_response() -> None:
    """Structural guard: verify.py imports + uses EventSourceResponse."""
    from pathlib import Path

    src = Path("src/api/verify.py").read_text()
    assert "EventSourceResponse" in src
    assert "run_in_executor" in src


def test_verify_router_has_semaphore() -> None:
    """Structural guard: a Semaphore (or Lock) enforces sequential Stage B (OQ-2)."""
    from pathlib import Path

    src = Path("src/api/verify.py").read_text()
    # Strip comment lines to avoid matching prose.
    non_comment = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "Semaphore" in non_comment


def test_verify_sse_404_for_missing_job(client: TestClient) -> None:
    r = client.get("/api/verify/sse/does-not-exist")
    assert r.status_code == 404


# --- /api/eval/run -----------------------------------------------------------


def test_post_eval_run_unknown_opponent_returns_400(client: TestClient) -> None:
    r = client.post(
        "/api/eval/run",
        json={"opponent": "totally-fake", "hands": 100, "seed": 42},
    )
    assert r.status_code == 400, r.text
    assert "totally-fake" in r.json()["detail"]


def test_post_eval_run_known_opponent_returns_job_id(client: TestClient) -> None:
    with patch("src.api.eval._run_match") as fake_run:

        async def _noop(*args, **kwargs):
            return None

        fake_run.side_effect = _noop
        r = client.post(
            "/api/eval/run",
            json={"opponent": "random", "hands": 100, "seed": 42},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["opponent"] == "random"
    assert body["hands"] == 100
    assert body["status"] in ("queued", "running")


def test_post_eval_run_defaults_table_size_6(client: TestClient) -> None:
    """GUI/API default is 6-max (the corpus is 6-max; HU is out-of-distribution)."""
    with patch("src.api.eval._run_match") as fake_run:

        async def _noop(*args, **kwargs):
            return None

        fake_run.side_effect = _noop
        r = client.post("/api/eval/run", json={"opponent": "random", "hands": 100, "seed": 42})
    assert r.status_code == 200, r.text
    assert r.json()["table_size"] == 6


def test_post_eval_run_bad_table_size_400(client: TestClient) -> None:
    r = client.post("/api/eval/run", json={"opponent": "random", "table_size": 99})
    assert r.status_code == 400, r.text


def test_post_eval_run_opponents_wrong_length_400(client: TestClient) -> None:
    r = client.post("/api/eval/run", json={"opponents": ["random"], "table_size": 6})
    assert r.status_code == 400, r.text


def test_post_eval_run_opponents_unknown_400(client: TestClient) -> None:
    r = client.post("/api/eval/run", json={"opponents": ["random", "totally-fake"], "table_size": 3})
    assert r.status_code == 400, r.text
    assert "totally-fake" in r.json()["detail"]


def test_post_eval_run_per_seat_label(client: TestClient) -> None:
    with patch("src.api.eval._run_match") as fake_run:

        async def _noop(*args, **kwargs):
            return None

        fake_run.side_effect = _noop
        r = client.post(
            "/api/eval/run",
            json={"opponents": ["random", "TAG-profile"], "table_size": 3, "hands": 100},
        )
    assert r.status_code == 200, r.text
    assert r.json()["opponent"] == "random+TAG-profile"


def test_eval_run_validates_opponent_against_registry() -> None:
    """Source-level grep: eval.py references BASELINE_REGISTRY or REGISTRY."""
    from pathlib import Path

    src = Path("src/api/eval.py").read_text()
    assert "BASELINE_REGISTRY" in src or "REGISTRY" in src


def test_eval_router_uses_event_source_response() -> None:
    from pathlib import Path

    src = Path("src/api/eval.py").read_text()
    assert "EventSourceResponse" in src


def test_eval_router_has_semaphore() -> None:
    from pathlib import Path

    src = Path("src/api/eval.py").read_text()
    non_comment = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "Semaphore" in non_comment


def test_eval_sse_404_for_missing_job(client: TestClient) -> None:
    r = client.get("/api/eval/sse/does-not-exist")
    assert r.status_code == 404


# --- /api/eval/compare + /api/eval/versions ----------------------------------


def test_post_eval_compare_returns_job_id(client: TestClient) -> None:
    """POST /api/eval/compare spawns a head-to-head job and echoes the versions."""
    with patch("src.api.eval._run_compare") as fake_run:

        async def _noop(*args, **kwargs):
            return None

        fake_run.side_effect = _noop
        r = client.post("/api/eval/compare", json={"v_old": 1, "v_new": 2, "hands": 100, "seed": 42})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["v_old"] == 1 and body["v_new"] == 2
    assert body["hands"] == 100
    assert body["status"] in ("queued", "running")


def test_post_eval_compare_same_version_400(client: TestClient) -> None:
    r = client.post("/api/eval/compare", json={"v_old": 2, "v_new": 2})
    assert r.status_code == 400, r.text


def test_get_eval_versions_returns_list(client: TestClient) -> None:
    rows = [
        {
            "version": 2,
            "cutoff_ts": 100,
            "source": "manual",
            "label": "v1",
            "created_at": "2026-06-24T00:00:00Z",
        },
        {
            "version": 1,
            "cutoff_ts": 0,
            "source": "manual",
            "label": "genesis",
            "created_at": "2026-06-24T00:00:00Z",
        },
    ]
    with patch("src.study.corpus_versions.list_versions", return_value=rows):
        r = client.get("/api/eval/versions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 2
    assert body[0]["version"] == 2 and body[0]["label"] == "v1"


# --- /api/eval/leaderboard ---------------------------------------------------


def test_get_eval_leaderboard_returns_aggregated_rows() -> None:
    """The leaderboard runs a real SQL query, so we patch the cursor."""
    from src.api.deps import get_tsdb
    from src.api.main import app

    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [
        ("random", 5, 5000, 12.4, 8.0, 16.0, 5, 0, 0, 0),
    ]
    mock_cur.description = [
        ("opponent",),
        ("n_matches",),
        ("total_hands",),
        ("avg_bb_per_100",),
        ("avg_ci_low",),
        ("avg_ci_high",),
        ("wins",),
        ("losses",),
        ("ties",),
        ("regressions",),
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    def _stub():
        yield mock_conn

    app.dependency_overrides[get_tsdb] = _stub
    try:
        with TestClient(app) as c:
            r = c.get("/api/eval/leaderboard")
    finally:
        # don't clear other overrides
        app.dependency_overrides.pop(get_tsdb, None)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["opponent"] == "random"
    assert body[0]["n_matches"] == 5


# --- /api/eval/matches -------------------------------------------------------


def test_get_eval_matches_with_filters() -> None:
    """Filter forwarding via SQL parameter binding."""
    from src.api.deps import get_tsdb
    from src.api.main import app

    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [
        (
            "m1",
            "random",
            1000,
            42,
            12.4,
            8.0,
            16.0,
            "won",
            "0.1.0",
            "2026-05-19T00:00:00Z",
            "2026-05-19T01:00:00Z",
        ),
    ]
    mock_cur.description = [
        ("match_id",),
        ("opponent",),
        ("hands",),
        ("seed",),
        ("bb_per_100",),
        ("ci_low",),
        ("ci_high",),
        ("status",),
        ("engine_version",),
        ("started_at",),
        ("finished_at",),
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    def _stub():
        yield mock_conn

    app.dependency_overrides[get_tsdb] = _stub
    try:
        with TestClient(app) as c:
            r = c.get("/api/eval/matches?opponent=random&status=won&limit=10")
    finally:
        app.dependency_overrides.pop(get_tsdb, None)

    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["opponent"] == "random"
    assert body[0]["status"] == "won"


# --- DELETE /api/eval/matches/{match_id} -------------------------------------


def _delete_client(rowcount: int):
    """TestClient whose tsdb cursor reports `rowcount` deleted rows."""
    from src.api.deps import get_tsdb
    from src.api.main import app

    mock_cur = MagicMock()
    mock_cur.rowcount = rowcount
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    def _stub():
        yield mock_conn

    app.dependency_overrides[get_tsdb] = _stub
    return TestClient(app), mock_conn


def test_delete_match_ok() -> None:
    """Existing match deletes and commits, returning the row count."""
    from src.api.deps import get_tsdb
    from src.api.main import app

    client, conn = _delete_client(rowcount=1)
    mid = "11111111-1111-1111-1111-111111111111"
    try:
        with client as c:
            r = c.delete(f"/api/eval/matches/{mid}")
    finally:
        app.dependency_overrides.pop(get_tsdb, None)
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": 1}
    conn.commit.assert_called_once()


def test_delete_match_404_when_absent() -> None:
    """No matching row → 404."""
    from src.api.deps import get_tsdb
    from src.api.main import app

    client, _ = _delete_client(rowcount=0)
    mid = "22222222-2222-2222-2222-222222222222"
    try:
        with client as c:
            r = c.delete(f"/api/eval/matches/{mid}")
    finally:
        app.dependency_overrides.pop(get_tsdb, None)
    assert r.status_code == 404


def test_delete_match_400_on_bad_uuid() -> None:
    """Malformed match_id is rejected before touching the DB."""
    from src.api.deps import get_tsdb
    from src.api.main import app

    client, _ = _delete_client(rowcount=1)
    try:
        with client as c:
            r = c.delete("/api/eval/matches/not-a-uuid")
    finally:
        app.dependency_overrides.pop(get_tsdb, None)
    assert r.status_code == 400
