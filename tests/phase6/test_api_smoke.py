"""Smoke tests for the Phase 6 FastAPI app (src/api/main.py).

Covers:
- app construction + title/version
- /api/health endpoint
- CORS allowlist behavior for local browser and desktop origins
- StudyConfig defaults are loopback-only (never a wildcard bind)

All mock-free: TestClient hits the real app object.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from src.api.main import app

    return TestClient(app)


def test_app_titles_correctly() -> None:
    from src.api.main import app

    assert app.title == "poker-engine study"
    assert app.version == "0.2.0"


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # JobRegistry should report a count even when empty.
    assert "n_jobs" in body
    assert isinstance(body["n_jobs"], int)


def test_cors_preflight_for_tauri_dev_origin(client: TestClient) -> None:
    r = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:1420",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert r.status_code in (200, 204)
    headers_lower = {k.lower(): v for k, v in r.headers.items()}
    # Starlette CORS echoes the allowed origin back on a matched preflight.
    assert headers_lower.get("access-control-allow-origin") == "http://localhost:1420"


def test_cors_preflight_for_tauri_macos_prod_origin(client: TestClient) -> None:
    r = client.options(
        "/api/health",
        headers={
            "Origin": "tauri://localhost",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code in (200, 204)
    headers_lower = {k.lower(): v for k, v in r.headers.items()}
    assert headers_lower.get("access-control-allow-origin") == "tauri://localhost"


def test_cors_blocks_random_origin(client: TestClient) -> None:
    """Verify the allowlist actually filters: unlisted origin gets no ACAO header."""
    r = client.options(
        "/api/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    headers_lower = {k.lower(): v for k, v in r.headers.items()}
    # Either no ACAO header at all, or it's NOT echoing the evil origin.
    assert headers_lower.get("access-control-allow-origin") != "https://evil.example.com"


def test_fastapi_default_host_is_loopback() -> None:
    from src._config import StudyConfig

    cfg = StudyConfig()
    assert cfg.fastapi_host == "127.0.0.1"
    assert cfg.fastapi_host not in ("0.0.0.0", "::", "*")
    assert cfg.fastapi_port == 8765


def test_main_module_never_binds_to_zero_zero_zero_zero() -> None:
    """The main.py source MUST NOT contain a literal 0.0.0.0 outside comments.

    This guards against accidental config drift — local-only binding is a
    security invariant (D-02 + T-06-29 mitigation).
    """
    from pathlib import Path

    src = Path("src/api/main.py").read_text()
    # Strip comment lines, then assert no literal 0.0.0.0 appears.
    non_comment = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "0.0.0.0" not in non_comment
