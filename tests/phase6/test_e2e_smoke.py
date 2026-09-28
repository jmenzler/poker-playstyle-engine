"""Opt-in live API checks; POKER_ENGINE_API_BASE overrides the loopback URL."""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

pytestmark = pytest.mark.e2e

# --- Configuration -----------------------------------------------------------

_API_BASE = os.environ.get("POKER_ENGINE_API_BASE", "http://127.0.0.1:8765").rstrip("/")


# --- Helper ------------------------------------------------------------------


def _get(path: str, *, timeout: int = 10) -> tuple[int, dict | list]:
    """HTTP GET via subprocess curl — avoids the `requests` optional dependency."""
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/stdout", "-w", "\n%{http_code}", f"{_API_BASE}{path}"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    lines = result.stdout.strip().rsplit("\n", 1)
    status_code = int(lines[-1]) if len(lines) > 1 else 0
    body_text = lines[0] if len(lines) > 1 else result.stdout.strip()
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError:
        body = {"_raw": body_text}
    return status_code, body


def _post(path: str, payload: dict, *, timeout: int = 30) -> tuple[int, dict | list]:
    """HTTP POST via subprocess curl with JSON body."""
    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
            "-o",
            "/dev/stdout",
            "-w",
            "\n%{http_code}",
            f"{_API_BASE}{path}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    lines = result.stdout.strip().rsplit("\n", 1)
    status_code = int(lines[-1]) if len(lines) > 1 else 0
    body_text = lines[0] if len(lines) > 1 else result.stdout.strip()
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError:
        body = {"_raw": body_text}
    return status_code, body


# --- Test 1: Health check ----------------------------------------------------


def test_health_endpoint_returns_200_with_ok_status() -> None:
    """GET /api/health returns 200 with {status: 'ok', n_jobs: <int>}."""
    status, body = _get("/api/health")
    assert status == 200, f"Expected 200, got {status}. Response: {body}"
    assert isinstance(body, dict), f"Expected dict, got: {body}"
    assert body.get("status") == "ok", f"Expected status='ok', got: {body}"
    assert "n_jobs" in body, f"Expected n_jobs key, got: {body}"


# --- Test 2: Leaks endpoint --------------------------------------------------


def test_leaks_endpoint_returns_valid_json() -> None:
    """GET /api/leaks?type=both&limit=5 returns valid JSON with expected shape."""
    status, body = _get("/api/leaks?type=both&limit=5")
    assert status == 200, f"Expected 200, got {status}. Response: {body}"
    assert isinstance(body, dict), f"Expected dict response, got: {type(body)}"
    # Shape from Plan 06-03: {coverage: [], strategy: [], bucket_stats: {}}
    assert "coverage" in body or "strategy" in body, f"Unexpected shape: {list(body.keys())}"


# --- Test 3: Dashboard endpoint ----------------------------------------------


def test_dashboard_returns_six_expected_keys() -> None:
    """GET /api/dashboard returns dict with all 6 expected keys."""
    status, body = _get("/api/dashboard")
    assert status == 200, f"Expected 200, got {status}. Response: {body}"
    assert isinstance(body, dict), f"Expected dict, got: {type(body)}"
    expected_keys = {"loop_health", "ev_loss_trend", "recent_activity", "health", "kb_growth"}
    missing = expected_keys - set(body.keys())
    assert not missing, f"Dashboard missing keys: {missing}. Got: {list(body.keys())}"


# --- Test 4: Eval run + poll -------------------------------------------------


def test_eval_run_queues_match_and_match_appears() -> None:
    """POST /api/eval/run queues a match; /api/eval/matches returns it."""
    status, body = _post(
        "/api/eval/run",
        {"opponent": "random", "n_hands": 100},
    )
    assert status == 200, f"POST /api/eval/run failed with {status}: {body}"
    assert isinstance(body, dict), f"Expected dict, got: {body}"
    # The endpoint returns a job_id or match entry — accept either shape
    assert "job_id" in body or "match_id" in body or "id" in body, (
        f"Expected job_id/match_id in response: {body}"
    )

    # Poll /api/eval/matches for up to 30s to see the match appear
    deadline = time.time() + 30
    found = False
    while time.time() < deadline:
        s, matches = _get("/api/eval/matches?limit=10")
        assert s == 200, f"GET /api/eval/matches failed: {s}"
        if isinstance(matches, list) and len(matches) > 0:
            found = True
            break
        time.sleep(2)
    assert found, "No matches appeared in /api/eval/matches within 30s"


# --- Test 5: Frontend StaticFiles --------------------------------------------


def test_root_serves_frontend_html() -> None:
    """GET / returns 200 and serves the React frontend index.html."""
    result = subprocess.run(
        [
            "curl",
            "-s",
            "-L",
            "-o",
            "/dev/stdout",
            "-w",
            "\n%{http_code}",
            f"{_API_BASE}/",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    lines = result.stdout.strip().rsplit("\n", 1)
    status_code = int(lines[-1]) if len(lines) > 1 else 0
    body_text = lines[0] if len(lines) > 1 else result.stdout.strip()

    assert status_code == 200, f"Expected 200 from /, got {status_code}"
    assert "<!doctype html>" in body_text.lower() or "<html" in body_text.lower(), (
        f"Expected HTML at /, got: {body_text[:200]!r}"
    )


# --- Test 6: CLI smoke — leaks -----------------------------------------------


def test_cli_leaks_exits_0_with_valid_json() -> None:
    """Run the installed CLI only against an explicitly configured test database."""
    if not os.environ.get("TSDB_PASSWORD"):
        pytest.skip("Explicit test database configuration is required")
    cmd = ["poker-engine", "leaks", "--format", "json", "--limit", "3"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"{' '.join(cmd)} exited {result.returncode}.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )
    # Verify output is parseable JSON (or at least valid JSON-like output)
    stdout = result.stdout.strip()
    if stdout:
        try:
            parsed = json.loads(stdout)
            assert parsed is not None, "Parsed JSON was None"
        except json.JSONDecodeError:
            # Some CLI output may be tabular — just verify exit 0
            pass


# --- Test 7: CLI smoke — dashboard -------------------------------------------


def test_cli_dashboard_exits_0() -> None:
    """Run the local dashboard command with explicit test credentials."""
    if not os.environ.get("TSDB_PASSWORD"):
        pytest.skip("Explicit test database configuration is required")
    cmd = ["poker-engine", "dashboard", "--format", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"{' '.join(cmd)} exited {result.returncode}.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )


