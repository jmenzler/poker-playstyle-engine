"""Root test configuration.

Imports shared fixtures from tests/property/conftest.py so they are available
to all test suites (unit, integration, property) without double-registration.
"""

import os
import socket
from pathlib import Path

import pytest

pytest_plugins = ["tests.export_fixtures"]

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def migrations_dir(project_root: Path) -> Path:
    return project_root / "migrations"


@pytest.fixture(scope="session")
def tsdb_dsn() -> str:
    """DSN for integration tests. Points at a test DB; TimescaleDB container must be reachable.

    Uses explicit environment variables to connect to caller-supplied test services.
    Falls back to local defaults to make CI explicit about what is being targeted.
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_TEST_DB", "poker_engine_test")
    user = os.environ.get("TSDB_USER", "poker")
    pwd = os.environ.get("TSDB_PASSWORD", "CHANGE_ME_LOCAL_ONLY")
    return f"host={host} port={port} dbname={db} user={user} password={pwd}"


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block all socket I/O to enforce CANON-01 invariant #9.

    Monkeypatches socket.socket and socket.create_connection so that any
    attempt to open a network connection during encode() raises immediately.
    Tests that use this fixture assert that canonicalization is pure and
    performs zero network I/O.
    """
    _msg = "encode() attempted network I/O — violates CANON-01"

    def _raise(*args: object, **kwargs: object) -> None:
        raise AssertionError(_msg)

    monkeypatch.setattr(socket, "socket", _raise)
    monkeypatch.setattr(socket, "create_connection", _raise, raising=False)
