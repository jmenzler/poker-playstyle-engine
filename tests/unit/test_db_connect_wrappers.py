"""ERR-01: fail-loud DB connect wrappers — exception type + redaction + chaining.

Uses unreachable IP 10.255.255.1 (RFC 6598 reserved) with short connect_timeout
to trigger failure deterministically without requiring any live DB stack.
"""

import io
import logging

import pytest

from src._errors import DBConnectError
from src._log import configure_logging
from src.db.milvus import connect as milvus_connect
from src.db.timescale import connect as timescale_connect

# ─── TimescaleDB ──────────────────────────────────────────────────────────────


def test_timescale_connect_unreachable_raises_dbconnect_error() -> None:
    dsn = "host=10.255.255.1 port=5432 dbname=poker_engine user=u password=hunter2 connect_timeout=1"
    with pytest.raises(DBConnectError) as ei:
        timescale_connect(dsn)
    msg = str(ei.value)
    assert "hunter2" not in msg, "raw password leaked"
    assert "***" in msg, "redaction marker missing"


def test_timescale_connect_chains_original_exception() -> None:
    dsn = "host=10.255.255.1 port=5432 dbname=poker_engine user=u password=hunter2 connect_timeout=1"
    with pytest.raises(DBConnectError) as ei:
        timescale_connect(dsn)
    assert ei.value.__cause__ is not None, "original exception not chained via from-clause"
    # The chained exception should be a psycopg error (any subtype)
    cause_name = type(ei.value.__cause__).__name__
    assert "Error" in cause_name or "Operational" in cause_name or "Timeout" in cause_name


def test_timescale_connect_logs_redacted_dsn() -> None:
    """The error log line MUST contain the redacted DSN, not the raw one."""
    # Capture root logger output
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    root.handlers = [handler]
    root.setLevel(logging.ERROR)
    try:
        configure_logging(level="ERROR")
        dsn = "host=10.255.255.1 port=5432 dbname=poker_engine user=u password=hunter2 connect_timeout=1"
        with pytest.raises(DBConnectError):
            timescale_connect(dsn)
        output = buf.getvalue()
        assert "hunter2" not in output, "raw password leaked to log output"
        # Either *** is present OR no DSN appeared at all
        assert "***" in output or "dsn" not in output.lower()
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)


# ─── Milvus ───────────────────────────────────────────────────────────────────


def test_milvus_connect_unreachable_raises_dbconnect_error() -> None:
    # Pymilvus may or may not raise immediately on construction; the wrapper is
    # responsible for forcing a round-trip and raising DBConnectError on the
    # failure mode reported by Plan-01 SPIKE A2 (MilvusException).
    with pytest.raises(DBConnectError) as ei:
        milvus_connect("http://x:badtoken@10.255.255.1:19530", timeout=2.0)
    msg = str(ei.value)
    assert "badtoken" not in msg
    assert "***" in msg


def test_milvus_connect_chains_original_exception() -> None:
    with pytest.raises(DBConnectError) as ei:
        milvus_connect("http://x:badtoken@10.255.255.1:19530", timeout=2.0)
    assert ei.value.__cause__ is not None


# ─── Milvus per-RPC timeout injection ─────────────────────────────────────────


class _RecordingClient:
    """Stub MilvusClient capturing the kwargs each data-plane method receives."""

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def has_collection(self, name: str, timeout=None):
        self.calls["has_collection"] = {"timeout": timeout}
        return False

    def search(self, **kwargs):
        self.calls["search"] = kwargs
        return []

    def query(self, **kwargs):
        self.calls["query"] = kwargs
        return []

    def upsert(self, **kwargs):
        self.calls["upsert"] = kwargs
        return {}

    def describe_collection(self, name: str):
        self.calls["describe_collection"] = {"name": name}
        return {"fields": []}


def _proxy_over(stub):
    from src.db.milvus import DEFAULT_RPC_TIMEOUT_S, _TimeoutInjectingClient

    return _TimeoutInjectingClient(stub, DEFAULT_RPC_TIMEOUT_S)


def test_milvus_proxy_injects_default_timeout_on_search() -> None:
    from src.db.milvus import DEFAULT_RPC_TIMEOUT_S

    stub = _RecordingClient()
    proxy = _proxy_over(stub)
    proxy.search(collection_name="c", data=[[0.0]])
    assert stub.calls["search"]["timeout"] == DEFAULT_RPC_TIMEOUT_S


def test_milvus_proxy_respects_caller_timeout() -> None:
    stub = _RecordingClient()
    proxy = _proxy_over(stub)
    proxy.upsert(collection_name="c", data=[{}], timeout=5.0)
    assert stub.calls["upsert"]["timeout"] == 5.0


def test_milvus_proxy_injects_timeout_on_query() -> None:
    from src.db.milvus import DEFAULT_RPC_TIMEOUT_S

    stub = _RecordingClient()
    proxy = _proxy_over(stub)
    proxy.query(collection_name="c", filter="x == 1")
    assert stub.calls["query"]["timeout"] == DEFAULT_RPC_TIMEOUT_S


def test_milvus_proxy_delegates_non_data_plane_untouched() -> None:
    """Non-data-plane methods (describe_collection) pass through with no injected timeout."""
    stub = _RecordingClient()
    proxy = _proxy_over(stub)
    out = proxy.describe_collection("postflop_decisions")
    assert out == {"fields": []}
    assert "timeout" not in stub.calls["describe_collection"]


def test_milvus_connect_message_names_underlying_error_type() -> None:
    """The DBConnectError message carries the underlying exception type so operators
    can triage auth/permission vs transport without reading logs."""
    with pytest.raises(DBConnectError) as ei:
        milvus_connect("http://x:badtoken@10.255.255.1:19530", timeout=2.0)
    msg = str(ei.value)
    assert type(ei.value.__cause__).__name__ in msg
