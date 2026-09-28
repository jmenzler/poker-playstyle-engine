"""OBS-01: structured JSON logging required fields."""

import io
import json
import logging
import re


def _capture_log(logger_name: str, level: str = "INFO"):
    """Helper: configure structlog to write to an in-memory buffer; return (buffer, logger)."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level))

    from src._log import configure_logging, get_logger

    configure_logging(level=level)
    return buf, get_logger(logger_name)


def test_log_emits_json_with_required_fields():
    buf, log = _capture_log("test.component")
    log.info("hand.start", cluster_key="abc123", n_hands=5)
    line = buf.getvalue().strip().splitlines()[-1]
    record = json.loads(line)
    # OBS-01 required fields
    assert "level" in record
    assert record["level"].lower() == "info"
    assert "timestamp" in record
    assert "event" in record
    assert record["event"] == "hand.start"
    # component comes from logger_name via stdlib.add_logger_name
    # structlog may put logger name under "logger" or "component" depending on processors;
    # accept either spelling — but OBS-01 mandates the LOGGER NAME ("test.component") appears
    assert any(v == "test.component" for k, v in record.items() if isinstance(v, str))
    # Extra kwargs from call site
    assert record["cluster_key"] == "abc123"
    assert record["n_hands"] == 5


def test_timestamp_iso8601_utc():
    buf, log = _capture_log("test.component")
    log.info("evt")
    line = buf.getvalue().strip().splitlines()[-1]
    record = json.loads(line)
    ts = record["timestamp"]
    # ISO 8601 with UTC marker (Z or +00:00)
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts)
    assert ts.endswith("Z") or "+00:00" in ts


def test_configure_logging_is_idempotent():
    from src._log import configure_logging

    configure_logging(level="INFO")
    configure_logging(level="INFO")  # second call must not raise
