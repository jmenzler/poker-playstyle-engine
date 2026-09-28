"""OBS-02: bound contextvars propagate to log records within the bound block."""

import io
import json
import logging


def _setup():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers = [h]
    root.setLevel(logging.INFO)
    from src._log import configure_logging, get_logger

    configure_logging(level="INFO")
    return buf, get_logger("test.binding")


def test_bound_contextvars_propagate():
    buf, log = _setup()
    from structlog.contextvars import bound_contextvars

    with bound_contextvars(session_id="s-001", cluster_key="c-xyz"):
        log.info("inside")
    line = buf.getvalue().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["session_id"] == "s-001"
    assert record["cluster_key"] == "c-xyz"


def test_contextvars_do_not_leak():
    buf, log = _setup()
    from structlog.contextvars import bound_contextvars

    with bound_contextvars(session_id="s-leak-test"):
        log.info("inside")
    log.info("outside")
    lines = [json.loads(ln) for ln in buf.getvalue().strip().splitlines() if ln]
    # Get the LAST "outside" line
    outside = [r for r in lines if r.get("event") == "outside"][-1]
    assert "session_id" not in outside
