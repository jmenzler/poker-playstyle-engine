"""OBS-01 / OBS-02: structlog JSON logging configuration.

Call configure_logging() once at process entry; thereafter use get_logger(name).
Bind per-request context via structlog.contextvars.bound_contextvars(...).
"""

import logging
from typing import cast

import structlog

_CONFIGURED: bool = False


def configure_logging(level: str = "INFO") -> None:
    """Idempotent. Configure structlog to emit one JSON object per log line.

    Required output fields (OBS-01):
    - level: log level (e.g., "info")
    - timestamp: ISO 8601 UTC (e.g., "2026-05-14T12:34:56.789Z" or "...+00:00")
    - component: logger name (added via stdlib.add_logger_name -> "logger" key by default)
    - event: the message string
    - plus any call-site keyword arguments and bound contextvars
    """
    global _CONFIGURED
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    # Configure root logger; tests may replace handlers before calling us.
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(numeric_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.add_logger_name,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # tests may reconfigure between calls
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structlog bound logger. Safe to call before configure_logging."""
    return cast(structlog.BoundLogger, structlog.get_logger(name))
