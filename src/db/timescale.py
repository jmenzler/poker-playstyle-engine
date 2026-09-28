"""ERR-01: fail-loud TimescaleDB connect wrapper.

All TimescaleDB-touching code in poker-engine MUST go through this function.
Direct ``psycopg.connect()`` outside ``src/db/timescale.connect`` is an anti-pattern
(silent fallback risk — see PROJECT.md ERR-01).

SPIKE A1 (.planning/phases/01-foundation/01-01-SPIKE.md) confirmed that
``psycopg.connect()`` against an unreachable host raises
``psycopg.errors.ConnectionTimeout``, a subclass of ``psycopg.OperationalError``.
Catching ``OperationalError`` therefore covers all connect-failure modes.
"""

import psycopg

from src._errors import DBConnectError
from src._log import get_logger
from src._redact import redact_dsn

log = get_logger("db.timescale")


def connect(dsn: str, *, autocommit: bool = False) -> psycopg.Connection:
    """Open a psycopg connection. Raise ``DBConnectError`` on failure with redacted DSN.

    Args:
        dsn: psycopg DSN (URL or KV form). May contain a password — the caller is
            responsible for passing a real DSN; this function redacts before any
            log or exception use.
        autocommit: passed to ``psycopg.connect``.

    Returns:
        ``psycopg.Connection`` on success.

    Raises:
        DBConnectError: on any psycopg connect failure. Message contains the redacted
            DSN (NOT the raw password). Original exception chained via ``from e``.
    """
    try:
        return psycopg.connect(dsn, autocommit=autocommit)
    # SPIKE A1 verdict: psycopg.OperationalError covers ConnectionTimeout subclass.
    except psycopg.OperationalError as e:
        redacted = redact_dsn(dsn)
        log.error("db.connect.failed", dsn=redacted, error=str(e))
        raise DBConnectError(f"TimescaleDB connect failed: {redacted}") from e
