"""Idempotent SQL migration runner (Decision 4).

Apply all migrations/*.sql files (excluding the schema_migrations.sql bootstrap) in
lexical order. Track applied versions + SHA-256 checksums in the schema_migrations table.
Refuse to apply to any DB whose name does not contain 'poker' (T-1-02 safeguard).

Usage (CLI):
    TSDB_HOST=... TSDB_PORT=... TSDB_DB=... TSDB_USER=... TSDB_PASSWORD=... \\
        uv run python -m runner.migrate
"""

import hashlib
import os
from pathlib import Path

import psycopg

from src._errors import DBConnectError  # noqa: F401  — re-exported: connect() raises this (ERR-01)
from src._log import get_logger
from src.db.timescale import connect

log = get_logger("runner.migrate")

MIGRATIONS_DIR: Path = Path(__file__).resolve().parent.parent / "migrations"
TRACKING_FILE = "schema_migrations.sql"


def _iter_migrations(directory: Path) -> list[Path]:
    """Yield migration files in lexical order, excluding the tracking-table bootstrap."""
    return sorted(p for p in directory.glob("*.sql") if p.name != TRACKING_FILE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_tracking(conn: psycopg.Connection, directory: Path) -> None:
    """Apply schema_migrations.sql (itself idempotent — CREATE TABLE IF NOT EXISTS)."""
    boot = directory / TRACKING_FILE
    with conn.cursor() as cur:
        cur.execute(boot.read_text())
    conn.commit()


def _refuse_non_poker_db(conn: psycopg.Connection) -> None:
    """T-1-02 mitigation: refuse if DB name does not contain 'poker'."""
    with conn.cursor() as cur:
        cur.execute("SELECT current_database();")
        row = cur.fetchone()
    db_name: str = row[0] if row else ""
    if "poker" not in db_name.lower():
        raise RuntimeError(
            f"Refusing to run migrations: database name '{db_name}' does not contain 'poker'. "
            f"Decision 4 + T-1-02 safeguard."
        )


def run(dsn: str) -> int:
    """Apply pending migrations. Returns count newly applied. Idempotent.

    Args:
        dsn: psycopg DSN (URL or KV form). Password is redacted before any log or
            exception — see ERR-01 in src/db/timescale.py.

    Returns:
        Number of migration files newly applied (0 if already up-to-date).

    Raises:
        DBConnectError: if connection to the database fails.
        RuntimeError: if the target DB name does not contain 'poker' (T-1-02).
        RuntimeError: if a migration file's checksum differs from the stored record
            (Pitfall 5 — mid-flight edit detected).
    """
    conn: psycopg.Connection = connect(dsn)  # raises DBConnectError on failure (ERR-01)
    try:
        _refuse_non_poker_db(conn)
        _bootstrap_tracking(conn, MIGRATIONS_DIR)
        applied = 0
        for mpath in _iter_migrations(MIGRATIONS_DIR):
            checksum = _sha256(mpath)
            version = mpath.stem  # e.g., "001_observations"
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT checksum FROM schema_migrations WHERE version = %s;",
                    (version,),
                )
                row = cur.fetchone()
            if row is None:
                # New migration — apply in a transaction.
                with conn.cursor() as cur:
                    cur.execute(mpath.read_text())
                    cur.execute(
                        "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s);",
                        (version, checksum),
                    )
                conn.commit()
                log.info("migration.applied", version=version, checksum=checksum)
                applied += 1
            elif row[0] == checksum:
                log.info("migration.skip", version=version, reason="already_applied")
            else:
                # Pitfall 5: mid-flight edit detected — stored checksum differs.
                raise RuntimeError(
                    f"Checksum mismatch for {version}: stored={row[0]} "
                    f"current={checksum}. Refusing to re-apply edited migration. "
                    f"Manual remediation required (Pitfall 5)."
                )
        return applied
    finally:
        conn.close()


def _build_dsn_from_env() -> str:
    keys = ("TSDB_HOST", "TSDB_PORT", "TSDB_DB", "TSDB_USER", "TSDB_PASSWORD")
    env = {k: os.environ.get(k) for k in keys}
    missing = [k for k, v in env.items() if v is None]
    if missing:
        raise RuntimeError(f"missing env vars: {missing}")
    return (
        f"host={env['TSDB_HOST']} port={env['TSDB_PORT']} "
        f"dbname={env['TSDB_DB']} user={env['TSDB_USER']} "
        f"password={env['TSDB_PASSWORD']}"
    )


if __name__ == "__main__":
    count = run(_build_dsn_from_env())
    print(f"Applied {count} migration(s).")
