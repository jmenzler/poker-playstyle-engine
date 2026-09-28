"""STOR-04 acceptance: migration runner idempotency + checksum + non-poker-db refusal.

These tests REQUIRE a running TimescaleDB stack. They are marked ``integration``
so unit-only sweeps skip them automatically.

To run against a disposable test database:
    TSDB_HOST=127.0.0.1 TSDB_PORT=55432 TSDB_USER=poker TSDB_PASSWORD=<pwd> \\
    TSDB_TEST_DB=poker_engine_test \\
    uv run pytest tests/integration/test_migrations_idempotent.py -m integration -x -v
"""

from pathlib import Path
from unittest.mock import patch

import psycopg
import pytest

from src._errors import DBConnectError

pytestmark = pytest.mark.integration

# Tables created/owned by the 14 migration files. Used to clean up between
# tests; CASCADE handles any FK chains.
_TRACKED_TABLES = (
    "schema_migrations",
    "observations",
    "strategy_nodes",
    "weight_overlay",
    "patches",
    "metrics",
    "villain_logs",
    "leak_suppressions",
    "matches",
)


@pytest.fixture
def clean_test_db(tsdb_dsn: str) -> str:  # type: ignore[return]
    """Drop all known tables in the test DB before each test so the runner sees a blank slate."""
    with psycopg.connect(tsdb_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for tbl in _TRACKED_TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE;")
    yield tsdb_dsn  # type: ignore[misc]
    # No teardown — leave tables in place; next call will drop them.


def test_apply_against_clean_db_applies_all(clean_test_db: str) -> None:
    """run() against an empty DB applies all 14 migration files and returns 14.

    Count = `ls migrations/*.sql | grep -v schema_migrations | wc -l` = 14.
    """
    from runner.migrate import run

    applied = run(clean_test_db)
    assert applied == 14, f"expected 14 migrations applied, got {applied}"

    # Verify schema_migrations has exactly 14 rows.
    with psycopg.connect(clean_test_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM schema_migrations;")
            row = cur.fetchone()
    assert row is not None
    assert row[0] == 14, f"expected 14 rows in schema_migrations, got {row[0]}"


def test_re_apply_is_noop(clean_test_db: str) -> None:
    """run() on an already-migrated DB returns 0; no new rows in schema_migrations."""
    from runner.migrate import run

    first = run(clean_test_db)
    second = run(clean_test_db)

    assert first == 14, f"first run should apply 14, got {first}"
    assert second == 0, f"second run should apply 0 (noop), got {second}"

    # Row count must still be 14 (no duplicates).
    with psycopg.connect(clean_test_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM schema_migrations;")
            row = cur.fetchone()
    assert row is not None
    assert row[0] == 14, f"expected 14 rows after two runs, got {row[0]}"


def test_checksum_mismatch_raises(clean_test_db: str, tmp_path: Path) -> None:
    """After applying migrations, a tampered file causes run() to raise RuntimeError."""
    from runner import migrate

    # First apply the real migrations.
    migrate.run(clean_test_db)

    # Build a tampered migrations directory.
    tampered_dir = tmp_path / "migrations"
    tampered_dir.mkdir()

    real_migrations_dir = Path(migrate.MIGRATIONS_DIR)

    # Copy schema_migrations.sql verbatim (bootstrap file).
    (tampered_dir / "schema_migrations.sql").write_bytes(
        (real_migrations_dir / "schema_migrations.sql").read_bytes()
    )

    # Tamper 003_weight_overlay.sql.
    real_003 = real_migrations_dir / "003_weight_overlay.sql"
    (tampered_dir / "003_weight_overlay.sql").write_text(
        real_003.read_text() + "\n-- TAMPERED COMMENT THAT CHANGES THE CHECKSUM\n"
    )

    # Copy all other migration files verbatim.
    for name in (
        "001_observations.sql",
        "002_strategy_nodes.sql",
        "004_patches.sql",
        "005_metrics.sql",
        "006_villain_logs.sql",
        "007_indexes.sql",
        "008_flagged_sparse.sql",
        "009_metrics_unique.sql",
        "010_autoloop_source.sql",
        "011_locked_from_autoloop.sql",
        "012_leak_suppressions.sql",
        "013_solver_source.sql",
        "014_matches.sql",
    ):
        (tampered_dir / name).write_bytes((real_migrations_dir / name).read_bytes())

    # Point runner at tampered dir and expect RuntimeError.
    with patch.object(migrate, "MIGRATIONS_DIR", tampered_dir):
        with pytest.raises(RuntimeError, match=r"[Cc]hecksum mismatch"):
            migrate.run(clean_test_db)


def test_refuses_non_poker_db(tsdb_dsn: str) -> None:
    """run() raises RuntimeError before touching any DDL if the DB name lacks 'poker'."""
    from runner.migrate import run

    # Replace dbname with 'postgres' (always exists; never contains 'poker').
    non_poker_dsn = tsdb_dsn.replace(
        f"dbname={tsdb_dsn.split('dbname=')[1].split()[0]}",
        "dbname=postgres",
    )

    with pytest.raises(RuntimeError, match="does not contain 'poker'"):
        run(non_poker_dsn)


def test_dsn_redacted_on_failure() -> None:
    """ERR-01: a bad DSN must raise DBConnectError with *** redaction, not the raw password."""
    from runner.migrate import run

    bad_dsn = "host=10.255.255.1 port=5432 dbname=poker_engine_test user=x password=hunter2 connect_timeout=1"
    with pytest.raises(DBConnectError) as exc_info:
        run(bad_dsn)

    msg = str(exc_info.value)
    assert "hunter2" not in msg, f"raw password leaked in exception message: {msg!r}"
    assert "***" in msg, f"redaction marker '***' missing from exception message: {msg!r}"
