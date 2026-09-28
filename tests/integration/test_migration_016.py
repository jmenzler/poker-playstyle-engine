"""Migration 016 integration test: felt_snapshot + hand_id + decision_id columns.

Tests REQUIRE a running TimescaleDB instance. They are marked ``integration``
so unit-only sweeps skip them automatically.

To run:
    TSDB_PASSWORD=<pwd> uv run pytest tests/integration/test_migration_016.py -m integration -x -v
"""

from __future__ import annotations

import psycopg
import pytest

pytestmark = pytest.mark.integration

_NEW_COLUMNS = {"felt_snapshot", "hand_id", "decision_id"}
_EXPECTED_INDEX = "idx_observations_hand_id"


def test_migration_016_adds_columns(tsdb_dsn: str) -> None:
    """Applying migration 016 adds felt_snapshot (jsonb), hand_id (text), decision_id (text)."""
    sql = open(
        __file__.replace("tests/integration/test_migration_016.py", "migrations/016_felt_hand_id.sql")
    ).read()

    with psycopg.connect(tsdb_dsn, autocommit=True) as conn:
        conn.execute(sql)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'observations'
                  AND column_name = ANY(%s)
                """,
                (list(_NEW_COLUMNS),),
            )
            rows = cur.fetchall()

    found = {r[0] for r in rows}
    missing = _NEW_COLUMNS - found
    assert not missing, f"columns missing after migration 016: {missing}"

    col_types = {r[0]: r[1] for r in rows}
    assert col_types["felt_snapshot"] == "jsonb", f"felt_snapshot type: {col_types['felt_snapshot']}"
    assert col_types["hand_id"] == "text", f"hand_id type: {col_types['hand_id']}"
    assert col_types["decision_id"] == "text", f"decision_id type: {col_types['decision_id']}"


def test_migration_016_creates_hand_id_index(tsdb_dsn: str) -> None:
    """Applying migration 016 creates idx_observations_hand_id plain index."""
    sql = open(
        __file__.replace("tests/integration/test_migration_016.py", "migrations/016_felt_hand_id.sql")
    ).read()

    with psycopg.connect(tsdb_dsn, autocommit=True) as conn:
        conn.execute(sql)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'observations'
                  AND indexname = %s
                """,
                (_EXPECTED_INDEX,),
            )
            row = cur.fetchone()

    assert row is not None, f"index {_EXPECTED_INDEX!r} not found after migration 016"


def test_migration_016_is_idempotent(tsdb_dsn: str) -> None:
    """Applying migration 016 twice raises no error (ADD COLUMN IF NOT EXISTS)."""
    sql = open(
        __file__.replace("tests/integration/test_migration_016.py", "migrations/016_felt_hand_id.sql")
    ).read()

    with psycopg.connect(tsdb_dsn, autocommit=True) as conn:
        conn.execute(sql)
        conn.execute(sql)
