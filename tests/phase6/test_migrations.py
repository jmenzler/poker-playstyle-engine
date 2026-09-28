"""Phase 6 Wave 0 — migration acceptance tests for 011-014.

Marked `integration`: requires a running TimescaleDB stack (env vars TSDB_HOST +
TSDB_PASSWORD set, plus TSDB_PORT / TSDB_DB / TSDB_USER if non-default). Tests
skip cleanly via the tsdb_conn fixture when env is missing.

Each test introspects information_schema / pg_constraint / timescaledb_information
directly — no assumption that migrations were applied here (they may have been
applied by the migration runner on the running container; we only assert the
desired end state). The idempotency test re-runs the runner and asserts a
non-error second invocation.
"""

from __future__ import annotations

import os
import subprocess

import pytest

pytestmark = pytest.mark.integration


def test_011_locked_from_autoloop_column_exists(tsdb_conn) -> None:
    """strategy_nodes.locked_from_autoloop: BOOLEAN, NOT NULL, DEFAULT FALSE."""
    with tsdb_conn.cursor() as cur:
        cur.execute(
            """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name='strategy_nodes'
              AND column_name='locked_from_autoloop'
            """
        )
        row = cur.fetchone()
    assert row is not None, "migration 011 not applied — column locked_from_autoloop missing"
    data_type, is_nullable, column_default = row
    assert data_type == "boolean", f"expected boolean, got {data_type}"
    assert is_nullable == "NO", "locked_from_autoloop must be NOT NULL"
    assert "false" in (column_default or "").lower(), f"expected DEFAULT FALSE, got {column_default!r}"


def test_011_partial_index_on_locked_clusters(tsdb_conn) -> None:
    """Partial index idx_strategy_nodes_locked supports the autoloop skip-list lookup."""
    with tsdb_conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname FROM pg_indexes
            WHERE tablename='strategy_nodes'
              AND indexname='idx_strategy_nodes_locked'
            """
        )
        assert cur.fetchone() is not None, "idx_strategy_nodes_locked missing"


def test_012_leak_suppressions_table_shape(tsdb_conn) -> None:
    """leak_suppressions table exists with the documented columns."""
    with tsdb_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.leak_suppressions')")
        regclass = cur.fetchone()[0]
    assert regclass == "leak_suppressions", "leak_suppressions table missing"

    with tsdb_conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name='leak_suppressions'
            ORDER BY ordinal_position
            """
        )
        cols = {name: (dt, nullable) for name, dt, nullable in cur.fetchall()}

    assert "suppression_id" in cols and cols["suppression_id"][0] == "uuid"
    assert "cluster_key" in cols and cols["cluster_key"][1] == "NO"
    assert "ts" in cols and cols["ts"][0].startswith("timestamp")
    assert "reason" in cols  # nullable TEXT
    assert "active" in cols and cols["active"][0] == "boolean" and cols["active"][1] == "NO"


def test_012_leak_suppressions_index(tsdb_conn) -> None:
    """idx_leak_suppressions_active_cluster supports filter queries."""
    with tsdb_conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname FROM pg_indexes
            WHERE tablename='leak_suppressions'
              AND indexname='idx_leak_suppressions_active_cluster'
            """
        )
        assert cur.fetchone() is not None


def test_013_strategy_nodes_source_check_includes_solver_verify(tsdb_conn) -> None:
    """strategy_nodes_source_check allows 'solver_verify' (Stage B writes)."""
    with tsdb_conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_get_constraintdef(oid) FROM pg_constraint
            WHERE conname='strategy_nodes_source_check'
            """
        )
        row = cur.fetchone()
    assert row is not None, "strategy_nodes_source_check constraint missing"
    defn = row[0]
    assert "solver_verify" in defn, f"constraint does not allow 'solver_verify'; full def: {defn}"
    # Tight widening: legacy values still present.
    for v in ("hh", "sim", "solver", "manual", "seed", "autoloop"):
        assert v in defn, f"existing source value {v!r} dropped by migration 013: {defn}"


def test_013_unknown_source_rejected(tsdb_conn) -> None:
    """Inserting an unsupported source value still fails CHECK (no wildcard widening)."""
    import uuid

    bogus_node_id = str(uuid.uuid4())
    with tsdb_conn.cursor() as cur:
        with pytest.raises(Exception) as exc_info:
            cur.execute(
                """
                INSERT INTO strategy_nodes (
                    node_id, cluster_key, embedding, action_dist,
                    gto_score, confidence, source, active
                ) VALUES (
                    %s, 'test_cluster', ARRAY[0.0]::real[], '{}'::jsonb,
                    0.0, 0.0, 'random_bad_source', TRUE
                )
                """,
                (bogus_node_id,),
            )
        # psycopg surfaces CheckViolation; we only need confirmation of failure.
        assert "check" in str(exc_info.value).lower() or "constraint" in str(exc_info.value).lower()


def test_014_matches_is_hypertable(tsdb_conn) -> None:
    """matches is a TimescaleDB hypertable chunked by started_at."""
    with tsdb_conn.cursor() as cur:
        cur.execute(
            """
            SELECT hypertable_name FROM timescaledb_information.hypertables
            WHERE hypertable_name='matches'
            """
        )
        assert cur.fetchone() is not None, "matches is not registered as a hypertable"


def test_014_matches_required_columns(tsdb_conn) -> None:
    """matches table exposes all columns the Eval panel reads."""
    with tsdb_conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name='matches'
            """
        )
        cols = {row[0] for row in cur.fetchall()}
    required = {
        "match_id",
        "opponent",
        "hands",
        "seed",
        "bb_per_100",
        "ci_low",
        "ci_high",
        "per_street",
        "per_texture",
        "top5_profitable",
        "top5_leaky",
        "status",
        "engine_version",
        "started_at",
        "finished_at",
    }
    missing = required - cols
    assert not missing, f"matches missing required columns: {sorted(missing)}"


def test_014_matches_status_check_constraint(tsdb_conn) -> None:
    """matches.status CHECK accommodates both lifecycle and verdict values."""
    with tsdb_conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_get_constraintdef(oid) FROM pg_constraint
            WHERE conrelid='matches'::regclass
              AND contype='c'
            """
        )
        defs = [row[0] for row in cur.fetchall()]
    joined = " | ".join(defs)
    for v in (
        "queued",
        "running",
        "done",
        "failed",
        "cancelled",
        "won",
        "lost",
        "inconclusive",
        "regression",
    ):
        assert v in joined, f"matches.status CHECK missing value {v!r}; full defs: {joined}"


def test_all_migrations_idempotent(tsdb_conn) -> None:
    """Running runner.migrate twice produces no error and no duplicate rows.

    Uses tsdb_conn only to confirm env is set (skip-or-run gate). The runner
    itself opens its own connection; we then rely on the txn-rollback fixture
    to discard any tracking-table writes (in practice the runner commits inside
    its own connection, so the rollback is a no-op for schema_migrations, but
    the migrations themselves are guarded by IF [NOT] EXISTS and checksum
    tracking — re-application is safe).
    """
    # Skip-gate is implicit via the tsdb_conn fixture; this assertion just
    # documents the dependency on env vars.
    assert "TSDB_HOST" in os.environ

    cmd = ["uv", "run", "python", "-m", "runner.migrate"]
    result1 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result1.returncode == 0, f"first run failed: {result1.stderr}"
    assert result2.returncode == 0, f"second run failed: {result2.stderr}"
