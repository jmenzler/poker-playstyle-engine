"""Phase 5 Plans 02 + 08: PTCH-01 enforcement — schema + runtime + codebase.

Verifies three layers of the "PatchEngine is the sole writer" invariant:
1. (Plan 02) Schema CHECK constraint rejects invalid source values.
2. (Plan 02) source='autoloop' is accepted after migration 010.
3. (Plan 08) PatchEngine.apply is the sole runtime path:
   - Direct INSERT INTO strategy_nodes succeeds at the DB level (schema does not
     prevent it — that is the pre-commit hook's job).
   - A subprocess grep over src/ (excluding patch_engine.py) returns EMPTY, proving
     no module writes to strategy_nodes directly in the codebase.

Requires: migration 010 applied on the PC TimescaleDB.
Run with: pytest -m integration tests/integration/test_no_direct_strategy_nodes_write.py -v
"""

from __future__ import annotations

import subprocess
import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_invalid_source_rejected_by_check_constraint(tsdb_dsn: str) -> None:
    """source='INVALID_SOURCE' raises CheckViolation from the strategy_nodes_source_check constraint.

    Mirrors test_strategy_nodes_constraints.py::test_source_check_constraint with an explicit
    BEGIN/ROLLBACK envelope to leave no side-effect in the DB.
    """
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO strategy_nodes "
                    "(node_id, cluster_key, embedding, action_dist, gto_score, confidence, source) "
                    "VALUES (gen_random_uuid(), 'test', '{0.0}'::real[], '{}'::jsonb, "
                    "0.5, 0.5, 'INVALID_SOURCE')"
                )
        finally:
            cur.execute("ROLLBACK")


def test_autoloop_source_accepted_by_check_constraint(tsdb_dsn: str) -> None:
    """source='autoloop' succeeds after migration 010 extends the CHECK constraint.

    Inserts a row with source='autoloop', asserts no exception, then ROLLBACKs to
    leave no side-effect. This is the schema-layer proof that migration 010 was applied.
    """
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            # Should NOT raise — migration 010 added 'autoloop' to the CHECK constraint
            cur.execute(
                "INSERT INTO strategy_nodes "
                "(node_id, cluster_key, embedding, action_dist, gto_score, confidence, source) "
                "VALUES (gen_random_uuid(), 'test_autoloop', '{0.0}'::real[], '{}'::jsonb, "
                "0.5, 0.5, 'autoloop')"
            )
            # If we reach here, the INSERT succeeded (no CheckViolation)
            # Verify one row exists with our test cluster_key
            cur.execute(
                "SELECT count(*) FROM strategy_nodes "
                "WHERE cluster_key = 'test_autoloop' AND source = 'autoloop'"
            )
            row = cur.fetchone()
            assert row is not None and row[0] == 1, "Expected 1 autoloop row in strategy_nodes after INSERT"
        finally:
            cur.execute("ROLLBACK")


def test_patch_engine_apply_is_the_sole_runtime_writer(
    tsdb_dsn: str,
    milvus_uri: str,
    milvus_token: str | None,
) -> None:
    """PTCH-01 runtime enforcement: PatchEngine is the ONLY writer to strategy_nodes.

    Three-part proof:
      Part A) PatchEngine.apply writes a row that is immediately readable via direct SELECT.
              This proves the writer works correctly.
      Part B) A direct INSERT INTO strategy_nodes (bypassing PatchEngine) SUCCEEDS at the
              DB level. The schema CHECK constraint does not prevent this — the pre-commit
              hook is the lint-time guard; this test documents that layered defense
              (schema allows it; only the hook + codebase discipline prevent it).
      Part C) A subprocess grep across src/ (excluding patch_engine.py) for
              "INSERT INTO strategy_nodes" or "UPDATE strategy_nodes" returns EMPTY.
              This is the codebase-contract proof: no other module writes to the table.

    Cleanup in finally: delete all test rows by cluster_key.
    """
    import os
    import pathlib

    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    unique_suffix = str(uuid.uuid4())[:8]
    cluster_key_pe = (
        f"hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop|ptch01={unique_suffix}"
    )
    cluster_key_direct = (
        f"hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop|direct={unique_suffix}"
    )
    embedding = [0.0] * 80

    # Inject Milvus env (PatchEngine uses env for Milvus host)
    original_milvus_host = os.environ.get("MILVUS_HOST", "")
    original_milvus_port = os.environ.get("MILVUS_PORT", "")
    host_port = milvus_uri.replace("http://", "").split(":")
    os.environ["MILVUS_HOST"] = host_port[0]
    if len(host_port) > 1:
        os.environ["MILVUS_PORT"] = host_port[1]

    try:
        with psycopg.connect(tsdb_dsn) as conn:
            # ── Part A: PatchEngine.apply writes and is readable via SELECT ──
            engine = PatchEngine()
            spec = PatchSpec(
                cluster_key=cluster_key_pe,
                action_dist={"fold": 0.1, "call": 0.2, "bet_50": 0.7},
                embedding=embedding,
                gto_score=0.5,
                confidence=0.5,
                prev_node_id=None,
                source="autoloop",
            )
            validation = ValidationResult(
                seed=0,
                ev_loss_delta=0.05,
                n_hands=50,
                confidence_interval=(0.01, 0.1),
                n_cluster_hits=5,
            )
            record = engine.apply(spec, validation=validation, _tsdb_conn=conn)

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source, active FROM strategy_nodes WHERE cluster_key = %s AND node_id = %s",
                    (cluster_key_pe, record.new_node_id),
                )
                row = cur.fetchone()

            assert row is not None, "PatchEngine.apply row not found via direct SELECT (Part A)"
            assert row[0] == "autoloop", f"source must be 'autoloop'; got {row[0]!r} (Part A)"
            assert row[1] is True, f"active must be TRUE after apply; got {row[1]} (Part A)"

            # ── Part B: Direct INSERT bypassing PatchEngine SUCCEEDS at DB level ──
            # This is intentional: the schema CHECK does not block it; only the pre-commit
            # hook and codebase discipline prevent direct writes in production.
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO strategy_nodes "
                    "(node_id, cluster_key, embedding, action_dist, gto_score, confidence, source) "
                    "VALUES (%s, %s, %s::real[], %s::jsonb, 0.5, 0.5, 'autoloop')",
                    (uuid.uuid4(), cluster_key_direct, embedding, '{"fold": 1.0}'),
                )
                # Verify the direct INSERT created a row
                cur.execute(
                    "SELECT count(*) FROM strategy_nodes WHERE cluster_key = %s",
                    (cluster_key_direct,),
                )
                direct_count = cur.fetchone()

            assert direct_count is not None and direct_count[0] == 1, (
                "Direct INSERT bypassing PatchEngine must succeed at DB level "
                "(schema allows it; pre-commit hook is the lint-time guard)"
            )
            conn.commit()

            # ── Part C: Codebase grep — no module writes strategy_nodes directly ──
            src_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "src"
            result = subprocess.run(
                [
                    "grep",
                    "-rE",
                    "INSERT INTO strategy_nodes|UPDATE strategy_nodes",
                    str(src_dir),
                    "--include=*.py",
                ],
                capture_output=True,
                text=True,
            )
            # Filter out patch_engine.py (the allowlisted writer)
            grep_lines = [line for line in result.stdout.splitlines() if "patch_engine.py" not in line]
            assert grep_lines == [], (
                "PTCH-01: codebase-grep found direct strategy_nodes writes outside patch_engine.py:\n"
                + "\n".join(grep_lines)
                + "\nAll strategy_nodes writes must go through PatchEngine."
            )

    finally:
        # Cleanup both test rows
        with psycopg.connect(tsdb_dsn) as cleanup_conn, cleanup_conn.cursor() as cur:
            cur.execute("DELETE FROM patches WHERE cluster_key = %s", (cluster_key_pe,))
            cur.execute("DELETE FROM strategy_nodes WHERE cluster_key = %s", (cluster_key_pe,))
            cur.execute("DELETE FROM strategy_nodes WHERE cluster_key = %s", (cluster_key_direct,))
            cleanup_conn.commit()

        # Restore Milvus env
        os.environ["MILVUS_HOST"] = original_milvus_host
        os.environ["MILVUS_PORT"] = original_milvus_port
