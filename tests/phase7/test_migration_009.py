"""METR-01..03 audit closure: pre-apply duplicate audit + post-apply UNIQUE introspection.

Integration tests against an explicitly configured disposable TimescaleDB:

    TSDB_HOST=127.0.0.1 TSDB_PORT=55432 TSDB_USER=poker \\
    TSDB_PASSWORD=<pwd> TSDB_DB=poker_engine \\
    uv run pytest tests/phase7/test_migration_009.py -m integration -x -v

See `.planning/phases/07-close-gap-boot-cli-watermark/07-RESEARCH.md` §Focus Area 3
for the dedup query if pre-apply audit returns non-zero.
"""

import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_pre_apply_audit_clean(tsdb_dsn):
    """METR-01..03 (audit, D-07-11c): pre-migration metrics dup check returns zero rows.

    Documents the pre-flight requirement. Run before applying migration 009 on PC.
    If non-zero, dedup via:
        DELETE FROM metrics a USING metrics b
        WHERE a.ts > b.ts AND a.session_id = b.session_id
          AND a.metric_name = b.metric_name
          AND a.cluster_key IS NOT DISTINCT FROM b.cluster_key;
    """
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT session_id, metric_name, cluster_key, ts, COUNT(*) AS n
              FROM metrics
              GROUP BY session_id, metric_name, cluster_key, ts
              HAVING COUNT(*) > 1
            ) d
            """
        )
        dup_count = cur.fetchone()[0]
    assert dup_count == 0, (
        f"Pre-migration dup audit FAILED: {dup_count} duplicate "
        "(session_id, metric_name, cluster_key, ts) tuples in metrics. "
        "Run RESEARCH §Focus Area 3 dedup query before applying migration 009."
    )


def test_migration_009_constraint_present(tsdb_dsn):
    """Schema introspection: confirm metrics_unique_per_session UNIQUE exists post-apply."""
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT constraint_type
            FROM information_schema.table_constraints
            WHERE table_name = 'metrics'
              AND constraint_name = 'metrics_unique_per_session'
            """
        )
        row = cur.fetchone()
    assert row is not None, "metrics_unique_per_session constraint not present — migration 009 not applied?"
    assert row[0] == "UNIQUE"
