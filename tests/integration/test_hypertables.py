"""STOR-02: observations + metrics are TimescaleDB hypertables with explicit chunk_time_interval."""

import psycopg
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "table,expected_chunk_interval_days",
    [
        ("observations", 1),
        ("metrics", 7),
    ],
)
def test_is_hypertable_with_chunk_interval(
    tsdb_dsn: str,
    table: str,
    expected_chunk_interval_days: int,
) -> None:
    with psycopg.connect(tsdb_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT hypertable_name FROM timescaledb_information.hypertables WHERE hypertable_name = %s",
            (table,),
        )
        row = cur.fetchone()
        assert row is not None, f"{table} is not a hypertable"

        # Chunk interval check
        cur.execute(
            "SELECT integer_interval, time_interval "
            "FROM timescaledb_information.dimensions "
            "WHERE hypertable_name = %s AND dimension_number = 1",
            (table,),
        )
        dim = cur.fetchone()
        assert dim is not None, f"No dimension found for hypertable {table}"
        time_interval = dim[1]
        assert time_interval is not None, f"{table} has no time_interval dimension"

        # time_interval is an INTERVAL — compare via day count
        cur.execute("SELECT EXTRACT(EPOCH FROM %s::INTERVAL) / 86400", (time_interval,))
        days = float(cur.fetchone()[0])  # type: ignore[index]
        assert days == pytest.approx(expected_chunk_interval_days, rel=1e-3), (
            f"{table} chunk_time_interval = {days}d, expected {expected_chunk_interval_days}d"
        )
