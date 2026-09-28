# rot-allow-file
"""ObservationWriter: batched INSERT into the observations hypertable (SIM-03).

Buffers per-decision rows in memory; flushes via psycopg ``executemany`` in
batches (default 500). The harness is responsible for calling ``close()`` at
end of session to flush any tail rows.

Schema (migrations/001_observations.sql):
    obs_id UUID, cluster_key TEXT, embedding REAL[], action_taken TEXT,
    ev_realized NUMERIC NULL, source TEXT ('hh'|'sim'),
    session_id TEXT, solver_label JSONB NULL, ts TIMESTAMPTZ
    PK: (obs_id, ts)

This writer populates obs_id, cluster_key, embedding, action_taken, source,
session_id, ts, flagged_sparse, max_neighbor_distance, hand_id, decision_id,
felt_snapshot.
ev_realized and solver_label are populated by Phase 4+ labeling.

Column order in _INSERT_SQL (12 columns):
    0  obs_id                 UUID
    1  cluster_key            TEXT
    2  embedding              REAL[]
    3  action_taken           TEXT
    4  source                 TEXT ('hh'|'sim')
    5  session_id             TEXT
    6  ts                     TIMESTAMPTZ
    7  flagged_sparse         BOOL NOT NULL DEFAULT FALSE  (Phase 4 Plan 02)
    8  max_neighbor_distance  REAL NULL                    (Phase 4 Plan 02)
    9  hand_id                TEXT NULL                    (Phase 10 Plan 02)
   10  decision_id            TEXT NULL                    (Phase 10 Plan 02)
   11  felt_snapshot          JSONB NULL (json.dumps'd)    (Phase 10 Plan 02)

ERR-01 contract: connections are opened via ``src.db.timescale.connect``, never
``psycopg.connect`` directly. The optional ``_conn`` kwarg accepts a pre-built
connection (mock) for unit tests; production code does not use it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from src._log import get_logger
from src.db import timescale

log = get_logger("sim.observation_writer")

DEFAULT_BATCH_SIZE = 500
PROGRESS_EVERY = 1_000  # OBS-03 throughput logging cadence

_INSERT_SQL = (
    "INSERT INTO observations "
    "(obs_id, cluster_key, embedding, action_taken, source, session_id, ts, "
    "flagged_sparse, max_neighbor_distance, hand_id, decision_id, felt_snapshot) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


class ObservationWriter:
    """Buffer observation rows; flush to TimescaleDB in batches.

    Always uses ``src.db.timescale.connect`` (ERR-01 fail-loud + redacted DSN).
    """

    def __init__(
        self,
        dsn: str,
        session_id: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        _conn=None,
    ) -> None:
        """Open writer.

        Args:
            dsn: TimescaleDB DSN; passed to ``src.db.timescale.connect``.
            session_id: tag for all rows written by this writer.
            batch_size: flush threshold; tuned for INSERT round-trip cost (~1ms each).
            _conn: optional pre-built connection (tests inject a mock).
                When non-None, ``dsn`` is ignored. NOT for production use.
        """
        self._conn = _conn if _conn is not None else timescale.connect(dsn)
        self._session_id = session_id
        self._batch: list[tuple] = []
        self._batch_size = batch_size
        self._total_written = 0

    def record(
        self,
        cluster_key: str,
        embedding: Sequence[float],
        action_taken: str,
        *,
        flagged_sparse: bool = False,
        max_neighbor_distance: float | None = None,
        hand_id: str | None = None,
        decision_id: str | None = None,
        felt_snapshot: dict | None = None,
    ) -> None:
        """Append one row to the buffer; auto-flush when full.

        Args:
            cluster_key: canonical cluster key from enc.hard_filter.
            embedding: float vector from EncodeResult.embedding.
            action_taken: canonical action string from decide_with_encoding.
            flagged_sparse: True when kNN neighborhood was sparse at decide-time.
                Defaults to False for backward compatibility.
            max_neighbor_distance: maximum COSINE distance among kNN neighbors
                at decide-time. None for NoStrategy fallback (zero neighbors)
                and for pre-migration rows. psycopg v3 maps None -> SQL NULL.
            hand_id: minted by harness as {session_id}_h{hand_idx}. None for
                HM-ingest rows.
            decision_id: minted by harness as {hand_id}_dp{decisions_this_hand}.
                None for HM-ingest rows.
            felt_snapshot: the 11 GameState fields verbatim (msgspec-decoded).
                json.dumps'd before storage; None for HM-ingest rows.
        """
        self._batch.append(
            (
                str(uuid.uuid4()),
                cluster_key,
                list(embedding),
                action_taken,
                "sim",
                self._session_id,
                datetime.now(UTC),
                flagged_sparse,
                max_neighbor_distance,
                hand_id,
                decision_id,
                json.dumps(felt_snapshot) if felt_snapshot is not None else None,
            )
        )
        if len(self._batch) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        """Execute buffered rows in a single executemany; commit; clear buffer.

        Passes a copy of the buffer to executemany so callers/mocks that hold a
        reference to the call args still see the dispatched rows after the
        internal buffer is cleared.
        """
        if not self._batch:
            return
        rows = list(self._batch)  # snapshot — keep call_args inspectable post-clear
        n = len(rows)
        with self._conn.cursor() as cur:
            cur.executemany(_INSERT_SQL, rows)
        self._conn.commit()
        self._total_written += n
        self._batch.clear()
        # OBS-03: log first flush (cold-start signal) and every PROGRESS_EVERY rows.
        if self._total_written < PROGRESS_EVERY or self._total_written % PROGRESS_EVERY == 0:
            log.info(
                "obs_writer.flush",
                n=n,
                total=self._total_written,
                session_id=self._session_id,
            )

    def close(self) -> int:
        """Flush remaining rows and close the connection.

        Returns the total number of rows written across the writer's lifetime.
        Safe to call even when the buffer is empty.
        """
        try:
            self._flush()
        finally:
            self._conn.close()
        return self._total_written

    def __enter__(self) -> ObservationWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def total_written(self) -> int:
        """Number of rows successfully flushed to the database."""
        return self._total_written
