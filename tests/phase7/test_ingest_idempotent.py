# rot-allow-file
"""Phase 7 / BOOT-04 regression — re-ingest is idempotent via Milvus PK
on deterministic decision_id (operative idempotency layer per D-07-11a).

NOT via a TSDB UNIQUE constraint — observations remains sim-only and Phase 2
writes to Milvus only. The Milvus PK is on
    decision_id = f"{hand_id}_dp{decision_idx}"  (tools/extract_decisions.py:266)
which is deterministic per source row, so re-running the same HH source
produces zero new entities.

PC-gated: requires a live Milvus instance and HH source files on disk.
Marked integration so it skips on CI / dev boxes lacking MILVUS_HOST.
"""

from __future__ import annotations

import os
import subprocess

import pytest

pytestmark = pytest.mark.integration

if not os.environ.get("MILVUS_HOST"):
    pytestmark = [
        pytest.mark.integration,
        pytest.mark.skip(reason="MILVUS_HOST not set — deferred to PC"),
    ]


def _row_counts(client) -> dict[str, int]:
    return {
        "preflop_decisions": client.get_collection_stats("preflop_decisions")["row_count"],
        "postflop_decisions": client.get_collection_stats("postflop_decisions")["row_count"],
    }


def test_reingest_produces_zero_new_milvus_rows(milvus_uri, milvus_token):
    """BOOT-04 (D-07-11a): re-running phase2_pipeline against the same HH source
    produces zero new entities in preflop_decisions + postflop_decisions because
    the Milvus PK on ``decision_id`` is deterministic.

    Operative invariant — re-ingest dedup lives at the Milvus PK layer, not
    at a TSDB UNIQUE (which would be illegal on a hypertable without the
    partition column ``ts`` in the constraint; the legal shape
    ``UNIQUE (obs_id, ts)`` is already the PK, and sim writer generates fresh
    uuid4() per row, so a hypothetical UNIQUE (obs_id) would never fire).
    """
    from pymilvus import MilvusClient

    client = MilvusClient(uri=milvus_uri, token=milvus_token)
    n_pre = _row_counts(client)

    # Re-run phase2 pipeline against the SAME source (no new HH data).
    result = subprocess.run(
        ["uv", "run", "python", "-m", "tools.phase2_pipeline", "--all"],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, f"pipeline re-run failed: {result.stderr}"

    n_post = _row_counts(client)
    assert n_post == n_pre, (
        f"Milvus row count changed on re-ingest: pre={n_pre} post={n_post}. "
        "BOOT-04 idempotency violated — check decision_id construction in "
        "tools/extract_decisions.py:266."
    )
