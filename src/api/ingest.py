# rot-allow-file
"""POST /api/ingest — HH ingest pipeline orchestrator (CLI-01 / D-10).

Body field ``rebuild`` flips between incremental (default) and full rebuild.
Backend ``ingest_incremental`` reports the watermark before/after via
``SELECT MAX(ts)``. Watermark stays None when only the phase2 Milvus-write
pipeline has run on this PC (D-07-2, D-07-11e) — re-ingest idempotency is
delivered by the Milvus PK on deterministic ``decision_id``, not by the
observations watermark.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_tsdb
from src.study.ingest import ingest_incremental, ingest_rebuild

router = APIRouter(prefix="/api", tags=["ingest"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]


class IngestRequest(BaseModel):
    """Ingest invocation payload. ``rebuild=True`` is destructive — caller is
    responsible for confirming the action (UI shows a confirmation modal).
    """

    rebuild: bool = False


@router.post("/ingest")
async def run_ingest(body: IngestRequest, conn: TsdbDep) -> dict:
    """Run incremental (default) or full-rebuild ingest pipeline."""
    if body.rebuild:
        return ingest_rebuild(_tsdb_conn=conn)
    return ingest_incremental(_tsdb_conn=conn)
