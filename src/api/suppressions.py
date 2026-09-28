"""Leak-suppressions endpoints (D-NEW-28).

POST    /api/suppressions               — insert an active suppression.
DELETE  /api/suppressions/{cluster_key} — deactivate all active rows for cluster.
GET     /api/suppressions/{cluster_key} — check whether an active suppression exists.

Append-only writes; ``unsuppress`` flips ``active = FALSE`` rather than deleting,
preserving audit history (per migration 012).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_tsdb
from src.study.suppressions import is_suppressed, suppress, unsuppress

router = APIRouter(prefix="/api", tags=["suppressions"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]


class SuppressionRequest(BaseModel):
    cluster_key: str
    reason: str = ""


@router.post("/suppressions")
async def add(body: SuppressionRequest, conn: TsdbDep) -> dict:
    """Insert a new active suppression. Returns ``suppression_id`` (UUID4)."""
    sid = suppress(body.cluster_key, reason=body.reason, _tsdb_conn=conn)
    return {"suppression_id": sid, "cluster_key": body.cluster_key, "active": True}


@router.delete("/suppressions/{cluster_key:path}")
async def remove(cluster_key: str, conn: TsdbDep) -> dict:
    """Deactivate all active suppressions for the cluster_key."""
    n = unsuppress(cluster_key, _tsdb_conn=conn)
    return {"cluster_key": cluster_key, "n_deactivated": n}


@router.get("/suppressions/{cluster_key:path}")
async def check(cluster_key: str, conn: TsdbDep) -> dict:
    """Return whether an active suppression exists for the cluster_key."""
    return {"cluster_key": cluster_key, "active": is_suppressed(cluster_key, _tsdb_conn=conn)}
