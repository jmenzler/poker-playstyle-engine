"""POST /api/edit-node — manual action_dist edit via PatchEngine (CLI-03 / D-11).
POST /api/edit-node/lock     — toggle locked_from_autoloop (D-NEW-26).

Backend validation (sum=1.0±0.001, no unknown keys, no negatives) raises
ValidationError — mapped here to HTTP 400 with the error message in the
response detail field (T-06-19 mitigation visible to the operator UI).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src._errors import ValidationError
from src.api.deps import get_milvus, get_tsdb
from src.study.edit_node import edit_node, set_lock_from_autoloop

router = APIRouter(prefix="/api", tags=["edit-node"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]
MilvusDep = Annotated[Any, Depends(get_milvus)]


class EditNodeRequest(BaseModel):
    """Manual edit payload. ``action_dist`` keys must be in CANONICAL_ACTIONS
    and frequencies must sum to 1.0 ± 0.001 — enforced by the backend.
    """

    cluster_key: str
    action_dist: dict[str, float]
    reason: str = ""


class LockRequest(BaseModel):
    """Lock/unlock a cluster from autoloop processing (D-NEW-26)."""

    cluster_key: str
    locked: bool


@router.post("/edit-node")
async def edit(
    body: EditNodeRequest,
    conn: TsdbDep,
    milvus: MilvusDep,
) -> dict:
    """Apply a manual action_dist edit via PatchEngine. Returns patch_id +
    new_node_id. ValidationError → 400.
    """
    try:
        return edit_node(
            cluster_key=body.cluster_key,
            action_dist=body.action_dist,
            reason=body.reason,
            _tsdb_conn=conn,
            _milvus=milvus,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/edit-node/lock")
async def lock(body: LockRequest, conn: TsdbDep) -> dict:
    """Set ``locked_from_autoloop`` for an active strategy_node.

    404 when no active row matches (cannot lock a cluster the engine has
    never seen).
    """
    n = set_lock_from_autoloop(body.cluster_key, body.locked, _tsdb_conn=conn)
    if n == 0:
        raise HTTPException(
            status_code=404,
            detail=f"no active strategy_node for cluster_key={body.cluster_key!r}",
        )
    return {"cluster_key": body.cluster_key, "locked": body.locked, "n_updated": n}
