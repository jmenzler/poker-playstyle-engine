"""Patches audit log endpoints (CLI-07 / D-14 / D-NEW-29).

GET /api/patches                       — reverse-chrono list with optional filters.
GET /api/patches/{patch_id}            — single-patch detail + chain (collapsible tree).
POST /api/patches/{patch_id}/rollback  — roll back a previously applied patch.
"""

from __future__ import annotations

import uuid as _uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_milvus, get_tsdb
from src.cli.rollback import RollbackResult, rollback_patch
from src.study.patches import list_patches, patch_detail

router = APIRouter(prefix="/api", tags=["patches"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]
MilvusDep = Annotated[Any, Depends(get_milvus)]


@router.get("/patches")
async def list_(
    conn: TsdbDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 20,
    cluster_key: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """Reverse-chronological patches list with chain metadata."""
    return list_patches(limit=limit, cluster_key=cluster_key, source=source, _tsdb_conn=conn)


@router.get("/patches/{patch_id}")
async def detail(
    patch_id: str,
    conn: TsdbDep,
    include_chain: bool = True,
) -> dict:
    """Single-patch detail + optional chronological chain. 404 when not found."""
    out = patch_detail(patch_id, include_chain=include_chain, _tsdb_conn=conn)
    if out is None:
        raise HTTPException(status_code=404, detail=f"patch {patch_id!r} not found")
    return out


@router.post("/patches/{patch_id}/rollback")
async def rollback(
    patch_id: str,
    conn: TsdbDep,
    milvus: MilvusDep,
) -> dict:
    """Roll back a previously applied patch.

    Returns a rollback receipt: {rollback_patch_id, restored_node_id, source, ts}.

    Raises:
        422: patch_id is not a valid UUID.
        404: patch_id not found in the patches table.
        409: patch already rolled back (idempotency guard).
    """
    try:
        patch_uuid = _uuid.UUID(patch_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid patch_id format: {patch_id!r}") from exc

    try:
        result: RollbackResult = rollback_patch(patch_uuid, tsdb_conn=conn, milvus_client=milvus)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        # "patch already rolled back" — idempotency guard
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "rollback_patch_id": result.rollback_patch_id,
        "restored_node_id": result.restored_node_id,
        "source": result.source,
        "ts": result.ts,
    }
