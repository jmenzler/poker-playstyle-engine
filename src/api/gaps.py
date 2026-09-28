"""FastAPI gap-resolution endpoints: queue, metadata, resolve, send-to-solver, history."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src._errors import ValidationError
from src.api.deps import get_milvus, get_tsdb
from src.study.gaps import get_edit_history, get_gap_metadata, list_gaps, resolve_gap, send_to_solver

router = APIRouter(prefix="/api", tags=["gaps"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]
MilvusDep = Annotated[Any, Depends(get_milvus)]


class ResolveGapRequest(BaseModel):
    action_dist: dict[str, float]


@router.get("/gaps")
async def list_gaps_endpoint(
    conn: TsdbDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    filter: Annotated[str, Query(pattern="^(all|hu|multiway)$")] = "all",
) -> list[dict]:
    return list_gaps(limit=limit, filter=filter, _tsdb_conn=conn)


@router.get("/gaps/{decision_id}")
async def get_gap(decision_id: str, conn: TsdbDep) -> dict:
    result = get_gap_metadata(decision_id, _tsdb_conn=conn)
    if result is None:
        raise HTTPException(status_code=404, detail=f"gap {decision_id!r} not found")
    return result


@router.post("/gaps/{decision_id}/resolve")
async def resolve(decision_id: str, body: ResolveGapRequest, conn: TsdbDep, milvus: MilvusDep) -> dict:
    # resolve_gap + PatchEngine.apply manage their own `conn.transaction()` blocks and rely on
    # those committing TSDB BEFORE the post-commit Milvus upsert. On the autocommit=False dep conn
    # those blocks nest as savepoints under the read txn and never commit, so get_tsdb's teardown
    # rollback silently discards the patch. Autocommit makes each transaction() a real commit
    # boundary (and preserves TSDB-first ordering).
    conn.autocommit = True
    try:
        return resolve_gap(decision_id, body.action_dist, _tsdb_conn=conn, _milvus=milvus)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/gaps/{decision_id}/send-to-solver")
async def send_solver(decision_id: str, conn: TsdbDep) -> dict:
    conn.autocommit = True  # commit the DELETE; see resolve() above
    try:
        return send_to_solver(decision_id, _tsdb_conn=conn)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/gaps/{decision_id}/history")
async def gap_history(decision_id: str, conn: TsdbDep) -> list[dict]:
    # Keyed directly off decision_id so history still resolves for already-patched
    # DPs, which are no longer present in the open-gap queue.
    return get_edit_history(decision_id, _tsdb_conn=conn)
