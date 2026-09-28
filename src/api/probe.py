"""Probe panel endpoints (D-NEW-24).

Mode A: GET /api/probe?cluster_key=...  — query a known cluster_key.
Mode B: POST /api/probe/spot            — encode a spot dict to a cluster_key,
                                          then dispatch to Mode A.

Both modes return the same 3-section response: engine_response, knn_neighbors,
solver_truth (D-NEW-24 / 06-04-SUMMARY §probe).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src._errors import CanonicalizeError
from src.api.deps import get_milvus, get_tsdb
from src.study.probe import probe_by_cluster_key, probe_by_spot, probe_range_for_actor

router = APIRouter(prefix="/api", tags=["probe"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]
MilvusDep = Annotated[Any, Depends(get_milvus)]


class SpotProbeRequest(BaseModel):
    """Mode B body — operator submits a partial spot dict; backend encodes
    it via the frozen z-score manifests into a canonical cluster_key.
    """

    spot: dict
    k: int = Field(default=5, ge=1, le=20)


class EncodeProbeRequest(BaseModel):
    """Encode-only body — a partial spot dict, no k (no query is run)."""

    spot: dict


class RangeBatchRequest(BaseModel):
    """Batch-encode 169 starting hands for one actor node."""

    spot: dict
    actor_position: str


@router.get("/probe")
async def probe_get(
    conn: TsdbDep,
    milvus: MilvusDep,
    cluster_key: Annotated[str, Query(...)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> dict:
    """Mode A — probe by cluster_key."""
    return probe_by_cluster_key(cluster_key, k=k, _tsdb_conn=conn, _milvus=milvus)


@router.post("/probe/spot")
async def probe_post(
    body: SpotProbeRequest,
    conn: TsdbDep,
    milvus: MilvusDep,
) -> dict:
    """Mode B — probe by spot dict (encoded to cluster_key by the backend)."""
    try:
        return probe_by_spot(body.spot, k=body.k, _tsdb_conn=conn, _milvus=milvus)
    except (CanonicalizeError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/probe/encode")
async def probe_encode(body: EncodeProbeRequest) -> dict:
    """Live preview — encode a spot dict to its cluster_key, no DB/Milvus query."""
    try:
        from tools.build_embedding import encode_spot_to_cluster_key
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"encoder not available: {e}") from e

    try:
        cluster_key = encode_spot_to_cluster_key(body.spot)
    except (CanonicalizeError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"cluster_key": cluster_key, "empty_reason": None}


@router.post("/probe/range-batch")
async def probe_range_batch(
    body: RangeBatchRequest,
    conn: TsdbDep,
    milvus: MilvusDep,
) -> dict:
    """Batch-encode all 169 starting hands for one actor node."""
    try:
        return await run_in_threadpool(
            probe_range_for_actor,
            body.spot,
            body.actor_position,
            _tsdb_conn=conn,
            _milvus=milvus,
        )
    except (CanonicalizeError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
