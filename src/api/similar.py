"""GET /api/similar/{cluster_key} — wraps src/study/similar.find_similar (CLI-02 / D-13).

API contract: always include action_dist preview per D-13 (JSON form). The
underlying backend defaults to ``include_action_dist=False`` for CLI text
brevity; the router flips it to True so the React panel + Probe drill-down
have action_dist available without a second request.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query

from src.api.deps import get_milvus, get_tsdb
from src.study.similar import find_similar

router = APIRouter(prefix="/api", tags=["similar"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]
MilvusDep = Annotated[Any, Depends(get_milvus)]


@router.get("/similar/{cluster_key:path}")
async def get_similar(
    conn: TsdbDep,
    milvus: MilvusDep,
    cluster_key: Annotated[str, Path(...)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[dict]:
    """Top-k nearest clusters via Milvus kNN, with action_dist preview."""
    return find_similar(
        cluster_key,
        k=k,
        include_action_dist=True,
        _tsdb_conn=conn,
        _milvus=milvus,
    )
