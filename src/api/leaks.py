"""GET /api/leaks — wraps src/study/leaks.rank_leaks (CLI-04 / D-12).

Returns the same dict shape the CLI's --format json emits (D-09 single source
of truth): ``{coverage: [...], strategy: [...], bucket_stats: {...}}``.

Filters baked into the SQL via the backend:
    * min_n (D-08 — Strategy-leak observation gate, default 20).
    * locked_from_autoloop (D-NEW-26 — excludes user-locked clusters).
    * leak_suppressions (D-NEW-28 — excludes operator-suppressed clusters).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_tsdb
from src.study.leaks import rank_leaks

router = APIRouter(prefix="/api", tags=["leaks"])


TsdbDep = Annotated[Any, Depends(get_tsdb)]


@router.get("/leaks")
async def list_leaks(
    conn: TsdbDep,
    type: Annotated[str, Query(pattern="^(coverage|strategy|both)$")] = "both",
    min_n: Annotated[int, Query(ge=1, le=10000)] = 20,
    limit: Annotated[int, Query(ge=1, le=500)] = 20,
) -> dict:
    """List leaks — Type 1 coverage + Type 2 strategy with EB shrinkage."""
    return rank_leaks(leak_type=type, min_n=min_n, limit=limit, _tsdb_conn=conn)
