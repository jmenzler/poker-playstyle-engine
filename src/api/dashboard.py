"""GET /api/dashboard — D-15 / D-NEW-27 verdict-led snapshot.

Wraps src/study/dashboard.dashboard_snapshot. Returns the 6-section dict
consumed by both the CLI text renderer (CLI-06) and the React Dashboard panel.

This route also overrides the `health` section of the snapshot with
live subsystem probes (milvus list_collections, POSTFLOP_CLI_BIN
existence, fastapi self-ping). The `_health()` helper in
src/study/dashboard.py is deliberately conservative (db only) — the
non-db probes are wired here so the CLI path stays dependency-free.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from src.api.deps import get_milvus, get_tsdb
from src.study.dashboard import dashboard_snapshot

router = APIRouter(prefix="/api", tags=["dashboard"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]
MilvusDep = Annotated[Any, Depends(get_milvus)]


@router.get("/dashboard")
async def snapshot(conn: TsdbDep, milvus: MilvusDep) -> dict:
    """6-section dashboard snapshot — loop_health, ev_loss_trend, recent_activity,
    health, session, kb_growth (D-NEW-27).
    """
    snap = dashboard_snapshot(_tsdb_conn=conn)
    snap["health"] = {**snap.get("health", {}), **_probe_subsystems(milvus)}
    return snap


def _probe_subsystems(milvus: Any) -> dict[str, str]:
    """Live milvus + solver + fastapi liveness."""
    out: dict[str, str] = {"fastapi": "healthy"}

    try:
        milvus.list_collections()
        out["milvus"] = "healthy"
    except Exception:
        out["milvus"] = "down"

    solver_bin = Path(
        os.environ.get("POSTFLOP_CLI_BIN", "~/postflop-cli/target/release/postflop-cli")
    ).expanduser()
    if not solver_bin.is_file():
        out["solver"] = "unavailable"
    elif _solver_running():
        out["solver"] = "running"
    else:
        out["solver"] = "healthy"

    return out


def _solver_running() -> bool:
    """True when a verify or eval job (both drive the solver) is in flight."""
    from src.api.jobs import registry

    return any(registry.in_flight_count(kind) > 0 for kind in ("verify", "eval"))
