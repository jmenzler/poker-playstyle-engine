# rot-allow-file
"""Hands observation browser endpoints (D-NEW-25).

Phase 10 W3 adds:
  - ``GET /api/hands/distinct?field=<whitelisted>`` — distinct-values endpoint (D-14)
    backed by ``list_distinct()`` in src/study/hands.py.  Non-whitelisted field → HTTP 400.
    Route MUST be declared before ``/hands/{obs_id}/replay`` to avoid FastAPI matching
    "distinct" as an obs_id path param.
  - ``list_hands`` / ``get_replay`` now return hand_id, decision_id, felt_snapshot (D-07, D-08).

Available filters (mirror src/study/hands.list_hands):
    session_id, cluster_key, obs_id, has_solver_label, limit.
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from src.api.deps import get_tsdb
from src.api.hm3 import parse_hm_replay
from src.study.felt_seats import reconstruct_sim_replay
from src.study.hands import get_replay, get_replay_by_hand, list_distinct, list_hands
from tools.build_hands_index import build_hands_index

_HAND_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

router = APIRouter(prefix="/api", tags=["hands"])

TsdbDep = Annotated[Any, Depends(get_tsdb)]


@router.get("/hands")
async def list_(
    conn: TsdbDep,
    source: Annotated[str, Query(pattern="^(all|hh|sim)$")] = "all",
    solved: bool | None = None,
    corpus_solved: bool | None = None,
    stake: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[dict]:
    """Hand-level list from hands_index — all/real/sim + solved/corpus filters, paginated."""
    return list_hands(
        source=source,
        solved=solved,
        corpus_solved=corpus_solved,
        stake=stake,
        limit=limit,
        offset=offset,
        _tsdb_conn=conn,
    )


@router.post("/hands/reindex")
async def reindex(conn: TsdbDep) -> dict:
    """Rebuild hands_index. SIM rows from observations; HM rows from the decision
    corpus at HANDS_INDEX_DECISIONS_PATH (joined to HM3_PATH for played_ts) when set."""
    decisions = os.environ.get("HANDS_INDEX_DECISIONS_PATH")
    hm3 = os.environ.get("HM3_PATH")
    return await run_in_threadpool(build_hands_index, decisions_path=decisions, hm3_path=hm3, _conn=conn)


@router.get("/hands/distinct")
async def distinct(
    conn: TsdbDep,
    field: Annotated[str, Query(min_length=1, max_length=32)],
) -> list[str]:
    """Distinct values for session_id, cluster_key, or obs_id (D-14).

    Non-whitelisted field returns HTTP 400.
    """
    try:
        return list_distinct(field, _tsdb_conn=conn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/hands/by-hand/{hand_id}/replay")
async def replay_by_hand(hand_id: str, conn: TsdbDep) -> list[dict]:
    """Unified by-hand_id replay endpoint returning one step shape for sim and HM sources.

    Declared before /hands/{obs_id}/replay to prevent FastAPI from matching "by-hand"
    as an obs_id path param (declaration-order routing rule).

    Implements RPLY-01 (ordered dp0..dpN timeline) + RPLY-04 (one shape, two producers).
    """
    if not _HAND_ID_RE.match(hand_id):
        raise HTTPException(status_code=400, detail="hand_id must be alphanumeric, underscore, or hyphen")

    rows = get_replay_by_hand(hand_id, _tsdb_conn=conn)
    if rows and rows[0].get("source") != "hh":
        # SIM dps are the action timeline; reconstruct one god-view replay
        # rather than cycling per-player spots. HM hands (source='hh') fall
        # through to the full per-action HM3 reconstruction below — the per-DP
        # observation rows exist only for the solve path, not the replay.
        return reconstruct_sim_replay(rows)

    # Return raw money + big_blind; the frontend (inBB) does the single bb conversion.
    # Normalizing here too would double-divide (pot would render 1/bb too large).
    hm_events = parse_hm_replay(hand_id)
    steps = []
    for idx, evt in enumerate(hm_events):
        sf = evt.get("spot_features") or {}
        steps.append(
            {
                "step_idx": idx,
                "street": evt.get("street", ""),
                "board": list(sf.get("board", [])),
                "hero_hole": list(sf.get("hero_hole", [])),
                "pot": evt.get("pot"),
                "big_blind": evt.get("big_blind"),
                "action_sequence": list(evt.get("action_sequence", [])) if "action_sequence" in evt else [],
                "action_taken": evt.get("action_taken"),
                "seat_map": evt.get("seat_map"),
                "button_seat": evt.get("button_seat"),
                "hero_seat": evt.get("hero_seat"),
                "actor": evt.get("actor"),
                "bet_amount": evt.get("bet_amount"),
                "street_committed": evt.get("street_committed"),
                "folded": evt.get("folded"),
                "shown_cards": evt.get("shown_cards"),
                "decision_id": evt.get("decision_id"),
            }
        )
    return steps


@router.get("/hands/{obs_id}/replay")
async def replay(obs_id: str, conn: TsdbDep) -> list[dict]:
    """Per-observation replay payload. Migration 016 adds felt_snapshot for SIM rows (D-08)."""
    return get_replay(obs_id, _tsdb_conn=conn)
