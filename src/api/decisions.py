"""GET /api/decisions/sample — browse Milvus decisions collections.

Read-only listing endpoint that pages through ``preflop_decisions`` or
``postflop_decisions`` with optional scalar filters. The embedding vector is
stripped from output (large and not human-readable).

The Milvus boolean expression is assembled from the optional query params by
joining each provided filter with ` and ` per pymilvus 3.x expression syntax.
An empty expression returns the first N rows by internal order (Milvus does not
guarantee ordering without a sort key, but is stable enough for browsing).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_milvus
from src.decision_engine.blending import CANONICAL_ACTIONS

router = APIRouter(prefix="/api", tags=["decisions"])

MilvusDep = Annotated[Any, Depends(get_milvus)]

# Closed vocabularies for the scalar hard-filter fields (mirrors
# scripts/setup_milvus_collection.py field comments). Query params are
# validated against these before being interpolated into the Milvus filter
# expression — _quote escaping is defense-in-depth, not the primary guard.
_STREET_VALUES = frozenset({"preflop", "flop", "turn", "river"})
_POT_TYPE_VALUES = frozenset({"limp", "srp", "3bet", "4bet", "5bet+"})
_HERO_POS_REL_VALUES = frozenset({"OOP", "IP"})
# hero_action_type is a mixed field: HH-ingest rows store raw event verbs;
# solver/patch rows store the canonical bucketed dominant action. Allow both.
_RAW_ACTION_VERBS = frozenset({"bet", "raise", "call", "check", "fold", "allin"})
_HERO_ACTION_TYPE_VALUES = frozenset(CANONICAL_ACTIONS) | _RAW_ACTION_VERBS


def _validate(value: str, allowed: frozenset[str], field: str) -> None:
    if value not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"invalid {field}={value!r}; allowed: {sorted(allowed)}",
        )


# Scalar fields surfaced to the client. Mirrors scripts/setup_milvus_collection.py
# ``_SCALAR_FIELDS`` plus the primary key. ``embedding`` is deliberately omitted.
_OUTPUT_FIELDS = [
    "decision_id",
    "street_class",
    "street",
    "pot_type",
    "hero_pos_rel",
    "n_players_active",
    "spr_x100",
    "hero_action_type",
    "active",
    "confidence",
    "gto_score",
    "feature_spec_version",
]


def _quote(v: str) -> str:
    """Escape single quotes for a Milvus VARCHAR literal."""
    return v.replace("'", "\\'")


@router.get("/decisions/sample")
async def sample(
    milvus: MilvusDep,
    collection: Annotated[Literal["preflop_decisions", "postflop_decisions"], Query()] = "postflop_decisions",
    street: Annotated[str | None, Query()] = None,
    pot_type: Annotated[str | None, Query()] = None,
    hero_pos_rel: Annotated[str | None, Query()] = None,
    hero_action_type: Annotated[str | None, Query()] = None,
    n_players_active: Annotated[int | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=16384)] = 50,
    offset: Annotated[int, Query(ge=0, le=100000)] = 0,
) -> dict:
    """Return up to ``limit`` rows from a decisions collection, scalar fields only.

    ``offset`` is forwarded to Milvus ``query`` for server-side paging. Milvus
    enforces ``offset + limit <= 16384`` per request; deeper paging needs
    ``QueryIterator`` which is out of scope here.
    """
    clauses: list[str] = []
    if street:
        _validate(street, _STREET_VALUES, "street")
        clauses.append(f"street == '{_quote(street)}'")
    if pot_type:
        _validate(pot_type, _POT_TYPE_VALUES, "pot_type")
        clauses.append(f"pot_type == '{_quote(pot_type)}'")
    if hero_pos_rel:
        _validate(hero_pos_rel, _HERO_POS_REL_VALUES, "hero_pos_rel")
        clauses.append(f"hero_pos_rel == '{_quote(hero_pos_rel)}'")
    if hero_action_type:
        _validate(hero_action_type, _HERO_ACTION_TYPE_VALUES, "hero_action_type")
        clauses.append(f"hero_action_type == '{_quote(hero_action_type)}'")
    if n_players_active is not None:
        clauses.append(f"n_players_active == {n_players_active}")
    expr = " and ".join(clauses)

    try:
        raw = milvus.query(
            collection_name=collection,
            filter=expr,
            limit=limit,
            offset=offset,
            output_fields=_OUTPUT_FIELDS,
        )
        total_obj = milvus.get_collection_stats(collection_name=collection)
        total = int(total_obj.get("row_count", 0)) if isinstance(total_obj, dict) else 0
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Milvus query failed: {e}") from e

    rows = [{k: v for k, v in dict(row).items() if k != "embedding"} for row in raw]
    return {
        "collection": collection,
        "filter": expr,
        "count": len(rows),
        "offset": offset,
        "limit": limit,
        "total": total,
        "rows": rows,
    }
