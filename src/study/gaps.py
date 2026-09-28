"""Human-in-the-loop gap resolution service (flagged_sparse queue + manual write path)."""

from __future__ import annotations

from typing import Any

from src._errors import ValidationError
from src._log import get_logger
from src.db import timescale
from src.decision_engine.blending import CANONICAL_ACTIONS
from src.eval.solver_cache import SolverCacheEntry, persist_solve
from src.patch_engine import PatchEngine, PatchSpec
from src.solver.flop_spot import is_multiway
from src.solver.queue_driver import _normalize_for_corpus
from src.study.edit_node import _sentinel_validation, _tsdb_dsn_from_env, _validate_action_dist

log = get_logger("study.gaps")

HUMAN_GAP_GTO_SCORE: float = 0.911  # honest trust weight; oddly-specific value doubles as DB fingerprint

_POSTFLOP_CHECK_TO: frozenset[str] = frozenset(
    {"check", "bet_25", "bet_33", "bet_50", "bet_75", "bet_100", "bet_150", "bet_overbet", "allin"}
)
_POSTFLOP_FACING_BET: frozenset[str] = frozenset(
    {"fold", "call", "raise_min", "raise_2_5x", "raise_3x", "raise_pot", "allin"}
)
_PREFLOP_CHECK_TO: frozenset[str] = frozenset({"check", "open_2_2bb", "open_3bb", "allin"})
_PREFLOP_FACING_BET: frozenset[str] = frozenset({"fold", "call", "3bet_3x", "3bet_4x", "4bet_2_5x", "allin"})


def _derive_legal_actions(felt_snapshot: dict) -> list[str]:
    """Legal CANONICAL_ACTIONS subset, street-aware (preflop open/3bet/4bet vs postflop bet/raise) and facing-bet-aware."""
    facing = float(felt_snapshot.get("hero_facing_bet_bb") or 0.0)
    preflop = str(felt_snapshot.get("street") or "").lower() == "preflop"
    if preflop:
        legal = _PREFLOP_FACING_BET if facing > 0 else _PREFLOP_CHECK_TO
    else:
        legal = _POSTFLOP_FACING_BET if facing > 0 else _POSTFLOP_CHECK_TO
    return [a for a in CANONICAL_ACTIONS if a in legal]


def list_gaps(limit: int = 50, filter: str = "all", *, _tsdb_conn: Any = None) -> list[dict]:
    """Return flagged_sparse observations ordered by max_neighbor_distance DESC.

    Excludes human_resolved spots. Applies multiway/hu/all filter. One row per decision_id.
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT obs_id, decision_id, hand_id, cluster_key, "
                "       max_neighbor_distance, felt_snapshot "
                "FROM observations "
                "WHERE flagged_sparse = TRUE "
                "  AND decision_id IS NOT NULL "
                "  AND hand_id IS NOT NULL "
                "  AND NOT EXISTS ( "
                "      SELECT 1 FROM solver_cache sc "
                "      WHERE sc.decision_id = observations.decision_id "
                "        AND sc.solver_version = 'human_resolved' "
                "  ) "
                "ORDER BY max_neighbor_distance DESC NULLS LAST "
                "LIMIT %s",
                (limit,),
            )
            col = [d[0] for d in cur.description]
            rows = cur.fetchall()
        result = [dict(zip(col, r, strict=True)) for r in rows]
        for row in result:
            felt = row.get("felt_snapshot") or {}
            row["is_multiway"] = is_multiway(felt)
            row["send_to_solver_disabled"] = row["is_multiway"]
        if filter != "all":
            want_multiway = filter == "multiway"
            result = [r for r in result if r["is_multiway"] is want_multiway]
        return result
    finally:
        if own_conn:
            conn.close()


def get_gap_metadata(decision_id: str, *, _tsdb_conn: Any = None) -> dict | None:
    """Return gap metadata for a single decision_id or None if not found.

    Server-internal: embedding is not forwarded to caller (resolve_gap re-fetches).
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT cluster_key, embedding, hand_id, felt_snapshot, max_neighbor_distance "
                "FROM observations WHERE decision_id = %s LIMIT 1",
                (decision_id,),
            )
            obs = cur.fetchone()
        if obs is None:
            return None
        cluster_key, _embedding, hand_id, felt, max_neighbor_distance = obs
        felt = felt or {}
        multiway = is_multiway(felt)
        legal_actions = _derive_legal_actions(felt)
        n_players_active = int(felt.get("opponents_remaining", 1)) + 1
        street = felt.get("street")

        return {
            "decision_id": decision_id,
            "hand_id": hand_id,
            "cluster_key": cluster_key,
            "legal_actions": legal_actions,
            "is_multiway": multiway,
            "max_neighbor_distance": max_neighbor_distance,
            "n_players_active": n_players_active,
            "street": street,
            "prev_node_id": None,
            "send_to_solver_disabled": multiway,
        }
    finally:
        if own_conn:
            conn.close()


def resolve_gap(
    decision_id: str,
    action_dist: dict[str, float],
    *,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
    _patch_engine: Any = None,
) -> dict:
    """Write the hero strategy_node for a flagged_sparse gap via PatchEngine.

    Validates → fetches obs embedding → normalizes → re-fetches prev_node_id immediately
    before apply → applies patch → clears flagged_sparse → writes human_resolved dedup.
    """
    log.info("study.gaps.resolve_gap.started", decision_id=decision_id)
    _validate_action_dist(action_dist)

    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT cluster_key, embedding, hand_id, felt_snapshot "
                "FROM observations WHERE decision_id = %s LIMIT 1",
                (decision_id,),
            )
            obs = cur.fetchone()
        if obs is None:
            raise ValidationError(f"no observation for decision_id={decision_id!r}")
        cluster_key, embedding, _hand_id, _felt_snapshot = obs

        collection = "preflop_decisions" if "street_class=preflop" in cluster_key else "postflop_decisions"
        raw_emb = list(embedding) if not isinstance(embedding, list) else embedding
        norm_emb = _normalize_for_corpus(raw_emb, collection)
        skip_milvus = norm_emb is None
        if skip_milvus:
            log.warning(
                "study.gaps.resolve_gap.embedding_norm_failed",
                decision_id=decision_id,
                cluster_key=cluster_key,
            )

        # On skip_milvus the raw vector must never reach the COSINE corpus, so the
        # spec carries an empty embedding (TSDB node persists; kNN is Milvus-only).
        spec = PatchSpec(
            cluster_key=cluster_key,
            decision_id=decision_id,
            action_dist=action_dist,
            embedding=norm_emb if norm_emb is not None else [],
            gto_score=HUMAN_GAP_GTO_SCORE,
            confidence=1.0,
            prev_node_id=None,
            source="manual",
        )

        engine = _patch_engine if _patch_engine is not None else PatchEngine()
        record = engine.apply(
            spec,
            validation=_sentinel_validation(),
            pre_ev_loss=None,
            post_ev_loss=None,
            skip_milvus=skip_milvus,
            _tsdb_conn=conn,
            _milvus=_milvus,
        )

        # Clear the flag and write the human_resolved dedup marker in one
        # transaction so list_gaps never sees the gap cleared without a marker
        # (or vice versa) after a partial failure. persist_solve's inner
        # transaction nests as a savepoint under this outer block.
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE observations SET flagged_sparse = FALSE WHERE decision_id = %s",
                    (decision_id,),
                )
            persist_solve(
                SolverCacheEntry(
                    cluster_key=cluster_key,
                    decision_id=decision_id,
                    action_dist={},
                    exploitability_pct=0.0,
                    solver_version="human_resolved",
                    spot_features=None,
                ),
                _tsdb_conn=conn,
            )

        log.info(
            "study.gaps.resolve_gap.complete",
            decision_id=decision_id,
            patch_id=str(record.patch_id),
        )
        return {"patch_id": str(record.patch_id), "new_node_id": str(record.new_node_id)}
    finally:
        if own_conn:
            conn.close()


def send_to_solver(decision_id: str, *, _tsdb_conn: Any = None) -> dict:
    """Re-enqueue a human-resolved gap into the solver priority queue.

    Raises ValidationError for multiway spots (solver is HU-only).
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT felt_snapshot FROM observations WHERE decision_id = %s LIMIT 1",
                (decision_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise ValidationError(f"no observation for decision_id={decision_id!r}")
        felt = row[0] or {}
        if is_multiway(felt):
            raise ValidationError("solver is HU-only; multiway gaps cannot be sent to solver")
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "DELETE FROM solver_cache WHERE decision_id = %s AND solver_version = 'human_resolved'",
                (decision_id,),
            )
        return {"decision_id": decision_id, "enqueued": True}
    finally:
        if own_conn:
            conn.close()


def get_edit_history(decision_id: str, *, _tsdb_conn: Any = None) -> list[dict]:
    """Reverse-chronological manual patches for a decision_id (the DP).

    Each entry {ts, action_dist, status}; status from patches.status since under
    coexist every strategy_node stays active=TRUE.
    """
    own_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT p.ts, sn.action_dist, COALESCE(p.status, 'applied') AS status "
                "FROM patches p "
                "JOIN strategy_nodes sn ON sn.node_id = p.new_node_id "
                "WHERE p.decision_id = %s AND p.source = 'manual' "
                "ORDER BY p.ts DESC",
                (decision_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "ts": row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]),
                "action_dist": row[1],
                "status": row[2],
            }
            for row in rows
        ]
    finally:
        if own_conn:
            conn.close()
