"""CLI-03 / FastAPI POST /api/edit-node — manual action_dist edit via PatchEngine.

D-11 wire-up: backend receives a parsed action_dist dict (CLI parses --set or --editor).
This module:
1. Validates frequencies (sum 1.0 ± 0.001, keys in CANONICAL_ACTIONS, no negatives).
2. Resolves prev_node_id + current embedding from strategy_nodes.
3. Constructs PatchSpec(source='manual').
4. Calls PatchEngine.apply() with a sentinel ValidationResult (Option A, see 06-PATTERNS).

Lock-from-autoloop (D-NEW-26): separate `set_lock_from_autoloop()` — does NOT touch
action_dist, so it bypasses PatchEngine. Direct UPDATE is acceptable per Phase 6
PATTERNS since the column was added by migration 011 specifically for this purpose.
"""

from __future__ import annotations

from typing import Any

from src._errors import ValidationError
from src._log import get_logger
from src.db import timescale
from src.decision_engine.blending import CANONICAL_ACTIONS
from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

log = get_logger("study.edit_node")

_SUM_TOLERANCE: float = 0.001
_CANONICAL_SET: frozenset[str] = frozenset(CANONICAL_ACTIONS)


def _sentinel_validation() -> ValidationResult:
    """Empty ValidationResult sentinel for manual + solver patches (Option A).

    Manual edits bypass A/B (user is authority); solver verifications bypass A/B
    (solver IS the ground truth being cached). The sentinel keeps the Phase 5
    PatchEngine.apply() signature unchanged while making the bypass explicit in
    the patches.validation JSONB column.
    """
    return ValidationResult(
        seed=0,
        ev_loss_delta=0.0,
        n_hands=0,
        confidence_interval=(0.0, 0.0),
        n_cluster_hits=0,
        zero_hits=True,
    )


def _validate_action_dist(action_dist: dict[str, float]) -> None:
    """Raise ValidationError on any of: unknown keys, negative freqs, sum != 1.0 ± 0.001."""
    bad_keys = set(action_dist) - _CANONICAL_SET
    if bad_keys:
        raise ValidationError(
            f"unknown action keys: {sorted(bad_keys)} (must be in CANONICAL_ACTIONS: {CANONICAL_ACTIONS})"
        )
    for k, v in action_dist.items():
        if v < 0:
            raise ValidationError(f"negative frequency for {k!r}: {v}")
    total = sum(action_dist.values())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValidationError(f"action_dist sum={total:.4f} (must equal 1.000 ± {_SUM_TOLERANCE})")


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables (mirrors patch_engine)."""
    import os

    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def edit_node(
    cluster_key: str,
    action_dist: dict[str, float],
    *,
    reason: str = "",
    _tsdb_conn: Any = None,
    _milvus: Any = None,
    _patch_engine: Any = None,
) -> dict:
    """Manual edit-node entry. Validates → resolves prev node → applies via PatchEngine.

    Args:
        cluster_key: target cluster's canonical key string.
        action_dist: candidate distribution (canonical 15-action vocab).
        reason: free-form operator note; surfaced in the returned dict.
        _tsdb_conn: injected for tests; production opens its own connection.
        _milvus: injected for tests; forwarded to PatchEngine.apply.
        _patch_engine: injected for tests; production constructs PatchEngine().

    Returns:
        JSON-serializable dict: {patch_id, new_node_id, source, reason}.

    Raises:
        ValidationError: on any of bad keys, negative freqs, sum check, or
            missing active strategy_nodes row.
    """
    log.info("study.edit_node.started", cluster_key=cluster_key, n_keys=len(action_dist))
    _validate_action_dist(action_dist)

    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        # Resolve the current active row for this cluster_key
        with conn.cursor() as cur:
            cur.execute(
                "SELECT node_id, embedding, gto_score, confidence FROM strategy_nodes "
                "WHERE cluster_key = %s AND active = TRUE LIMIT 1",
                (cluster_key,),
            )
            row = cur.fetchone()
        if row is None:
            raise ValidationError(
                f"no active strategy_node for cluster_key={cluster_key!r} — "
                f"cannot edit a cluster the engine has never seen"
            )
        _node_id, embedding, gto_score, confidence = row

        with conn.cursor() as cur:
            cur.execute(
                "SELECT decision_id FROM observations "
                "WHERE cluster_key = %s AND decision_id IS NOT NULL LIMIT 1",
                (cluster_key,),
            )
            dp_row = cur.fetchone()
        if dp_row is None:
            raise ValidationError(
                f"no observation decision_id to tag the edit for cluster_key={cluster_key!r}"
            )

        spec = PatchSpec(
            cluster_key=cluster_key,
            decision_id=dp_row[0],
            action_dist=action_dist,
            embedding=list(embedding) if not isinstance(embedding, list) else embedding,
            gto_score=float(gto_score) if gto_score is not None else 0.5,
            confidence=float(confidence) if confidence is not None else 0.5,
            prev_node_id=None,
            source="manual",
        )

        engine = _patch_engine if _patch_engine is not None else PatchEngine()
        record = engine.apply(
            spec,
            validation=_sentinel_validation(),
            pre_ev_loss=None,
            post_ev_loss=None,
            _tsdb_conn=conn,
            _milvus=_milvus,
        )
        log.info(
            "study.edit_node.complete",
            cluster_key=cluster_key,
            patch_id=str(record.patch_id),
        )
        return {
            "patch_id": str(record.patch_id),
            "new_node_id": str(record.new_node_id),
            "source": "manual",
            "reason": reason,
        }
    finally:
        if own_conn:
            conn.close()


def set_lock_from_autoloop(
    cluster_key: str,
    locked: bool,
    *,
    _tsdb_conn: Any = None,
) -> int:
    """Set strategy_nodes.locked_from_autoloop for the active row of a cluster.

    Returns the number of rows updated (0 if cluster_key has no active row, else 1).
    Does NOT go through PatchEngine: the column does not affect action_dist and
    therefore does not need the audit trail PatchEngine provides (documented
    PatchEngine bypass per 06-PATTERNS and threat T-06-20).
    """
    log.info("study.edit_node.set_lock", cluster_key=cluster_key, locked=locked)
    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "UPDATE strategy_nodes SET locked_from_autoloop = %s "
                "WHERE cluster_key = %s AND active = TRUE",
                (locked, cluster_key),
            )
            return cur.rowcount
    finally:
        if own_conn:
            conn.close()
