"""src/patch_engine.py — TSDB-first single write path for strategy_nodes.  long-ok

This module is the ONLY allowlisted writer for ``strategy_nodes`` rows.  Every
``INSERT INTO strategy_nodes`` or ``UPDATE strategy_nodes`` from any other module
is a contract violation, caught at three layers:

1. **Lint time** — ``.pre-commit-config.yaml`` §no-direct-strategy-nodes-write
   rejects any direct SQL against ``strategy_nodes`` outside this file.
2. **Schema level** — ``migrations/010_autoloop_source.sql`` adds ``'autoloop'``
   to the ``strategy_nodes.source`` CHECK constraint; out-of-vocabulary values
   raise ``CheckViolation`` before a row can commit.
3. **Runtime enforcement** — ``tests/integration/test_no_direct_strategy_nodes_write.py``
   attempts a direct INSERT and asserts the schema guard fires (PTCH-01).

Transaction sequence:
  1. Open TSDB transaction (``conn.transaction()``).
  2. INSERT INTO strategy_nodes the new node with active=TRUE. No cluster-
     supersede: patches coexist (keyed by decision_id) and the kNN blend resolves.
  3. INSERT INTO patches the audit row with status='applied' and validation JSONB.
  4. COMMIT TSDB transaction.
  5. Milvus upsert (best-effort) keyed by the source ``decision_id``, AFTER TSDB
     commit. If Milvus fails, log a warning and continue — TSDB is source of truth.

Rollback deletes the patch's Milvus row by ``decision_id``, sets its strategy_node
inactive, and marks the patches row status='rolled_back'.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import msgspec

from src._log import get_logger
from src.db import milvus as milvus_db
from src.db import timescale
from src.metrics.ev_loss import _street_class_from_cluster_key as _street_class

log = get_logger("patch_engine")

FEATURE_SPEC_VERSION = 2


# ---------------------------------------------------------------------------
# Public msgspec.Struct types
# ---------------------------------------------------------------------------


class ValidationResult(msgspec.Struct, frozen=True, kw_only=True):
    """A/B validation outcome attached to every patch (VALN-02).

    All fields are required except ``zero_hits`` (default False).
    ``confidence_interval`` is stored as a tuple at runtime; ``msgspec.to_builtins``
    converts it to a JSON-serializable list.
    """

    seed: int
    ev_loss_delta: float  # mean(baseline_evloss - candidate_evloss)
    n_hands: int  # total hands in the A/B run
    confidence_interval: tuple[float, float]  # (ci_low, ci_high)
    n_cluster_hits: int  # hands that hit the target cluster_key
    zero_hits: bool = False  # True if cluster never appeared in A/B


class PatchSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Prepared patch payload (no patch_id; assigned by PatchEngine.apply).

    Defaults follow 05-RESEARCH.md Pitfall 6: gto_score=0.5, confidence=0.5.
    ``source='autoloop'`` satisfies LOOP-06; 'manual' is valid for CLI-driven
    patches.

    Note on value ranges: msgspec validates types (float) but not value ranges.
    Callers MUST pass gto_score and confidence in [0, 1]; the engine does not
    enforce this constraint at runtime.
    """

    cluster_key: str
    decision_id: str  # source DP id; Milvus row PK + replay/rollback handle
    action_dist: dict[str, float]  # candidate distribution (kNN-rebalance output)
    embedding: list[float]  # representative observation's embedding
    gto_score: float = 0.5  # autoloop default per 05-RESEARCH.md Pitfall 6
    confidence: float = 0.5  # autoloop default per 05-RESEARCH.md Pitfall 6
    prev_node_id: uuid.UUID | None = None  # vestigial under coexist model; always None
    source: str = "autoloop"  # LOOP-06; 'manual' valid for CLI-driven patches
    street: str = ""  # postflop street (flop/turn/river); "" = unknown
    spr_x100: int = -1  # stack-to-pot ratio x100; -1 = sentinel (no spr filter)


class PatchRecord(msgspec.Struct, frozen=True, kw_only=True):
    """Persistent record of an apply or rollback."""

    patch_id: uuid.UUID
    ts: str  # ISO-8601 UTC timestamp string
    status: str  # 'applied' | 'rolled_back'
    cluster_key: str
    new_node_id: uuid.UUID
    prev_node_id: uuid.UUID | None


# ---------------------------------------------------------------------------
# PatchEngine
# ---------------------------------------------------------------------------


class PatchEngine:
    """Transaction coordinator for strategy_nodes writes.

    Stateless — no connection state is held; connections are opened per-call
    (or injected via ``_tsdb_conn`` / ``_milvus`` kwargs for unit tests).
    """

    def __init__(self) -> None:
        pass

    def apply(
        self,
        patch: PatchSpec,
        *,
        validation: ValidationResult,
        pre_ev_loss: float | None = None,
        post_ev_loss: float | None = None,
        skip_milvus: bool = False,
        _tsdb_conn: Any = None,
        _milvus: Any = None,
    ) -> PatchRecord:
        """Apply a patch (TSDB-first, Milvus post-commit). Returns a PatchRecord.

        ``skip_milvus=True`` bypasses the corpus upsert (no env client opened) — for
        a gap whose embedding could not be z-scored into corpus space.
        """
        conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
        client = None if skip_milvus else (_milvus if _milvus is not None else milvus_db.connect_from_env())

        # Step 1: TSDB transaction — INSERT node + patches audit row. No cluster-
        # supersede; patches coexist (keyed by decision_id), the kNN blend resolves.
        new_node_id = uuid.uuid4()
        patch_id = uuid.uuid4()
        ts_str = datetime.now(tz=UTC).isoformat()
        validation_json = json.dumps(_validation_to_dict(validation))

        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO strategy_nodes"
                " (node_id, cluster_key, embedding, action_dist, gto_score,"
                "  confidence, source, active, created_at)"
                " VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, TRUE, now())",
                (
                    new_node_id,
                    patch.cluster_key,
                    patch.embedding,
                    json.dumps(patch.action_dist),
                    patch.gto_score,
                    patch.confidence,
                    patch.source,
                ),
            )
            cur.execute(
                "INSERT INTO patches"
                " (patch_id, ts, source, cluster_key, prev_node_id, new_node_id,"
                "  pre_ev_loss, post_ev_loss, status, validation, notes, decision_id)"
                " VALUES (%s, now(), %s, %s, %s, %s, %s, %s, 'applied', %s::jsonb, NULL, %s)",
                (
                    patch_id,
                    patch.source,
                    patch.cluster_key,
                    None,
                    new_node_id,
                    pre_ev_loss,
                    post_ev_loss,
                    validation_json,
                    patch.decision_id,
                ),
            )

        # Step 2: Milvus upsert AFTER TSDB commit (TSDB is source of truth). PK is
        # the source decision_id so the patch coexists; re-patching overwrites it.
        if not skip_milvus:
            assert client is not None
            collection = (
                "preflop_decisions"
                if _street_class(patch.cluster_key).lower() == "preflop"
                else "postflop_decisions"
            )
            dominant_action = max(patch.action_dist, key=patch.action_dist.__getitem__)
            hard_filter = _parse_cluster_key(patch.cluster_key)
            node_row = {
                "decision_id": patch.decision_id,
                "embedding": patch.embedding,
                "street_class": hard_filter.get("street_class", ""),
                "street": patch.street,
                "pot_type": hard_filter.get("pot_type", ""),
                "hero_pos_rel": hard_filter.get("hero_pos_rel", ""),
                "n_players_active": int(hard_filter.get("n_players_active", "0")),
                "spr_x100": patch.spr_x100,
                # Required by the collection schema (not nullable). A patch's action_dist
                # keys are already canonical, so the raw-snap inputs use -1 sentinels.
                "hero_action_size_pot_frac": -1.0,
                "raise_ratio": -1.0,
                "hero_action_allin": False,
                "hero_action_type": dominant_action,
                "active": True,
                "confidence": patch.confidence,
                "gto_score": patch.gto_score,
                "feature_spec_version": FEATURE_SPEC_VERSION,
                "action_dist": json.dumps(patch.action_dist),
                "added_at": int(datetime.now(tz=UTC).timestamp()),
                "removed_at": 0,
            }
            try:
                client.upsert(collection_name=collection, data=[node_row])
            except Exception as exc:
                log.warning(
                    "patch_engine.milvus_upsert_failed",
                    patch_id=str(patch_id),
                    cluster_key=patch.cluster_key,
                    error=str(exc),
                )
                # Intentionally swallow: TSDB is source of truth (05-CONTEXT.md Decision 2).

        log.info(
            "patch_engine.apply.complete",
            cluster_key=patch.cluster_key,
            patch_id=str(patch_id),
            new_node_id=str(new_node_id),
            prev_node_id=str(patch.prev_node_id) if patch.prev_node_id else None,
        )

        return PatchRecord(
            patch_id=patch_id,
            ts=ts_str,
            status="applied",
            cluster_key=patch.cluster_key,
            new_node_id=new_node_id,
            prev_node_id=patch.prev_node_id,
        )

    def rollback(
        self,
        patch_id: uuid.UUID,
        *,
        _tsdb_conn: Any = None,
        _milvus: Any = None,
    ) -> PatchRecord:
        """Roll back a patch: set its strategy_node inactive, mark the patches row
        status='rolled_back' in place, and soft-delete its Milvus row: keep the row,
        set active=False + removed_at so as-of replays still see the pre-rollback state.

        Raises LookupError if patch_id is unknown, RuntimeError if already rolled back.
        """
        conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
        client = _milvus if _milvus is not None else milvus_db.connect_from_env()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT prev_node_id, new_node_id, cluster_key, status, source, decision_id"
                " FROM patches WHERE patch_id = %s",
                (patch_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"patch_id {patch_id} not found in patches table")
        orig_prev_node_id, orig_new_node_id, cluster_key, status, _orig_source, orig_decision_id = row
        if status == "rolled_back":
            raise RuntimeError(f"patch {patch_id} already rolled back")

        ts_str = datetime.now(tz=UTC).isoformat()
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "UPDATE strategy_nodes SET active = FALSE WHERE node_id = %s",
                (orig_new_node_id,),
            )
            cur.execute(
                "UPDATE patches SET status = 'rolled_back', notes = %s WHERE patch_id = %s",
                (f"rolled_back_at={ts_str}", patch_id),
            )

        # Soft-delete the patch's coexisting Milvus row by its source decision_id. upsert
        # replaces the whole entity, so re-fetch the full row before mutating active/removed_at.
        # Legacy patches (pre-migration, decision_id NULL) have no locatable row to soft-delete.
        if orig_decision_id is not None:
            collection = (
                "preflop_decisions"
                if _street_class(cluster_key).lower() == "preflop"
                else "postflop_decisions"
            )
            try:
                rows = client.query(
                    collection_name=collection,
                    filter=f'decision_id == "{orig_decision_id}"',
                    output_fields=["*"],
                    consistency_level="Strong",
                )
                if rows:
                    entity = dict(rows[0])
                    entity["active"] = False
                    entity["removed_at"] = int(datetime.now(tz=UTC).timestamp())
                    client.upsert(collection_name=collection, data=[entity])
            except Exception as exc:
                log.warning(
                    "patch_engine.rollback.milvus_softdelete_failed",
                    patch_id=str(patch_id),
                    decision_id=orig_decision_id,
                    cluster_key=cluster_key,
                    error=str(exc),
                )
                # Intentionally swallow: TSDB is source of truth.

        log.info(
            "patch_engine.rollback.complete",
            patch_id=str(patch_id),
            cluster_key=cluster_key,
            orig_new_node_id=str(orig_new_node_id),
            decision_id=orig_decision_id,
        )

        return PatchRecord(
            patch_id=patch_id,
            ts=ts_str,
            status="rolled_back",
            cluster_key=cluster_key,
            new_node_id=orig_new_node_id,
            prev_node_id=orig_prev_node_id,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables.

    TSDB_PASSWORD is required; all other vars have sensible defaults.
    Copied verbatim from src/metrics/ev_loss.py to keep patch_engine self-contained.
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def _parse_cluster_key(cluster_key: str) -> dict[str, str]:
    """Parse a cluster_key string into a dict of hard_filter components.

    Inverse of ``_cluster_key_from_filter`` in ``src/sim/harness.py``.

    Example:
        "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
        -> {"hero_pos_rel": "BTN", "n_players_active": "2",
            "pot_type": "srp", "street_class": "postflop"}
    """
    result: dict[str, str] = {}
    for token in cluster_key.split("|"):
        if "=" in token:
            k, v = token.split("=", 1)
            result[k] = v
    return result


def _validation_to_dict(v: ValidationResult) -> dict[str, Any]:
    """Convert a ValidationResult to a plain dict suitable for JSON serialization.

    Uses ``msgspec.to_builtins`` which converts ``confidence_interval`` tuple
    to a JSON-serializable list.
    """
    result = msgspec.to_builtins(v)
    return result  # type: ignore[no-any-return]
