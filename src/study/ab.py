"""CLI-05 / FastAPI POST /api/ab — sim A/B validation for a candidate patch.

Wraps src/validation/sim_ab.py::validate() using its exact Phase 5 signature:

    validate(patch_spec, config, *, baseline_engine, sim_adapter, seed,
             _tsdb_conn=None, _milvus=None) -> ValidationResult

`run_ab(cluster_key, patch_id)`:
1. SELECT the patches row JOINed to strategy_nodes for action_dist + embedding.
2. Construct a PatchSpec from that row (the candidate the validator wraps internally
   via StrategyOverride).
3. Construct AutoLoopConfig with n_hands_validation = n_hands.
4. Construct baseline_engine = KNNDecisionEngine() (the current production engine).
5. Construct sim_adapter = SimAdapter().
6. Call validate(...) and unwrap ValidationResult → dict.
"""

from __future__ import annotations

from typing import Any

from src._log import get_logger
from src.db import timescale
from src.patch_engine import PatchSpec
from src.validation.sim_ab import validate as _real_validate

log = get_logger("study.ab")


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables (mirrors patch_engine)."""
    import os

    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


def run_ab(
    cluster_key: str,
    patch_id: str,
    *,
    n_hands: int = 10000,
    seed: int = 42,
    _tsdb_conn: Any = None,
    _validate: Any = None,
    _baseline_engine: Any = None,
    _sim_adapter: Any = None,
    _milvus: Any = None,
) -> dict:
    """Run sim A/B for a candidate patch. Returns delta + CI + n_hands.

    Args:
        cluster_key: target cluster's canonical key string. Must equal the
            cluster_key stored on the patches row (otherwise ValueError).
        patch_id: UUID-as-string of the candidate patches row.
        n_hands: hands to drive in each of the paired A/B runs (default 10000).
        seed: paired seed for VALN-01 determinism (default 42).
        _tsdb_conn / _validate / _baseline_engine / _sim_adapter / _milvus:
            test-injection hooks. Production callers leave them None.

    Returns:
        JSON-serializable dict:
            {cluster_key, patch_id, ev_loss_delta, ci_low, ci_high, n_hands,
             seed, n_cluster_hits, zero_hits}.

    Raises:
        ValueError: if patch_id is not found, or its cluster_key does not match
            the cluster_key argument (sanity check guarding API misuse).
    """
    log.info(
        "study.ab.started",
        cluster_key=cluster_key,
        patch_id=patch_id,
        n_hands=n_hands,
    )
    own_conn = False
    conn = _tsdb_conn
    if conn is None:
        conn = timescale.connect(_tsdb_dsn_from_env())
        own_conn = True
    try:
        # 1. Read the patch row + the patched strategy_nodes row (new_node_id) for its
        #    action_dist + embedding — this becomes the candidate PatchSpec.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT p.patch_id, p.cluster_key, p.source, p.prev_node_id, p.new_node_id, "
                "       sn.action_dist, sn.embedding, sn.gto_score, sn.confidence, p.decision_id "
                "FROM patches p "
                "JOIN strategy_nodes sn ON sn.node_id = p.new_node_id "
                "WHERE p.patch_id = %s",
                (patch_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise ValueError(f"patch_id={patch_id!r} not found (or strategy_nodes row missing)")
        (
            _pid,
            ck,
            _source,
            _prev_node_id,
            _new_node_id,
            action_dist,
            embedding,
            gto_score,
            confidence,
            decision_id,
        ) = row

        # Sanity: cluster_key argument must match the patches row
        if ck != cluster_key:
            raise ValueError(f"cluster_key mismatch: arg={cluster_key!r} vs patches.cluster_key={ck!r}")

        # A patch without a decision_id (pre-migration row) cannot be re-keyed for the
        # coexist model — fabricating one would orphan a Milvus row. Fail loud.
        if decision_id is None:
            raise ValueError(
                f"patch {patch_id} has no decision_id (pre-migration row); "
                "re-create it via resolve_gap to get a properly-keyed patch before A/B"
            )

        patch_spec = PatchSpec(
            cluster_key=cluster_key,
            decision_id=decision_id,
            action_dist=action_dist,
            embedding=list(embedding) if not isinstance(embedding, list) else embedding,
            gto_score=float(gto_score) if gto_score is not None else 0.5,
            confidence=float(confidence) if confidence is not None else 0.5,
            prev_node_id=None,
            source="manual",
        )

        # 2. Build minimal AutoLoopConfig with the fields validate() reads.
        from src._config import AutoLoopConfig

        config = AutoLoopConfig(
            n_hands_validation=n_hands,
            bootstrap_resamples=1000,
            ci_alpha=0.05,
        )

        # 3. Resolve baseline engine + sim adapter (lazy import to keep tests fast)
        if _baseline_engine is None:
            from src.decision_engine.engine import KNNDecisionEngine

            baseline_engine = KNNDecisionEngine()
        else:
            baseline_engine = _baseline_engine

        if _sim_adapter is None:
            from src.sim.adapter import SimAdapter

            sim_adapter = SimAdapter()
        else:
            sim_adapter = _sim_adapter

        # 4. Call validate() with its EXACT Phase 5 signature.
        validate_fn = _validate if _validate is not None else _real_validate
        result = validate_fn(
            patch_spec,
            config,
            baseline_engine=baseline_engine,
            sim_adapter=sim_adapter,
            seed=seed,
            _tsdb_conn=conn,
            _milvus=_milvus,
        )

        return {
            "cluster_key": cluster_key,
            "patch_id": patch_id,
            "ev_loss_delta": float(result.ev_loss_delta),
            "ci_low": float(result.confidence_interval[0]),
            "ci_high": float(result.confidence_interval[1]),
            "n_hands": int(result.n_hands),
            "seed": int(result.seed),
            "n_cluster_hits": int(result.n_cluster_hits),
            "zero_hits": bool(result.zero_hits),
        }
    finally:
        if own_conn:
            conn.close()
