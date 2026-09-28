"""src/autoloop/driver.py — AutoLoopDriver: orchestrates the auto-improvement loop.  long-ok

Implements the AutoLoop protocol (src/protocols/auto_loop.py: step() -> int).

LOOP-01: AutoLoopDriver satisfies the AutoLoop runtime-checkable Protocol.
LOOP-02: Candidates processed SEQUENTIALLY in a single for loop — no threading, no asyncio.
LOOP-03: PatchEngine.apply called only when ci_low > 0 (acceptance gate after validate()).
LOOP-04: All parameters read from config/autoloop.toml via load_toml_config at start of step().
LOOP-05: If config.enabled is False, step() returns 0 IMMEDIATELY without opening any DB connection.
LOOP-06: All patches applied by the driver carry source='autoloop' in PatchSpec.

Error handling philosophy: catch-and-continue. One bad candidate never kills the cycle.
Each candidate is wrapped in try/except; errors are logged and processing moves to the next.

Calling order per cycle:
  1. Load config (LOOP-04). If enabled=False, return 0 (LOOP-05).
  2. Open DB connections (TSDB + Milvus).
  3. Resolve session_id of last RECORD run.
  4. select_patch_candidates(session_id, config) -> list[ClusterCandidate].
  5. For each candidate (sequential, LOOP-02):
       a. generate_candidate_action_dist(cluster_key) -> (action_dist, embedding).
       b. Compute pre_ev_loss via ev_loss() on the current snapshot.
       c. Build PatchSpec with source='autoloop' (LOOP-06).
       d. validate(patch_spec, config, ...) -> ValidationResult.
       e. If zero_hits OR ci_low <= 0: log discard reason; CONTINUE. (LOOP-03)
       f. PatchEngine.apply(patch_spec, ...). Errors bubble to the per-candidate
          catch-and-continue handler in step().
       g. Increment applied counter.
  6. Return applied count.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from src._config import AutoLoopConfig, load_toml_config
from src._errors import NoStrategyError
from src._log import get_logger
from src.autoloop.leak_detector import ClusterCandidate, select_patch_candidates
from src.autoloop.rebalance import generate_candidate_action_dist
from src.db import milvus as milvus_db
from src.db import timescale
from src.metrics.ev_loss import ev_loss
from src.patch_engine import PatchEngine, PatchSpec

# AutoLoop protocol is imported here for type-checking and instanceof tests.
# The import itself makes AutoLoopDriver a structural subtype at runtime.
from src.protocols.auto_loop import AutoLoop  # noqa: F401 (used by isinstance checks in tests)
from src.study import corpus_versions
from src.validation.sim_ab import validate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/autoloop.toml")

# Deterministic A/B seed. Override per-driver-instance for testing.
DEFAULT_VALIDATION_SEED = 0xFEEDBEEF


# ---------------------------------------------------------------------------
# AutoLoopDriver
# ---------------------------------------------------------------------------


class AutoLoopDriver:
    """Implements AutoLoop protocol. step() runs ONE full cycle.

    The driver is stateless with respect to DB connections — connections are
    opened fresh at each step() call and closed in the finally block.

    Constructor args:
        config_path: path to autoloop.toml (LOOP-04).
        validation_seed: RNG seed for paired-seed A/B (VALN-01).
        baseline_engine: KNNDecisionEngine; lazy-built on first step() with candidates
            if not provided. Inject for testing.
        sim_adapter: SimAdapter (RLCard wrapper); lazy-built similarly. Inject for testing.
    """

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        *,
        validation_seed: int = DEFAULT_VALIDATION_SEED,
        baseline_engine: Any = None,
        sim_adapter: Any = None,
    ) -> None:
        self._config_path = config_path
        self._validation_seed = validation_seed
        self._baseline_engine = baseline_engine
        self._sim_adapter = sim_adapter
        self._log = get_logger("autoloop.driver")

    def step(self) -> int:
        """Run one auto-loop cycle. Returns count of patches applied.

        LOOP-04: config loaded at entry.
        LOOP-05: enabled=False short-circuits BEFORE any DB connection.
        LOOP-02: candidates processed sequentially.
        LOOP-03: patch accepted only if ci_low > 0.
        LOOP-06: all patches carry source='autoloop'.
        """
        # --- 1. LOAD CONFIG FIRST — no DB I/O above this line (LOOP-04) ---
        config = load_toml_config(self._config_path, AutoLoopConfig)

        # --- 2. LOOP-05: enabled=False short-circuits BEFORE any DB connection ---
        if not config.enabled:
            self._log.info("autoloop.driver.step.disabled")
            return 0

        # --- 3. DB I/O is allowed from here on ---
        tsdb_conn = timescale.connect(_tsdb_dsn_from_env())
        milvus_client = milvus_db.connect_from_env()

        try:
            # --- 4. Resolve most recent session ---
            session_id = self._last_record_session_id(tsdb_conn)
            if session_id is None:
                self._log.warning("autoloop.driver.no_recent_session")
                return 0

            # --- 5. Detect leak candidates ---
            candidates = select_patch_candidates(
                session_id,
                config,
                _tsdb_conn=tsdb_conn,
            )
            self._log.info(
                "autoloop.driver.candidates",
                session_id=session_id,
                n_candidates=len(candidates),
            )

            # --- Lazy-init baseline_engine + sim_adapter only if needed ---
            if candidates and (self._baseline_engine is None or self._sim_adapter is None):
                self._baseline_engine, self._sim_adapter = self._build_engine_and_adapter()

            # --- 6. LOOP-02: sequential — never concurrent ---
            applied = 0
            patch_engine = PatchEngine()

            for candidate in candidates:
                try:
                    applied += self._process_candidate(
                        candidate,
                        config,
                        patch_engine,
                        tsdb_conn=tsdb_conn,
                        milvus_client=milvus_client,
                    )
                except Exception as exc:
                    self._log.error(
                        "autoloop.driver.candidate_failed",
                        cluster_key=candidate.cluster_key,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    continue

            self._log.info(
                "autoloop.driver.step_complete",
                session_id=session_id,
                applied=applied,
                candidates_attempted=len(candidates),
            )
            # Only snapshot when work landed — empty versions are meaningless.
            if applied > 0:
                corpus_versions.snapshot(
                    label=f"autoloop-{session_id}",
                    source="autoloop",
                    _tsdb_conn=tsdb_conn,
                )
            return applied

        finally:
            with contextlib.suppress(Exception):
                tsdb_conn.close()
            # MilvusClient in pymilvus 3.0 has no explicit close in single-host mode.

    def _process_candidate(
        self,
        candidate: ClusterCandidate,
        config: AutoLoopConfig,
        patch_engine: PatchEngine,
        *,
        tsdb_conn: Any,
        milvus_client: Any,
    ) -> int:
        """Process ONE candidate; return 1 if applied, 0 if discarded.

        Raises exceptions for the outer catch-and-continue loop to handle.
        Internal errors specific to rebalance/validation are handled here
        and result in returning 0 (discard) rather than propagating.
        """
        # --- 6a. Rebalance: generate candidate action_dist ---
        try:
            action_dist, representative_embedding, representative_decision_id = (
                generate_candidate_action_dist(
                    candidate.cluster_key,
                    _tsdb_conn=tsdb_conn,
                    _milvus=milvus_client,
                )
            )
        except NoStrategyError as exc:
            self._log.warning(
                "autoloop.driver.rebalance_failed",
                cluster_key=candidate.cluster_key,
                detail=str(exc),
            )
            return 0

        # --- 6b. Pre-ev_loss snapshot (before A/B) ---
        try:
            pre_ev_loss_val = ev_loss(
                candidate.cluster_key,
                _tsdb_conn=tsdb_conn,
                _milvus=milvus_client,
            )
        except NoStrategyError:
            pre_ev_loss_val = None  # column nullable; persist as NULL

        # --- 6c. Build PatchSpec — LOOP-06 source='autoloop' ---
        patch_spec = PatchSpec(
            cluster_key=candidate.cluster_key,
            decision_id=representative_decision_id,
            action_dist=action_dist,
            embedding=representative_embedding,
            gto_score=0.5,  # Neutral default (05-RESEARCH.md Pitfall 6)
            confidence=0.5,  # Neutral default
            prev_node_id=None,
            source="autoloop",  # LOOP-06
        )

        # --- 6d. A/B validate ---
        try:
            validation = validate(
                patch_spec,
                config,
                baseline_engine=self._baseline_engine,
                sim_adapter=self._sim_adapter,
                seed=self._validation_seed,
                _tsdb_conn=tsdb_conn,
                _milvus=milvus_client,
            )
        except Exception as exc:  # VALN-03: validation errors → no apply
            self._log.error(
                "autoloop.driver.validation_failed",
                cluster_key=candidate.cluster_key,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return 0

        # --- 6e. LOOP-03 acceptance gate: strict ci_low > 0 ---
        ci_low = validation.confidence_interval[0]
        if validation.zero_hits or ci_low <= 0:
            self._log.info(
                "autoloop.driver.discard",
                cluster_key=candidate.cluster_key,
                ci_low=ci_low,
                zero_hits=validation.zero_hits,
            )
            return 0

        # --- 6f. Apply ---
        # post_ev_loss_val is stored as NULL because pre_ev_loss_val is a KL divergence
        # (nats, always >= 0) while validation.ev_loss_delta is a mean log-likelihood ratio
        # (log P_cand - log P_base). Subtracting them produces a meaningless number.
        # NULL explicitly signals that post-patch KL was not independently computed.
        # See CR-01 in 05-REVIEW.md. A future phase can compute genuine post-KL by
        # re-running ev_loss() after PatchEngine.apply().
        post_ev_loss_val = None
        record = patch_engine.apply(
            patch_spec,
            validation=validation,
            pre_ev_loss=pre_ev_loss_val,
            post_ev_loss=post_ev_loss_val,
            _tsdb_conn=tsdb_conn,
            _milvus=milvus_client,
        )

        self._log.info(
            "autoloop.driver.applied",
            cluster_key=candidate.cluster_key,
            patch_id=str(record.patch_id),
            ci_low=ci_low,
            ev_loss_delta=validation.ev_loss_delta,
        )
        return 1

    def _last_record_session_id(self, conn: Any) -> str | None:
        """Most recent session_id with at least one ev_loss row in metrics.

        Returns None if the metrics table is empty or has no ev_loss rows.
        """
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id FROM metrics WHERE metric_name = 'ev_loss' ORDER BY ts DESC LIMIT 1"
            )
            row = cur.fetchone()
        return row[0] if row else None

    def _build_engine_and_adapter(self) -> tuple[Any, Any]:
        """Lazy-build baseline KNNDecisionEngine + SimAdapter on first need.

        Deferred to avoid heavy import cost on the LOOP-05 disabled path.
        """
        from src.decision_engine.engine import engine_from_env
        from src.sim.adapter import SimAdapter

        engine = engine_from_env()
        adapter = SimAdapter()
        return engine, adapter


# ---------------------------------------------------------------------------
# Env-var helpers (production path only — tests inject mocks via constructor)
# ---------------------------------------------------------------------------


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables.

    TSDB_PASSWORD is required; all other vars have sensible defaults.
    Copied verbatim from src/metrics/ev_loss.py to keep driver self-contained.
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
