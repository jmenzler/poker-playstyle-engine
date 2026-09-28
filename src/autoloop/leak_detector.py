"""Leak detector: scans metrics rows for clusters with EV loss above threshold.  long-ok

Five filters applied before ranking:
  1. tau_leak filter — applied in SQL (metric_name='ev_loss' AND value > tau_leak)
  2. min_observations guard — cumulative observation count >= config.min_observations
  3. seed-source filter — skip clusters whose current active node has source='seed'
     (seeds are curated GTO baselines; the auto-loop does not patch their region —
     this filter is the policy, since the engine no longer guards seeds at apply time)
  4. locked-from-autoloop filter — skip clusters whose active node has
     locked_from_autoloop=TRUE (operator UI lock; migration 011).
  5. leak-suppression filter — skip clusters with an active leak_suppressions row
     (operator UI [accept as intentional]; migration 012). Honors LOOP-01 +
     PTCH-01 operator invariant (BLOCKER 2 / INTG-02 closure).

Ranking formula (05-CONTEXT.md Decision 4, locked):
  score = ev_loss * log(max(n_obs, 2))

The log(n_obs) term gives a sample-size bonus to established clusters while
keeping high-ev_loss clusters competitive — pure ev_loss ranking over-weights
noisy tiny-sample clusters.

Returns at most config.max_patches_per_run candidates, sorted by score descending.
"""

from __future__ import annotations

import contextlib
import math
import os
import uuid
from typing import Any

import msgspec

from src._config import AutoLoopConfig
from src._log import get_logger
from src.db import timescale

log = get_logger("autoloop.leak_detector")

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_EV_LOSS_QUERY = (
    "SELECT cluster_key, value AS ev_loss "
    "FROM metrics "
    "WHERE session_id = %s AND metric_name = 'ev_loss' AND value > %s "
    "ORDER BY ts DESC, cluster_key ASC"
)

_OBSERVATION_COUNT_QUERY = "SELECT COUNT(*) FROM observations WHERE cluster_key = %s"

_ACTIVE_NODE_QUERY = (
    "SELECT sn.node_id, sn.source, sn.locked_from_autoloop, "
    "       EXISTS (SELECT 1 FROM leak_suppressions ls "
    "               WHERE ls.cluster_key = sn.cluster_key AND ls.active = TRUE) "
    "         AS is_suppressed "
    "FROM strategy_nodes sn "
    "WHERE sn.cluster_key = %s AND sn.active = TRUE "
    "LIMIT 1"
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ClusterCandidate(msgspec.Struct, frozen=True, kw_only=True):
    """A cluster candidate identified for patching.

    Fields:
        cluster_key:   cluster bucket identifier.
        ev_loss:       KL divergence value from the metrics row.
        n_obs:         cumulative observation count (all sessions).
        prev_node_id:  UUID of the current active strategy_node for this cluster,
                       or None if no active node exists.
        score:         ranking score = ev_loss * log(max(n_obs, 2)).
    """

    cluster_key: str
    ev_loss: float
    n_obs: int
    prev_node_id: uuid.UUID | None
    score: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_patch_candidates(
    session_id: str,
    config: AutoLoopConfig,
    *,
    _tsdb_conn: Any = None,
) -> list[ClusterCandidate]:
    """Return ranked leak candidates for patching.

    Reads metric rows for the given session, applies three filters, ranks by
    score = ev_loss * log(max(n_obs, 2)), and returns the top
    config.max_patches_per_run candidates.

    Args:
        session_id: The RECORD session to scan for leaked clusters.
        config: AutoLoopConfig with tau_leak, min_observations, max_patches_per_run.
        _tsdb_conn: Test-injection hook. Production callers NEVER pass this.

    Returns:
        List of ClusterCandidate objects sorted by score descending,
        length <= config.max_patches_per_run.
    """
    _owns_conn = _tsdb_conn is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    try:
        return _select_patch_candidates_inner(session_id, config, conn)
    finally:
        if _owns_conn:
            with contextlib.suppress(Exception):
                conn.close()


def _select_patch_candidates_inner(
    session_id: str,
    config: AutoLoopConfig,
    conn: Any,
) -> list[ClusterCandidate]:
    """Inner implementation of select_patch_candidates (connection already established)."""
    # Step 1: Fetch all clusters with ev_loss > tau_leak for this session.
    with conn.cursor() as cur:
        cur.execute(_EV_LOSS_QUERY, (session_id, config.tau_leak))
        leak_rows = cur.fetchall()

    log.info(
        "autoloop.leak_detector.ev_loss_rows",
        session_id=session_id,
        n_leak_rows=len(leak_rows),
        tau_leak=config.tau_leak,
    )

    candidates: list[ClusterCandidate] = []

    for cluster_key, ev_loss_val in leak_rows:
        # Step 2: n_obs guard — cumulative count across all sessions.
        with conn.cursor() as cur:
            cur.execute(_OBSERVATION_COUNT_QUERY, (cluster_key,))
            obs_row = cur.fetchone()
        n_obs = int(obs_row[0]) if obs_row else 0

        if n_obs < config.min_observations:
            log.info(
                "autoloop.leak_detector.skip.min_observations",
                cluster_key=cluster_key,
                n_obs=n_obs,
                min_observations=config.min_observations,
            )
            continue

        # Step 3: seed-source filter — skip if active node has source='seed'.
        with conn.cursor() as cur:
            cur.execute(_ACTIVE_NODE_QUERY, (cluster_key,))
            active_row = cur.fetchone()

        prev_node_id: uuid.UUID | None = None
        if active_row is not None:
            node_id_str, source, locked, suppressed = active_row
            if source == "seed":
                log.info(
                    "autoloop.leak_detector.skip.seed_source",
                    cluster_key=cluster_key,
                )
                continue
            if source == "manual":
                log.info(
                    "autoloop.leak_detector.skip.manual_source",
                    cluster_key=cluster_key,
                )
                continue
            if locked:
                log.info(
                    "autoloop.leak_detector.skip.locked",
                    cluster_key=cluster_key,
                )
                continue
            if suppressed:
                log.info(
                    "autoloop.leak_detector.skip.suppressed",
                    cluster_key=cluster_key,
                )
                continue
            # Non-seed, unlocked, unsuppressed active node: capture for PatchEngine supersession.
            prev_node_id = uuid.UUID(str(node_id_str))
        # else: no active node — keep existing prev_node_id=None semantics (cluster
        # has no active strategy_node yet; the patch creates the first one).

        # Step 4: compute ranking score.
        score = float(ev_loss_val) * math.log(max(n_obs, 2))

        candidates.append(
            ClusterCandidate(
                cluster_key=cluster_key,
                ev_loss=float(ev_loss_val),
                n_obs=n_obs,
                prev_node_id=prev_node_id,
                score=score,
            )
        )

    # Step 5: sort by score descending; truncate to max_patches_per_run.
    candidates.sort(key=lambda c: c.score, reverse=True)
    result = candidates[: config.max_patches_per_run]

    log.info(
        "autoloop.leak_detector.candidates_selected",
        session_id=session_id,
        n_candidates=len(result),
        max_patches_per_run=config.max_patches_per_run,
    )
    return result


# ---------------------------------------------------------------------------
# Env-var helpers (production path only — tests inject mocks directly)
# ---------------------------------------------------------------------------


def _tsdb_dsn_from_env() -> str:
    """Build a TimescaleDB DSN from environment variables.

    TSDB_PASSWORD is required; all other vars have sensible defaults.
    Copied verbatim from src/metrics/ev_loss.py (same pattern).
    """
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
