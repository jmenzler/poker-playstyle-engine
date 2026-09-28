"""Per-spot batch solver queue driver for 168h unattended operation.  long-ok

Drains 73,570 flagged_sparse observations priority-ordered by
frequency x ev_loss x neighbor_distance (D-01/D-02), solves each via
ProcessPoolExecutor (~11 workers, D-11), persists COMPLETE solve context to
solver_cache.spot_features BEFORE injecting through PatchEngine (D-16,
IRREVERSIBLE), skips already-covered spots (D-04), re-sims on drain (D-03),
runs aggressive cheap eval cadence (D-05/D-15), and guards with disk floor /
checkpoint-resume / heartbeat / retention cap (D-17).

Exports:
    QueueDriver       -- main driver class
    SolverQueueCheckpoint -- atomic checkpoint struct
    _check_disk       -- disk watchdog helper (also used in tests)
    _should_pause_on_loo -- auto-pause predicate (used in tests)
"""

from __future__ import annotations

import json
import re
import shutil
import signal
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from src._config import AutoLoopConfig, SolverQueueConfig, load_toml_config
from src._errors import (
    NoStrategyError,
    SolverParseError,
)
from src._log import get_logger
from src.autoloop.checkpoint import checkpoint_path_for, write_checkpoint
from src.canonicalizer.encoder import _STREET_DELIM, _infer_pot_type, _parse_action_tokens
from src.db.milvus import DEFAULT_RPC_TIMEOUT_S as _MILVUS_RPC_TIMEOUT_S
from src.eval.solver_cache import SolverCacheEntry, persist_solve
from src.patch_engine import FEATURE_SPEC_VERSION as _FEATURE_SPEC_VERSION
from src.patch_engine import _parse_cluster_key
from src.solver.flop_spot import flop_spot_key, flop_start_stack_bb, is_multiway
from src.solver.nav_line import MultiwayNavError, parse_nav_line
from src.solver.postflop_cli import (
    PostflopCliBackend,
    SolverResult,
    SolverSpot,
    raw_harvest_to_solver_result,
)
from src.solver.range_resolver import _resolve_ranges, build_range_lookup
from src.study import corpus_versions
from src.study.harvest_ranges import HarvestRangeRow, build_seat_payloads, persist_harvest_range
from tools.upsert_milvus import (
    build_postflop_weight_vector,
    build_preflop_weight_vector,
    normalize,
)
from tools.zscore_fit import load_manifest

log = get_logger("solver.queue_driver")

DEFAULT_CONFIG_PATH = Path("config/autoloop.toml")
PALETTE_DIR = Path("research/preflop-ranges/outputs")
_ZSCORE_PREFLOP_PATH = Path("tools/zscore_preflop.json")
_ZSCORE_POSTFLOP_PATH = Path("tools/zscore_postflop.json")


@lru_cache(maxsize=2)
def _load_zscore_transform(collection: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, std, weights) for a Milvus collection — the SAME transform the bulk
    loader (tools.upsert_milvus) and KNNDecisionEngine apply, so solver-written
    nodes land in the identical z-scored+weighted space the live query searches."""
    if collection == "preflop_decisions":
        mean, std = load_manifest(_ZSCORE_PREFLOP_PATH)
        return mean, std, build_preflop_weight_vector()
    mean, std = load_manifest(_ZSCORE_POSTFLOP_PATH)
    return mean, std, build_postflop_weight_vector()


def _normalize_for_corpus(raw_embedding: list, collection: str) -> list | None:
    """Apply the shared corpus transform to a raw observation embedding.

    Returns None (caller skips the Milvus write, solve still cached) when the
    embedding is absent or its dim disagrees with the manifest — never silently
    stores a wrong-space vector."""
    if not raw_embedding:
        return None
    raw = np.asarray(raw_embedding, dtype=np.float64)
    mean, std, weights = _load_zscore_transform(collection)
    if raw.shape != mean.shape:
        return None
    return normalize(raw, mean, std, weights).tolist()


_PRIORITY_SQL = """
SELECT
    o.obs_id,
    o.decision_id,
    o.cluster_key,
    o.max_neighbor_distance,
    o.felt_snapshot,
    COUNT(*) OVER (PARTITION BY o.cluster_key) AS cluster_freq,
    COALESCE(m.value, 0.0) AS ev_loss,
    COUNT(*) OVER (PARTITION BY o.cluster_key)
        * COALESCE(m.value, 0.0)
        * o.max_neighbor_distance AS priority_score,
    o.embedding,
    o.hand_id
FROM observations o
LEFT JOIN LATERAL (
    SELECT value FROM metrics
    WHERE cluster_key = o.cluster_key
      AND metric_name = 'ev_loss'
    ORDER BY ts DESC LIMIT 1
) m ON TRUE
WHERE o.flagged_sparse = TRUE
  AND o.cluster_key NOT LIKE '%%street_class=preflop%%'
  AND o.cluster_key LIKE '%%n_players_active=2%%'
  AND o.cluster_key NOT LIKE '%%pot_type=limp%%'
  AND NOT EXISTS (
      SELECT 1 FROM solver_cache sc
      WHERE sc.decision_id = o.decision_id
  )
ORDER BY (COALESCE(jsonb_array_length(o.felt_snapshot -> 'board_cards'), 0) = 3) DESC,
         o.max_neighbor_distance DESC
LIMIT %s
"""


class SolverQueueCheckpoint(msgspec.Struct, frozen=True, kw_only=True):
    """Atomic checkpoint for the solver queue driver.

    Written every checkpoint_every spots via write_checkpoint (atomic os.replace).
    Resume re-fetches from the top of the priority queue; correctness comes from the
    solver_cache dedup, not from last_obs_id (which records the last spot seen only).
    """

    run_id: str
    run_epoch: str
    spots_submitted: int
    spots_completed: int
    spots_failed: int
    last_obs_id: str
    last_eval_suite_at: int


def _read_solver_checkpoint(path: Path) -> SolverQueueCheckpoint | None:
    """Return the SolverQueueCheckpoint at path, or None if absent.

    Decodes as SolverQueueCheckpoint (not RunCheckpoint) — the two structs are distinct.
    A corrupt or mismatched file raises msgspec.DecodeError (fail loudly).
    """
    if not path.exists():
        return None
    return msgspec.json.decode(path.read_bytes(), type=SolverQueueCheckpoint)


def _check_disk(path: str = "/", *, floor_gb: float = 20.0) -> None:
    """Raise RuntimeError if free disk on path < floor_gb (D-17 disk watchdog).

    Call every checkpoint_every spots. On breach the driver writes a final
    checkpoint and exits cleanly — does not crash.
    """
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < floor_gb:
        raise RuntimeError(f"Disk below floor: {free_gb:.1f} GB free on {path} (floor={floor_gb} GB)")


def _should_pause_on_loo(
    loo_result: Any,
    *,
    tvd_floor: float,
    last_tvd: float | None,
) -> bool:
    """Return True to pause the driver (D-05/D-15): TVD > 2x floor, or absolute jump > 0.15."""  # long-ok
    tvd = loo_result.tier1_tvd
    return tvd > tvd_floor * 2.0 or (last_tvd is not None and tvd > last_tvd + 0.15)


class QueueDriver:
    """Per-spot priority queue driver for batch solver distillation."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        cfg = load_toml_config(config_path, AutoLoopConfig)
        self._cfg: SolverQueueConfig = cfg.solver_queue
        self._run_id = run_id or f"solver-queue-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        self._stop_requested = False
        self._palette_lookup: dict | None = None

    def _install_signal_handlers(self) -> None:
        def _handler(sig: int, frame: Any) -> None:
            log.info("solver.queue_driver.stop_signal", signal=sig, run_id=self._run_id)
            self._stop_requested = True

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _get_palette_lookup(self) -> dict:
        if self._palette_lookup is None:
            self._palette_lookup = build_range_lookup(PALETTE_DIR) if PALETTE_DIR.exists() else {}
        return self._palette_lookup

    def _fetch_batch(
        self,
        conn: Any,
        *,
        limit: int,
    ) -> list[dict]:
        """Fetch next batch of priority-ordered flagged_sparse observations.

        Resume position is NOT seeked here: the priority query re-ranks from the top
        each call, and already-solved spots are excluded by the solver_cache dedup
        (NOT EXISTS). Deduplicates by decision_id within the batch.
        """
        with conn.cursor() as cur:
            cur.execute(_PRIORITY_SQL, (limit,))
            rows = cur.fetchall()

        seen_decision_ids: set[str] = set()
        batch: list[dict] = []
        for row in rows:
            (
                obs_id,
                decision_id,
                cluster_key,
                max_neighbor_distance,
                felt_snapshot,
                cluster_freq,
                ev_loss,
                priority_score,
                embedding,
                hand_id,
            ) = row
            if decision_id in seen_decision_ids:
                continue
            seen_decision_ids.add(decision_id)
            batch.append(
                {
                    "obs_id": str(obs_id),
                    "decision_id": str(decision_id),
                    "cluster_key": str(cluster_key),
                    "max_neighbor_distance": float(max_neighbor_distance or 0.0),
                    "felt_snapshot": felt_snapshot or {},
                    "cluster_freq": int(cluster_freq or 0),
                    "ev_loss": float(ev_loss or 0.0),
                    "priority_score": float(priority_score or 0.0),
                    "embedding": list(embedding) if embedding is not None else [],
                    "hand_id": str(hand_id) if hand_id else "",
                }
            )
        return batch

    def _count_sparse(self, conn: Any) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM observations WHERE flagged_sparse = TRUE"
                " AND cluster_key NOT LIKE '%street_class=preflop%'"
                " AND NOT EXISTS (SELECT 1 FROM solver_cache sc WHERE sc.decision_id = observations.decision_id)"
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def _on_drain(self, conn: Any, *, _sim_runner: Callable[[], None] | None = None) -> None:
        """Invoked when flagged_sparse queue is exhausted (D-03 re-sim on drain)."""
        log.info("solver.queue_driver.drain", run_id=self._run_id)
        if _sim_runner is not None:
            _sim_runner()
            log.info("solver.queue_driver.resim_complete", run_id=self._run_id)

    def _ensure_retention_policy(self, conn: Any) -> None:
        """Add observations retention policy if not already present (D-17).

        Checks for existing policy before adding to avoid duplicate errors.
        """
        days = self._cfg.retention_days
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM timescaledb_information.jobs "
                    "WHERE proc_name = 'policy_retention' "
                    "  AND config->>'hypertable_name' = 'observations'"
                )
                row = cur.fetchone()
            if row and row[0] > 0:
                log.info("solver.queue_driver.retention_policy_exists")
                return
            with conn.cursor() as cur:
                cur.execute(f"SELECT add_retention_policy('observations', INTERVAL '{days} days')")
            log.info("solver.queue_driver.retention_policy_added", days=days)
        except Exception as exc:
            log.warning("solver.queue_driver.retention_policy_skip", error=str(exc))

    def run(
        self,
        *,
        _tsdb_conn: Any = None,
        _milvus: Any = None,
        _solver: Any = None,
        _patch_engine: Any = None,
        _sim_runner: Callable[[], None] | None = None,
        install_signals: bool = True,
    ) -> dict[str, Any]:
        """Drain the priority queue. Returns summary dict with spots_completed, spots_failed, duration_s."""
        if install_signals:
            self._install_signal_handlers()

        ckpt_path = checkpoint_path_for(self._run_id, self._cfg.checkpoint_path)
        existing = _read_solver_checkpoint(ckpt_path)

        if existing is not None:
            spots_submitted = existing.spots_submitted
            spots_completed = existing.spots_completed
            spots_failed = existing.spots_failed
            last_obs_id = existing.last_obs_id
            last_eval_suite_at = existing.last_eval_suite_at
            run_epoch = existing.run_epoch
            log.info(
                "solver.queue_driver.resume",
                run_id=self._run_id,
                spots_completed=spots_completed,
                last_obs_id=last_obs_id,
            )
        else:
            spots_submitted = 0
            spots_completed = 0
            spots_failed = 0
            last_obs_id = None
            last_eval_suite_at = 0
            run_epoch = datetime.now(UTC).isoformat()

        start_epoch = datetime.now(UTC)
        consecutive_db_failures = 0
        last_loo_tvd: float | None = None

        if _tsdb_conn is None:
            from src.db import timescale
            from src.eval.solver_cache import _tsdb_dsn_from_env

            # autocommit: each per-spot persist_solve / skip INSERT must commit immediately
            # so it is durable AND visible to the per-DP dedup (solver_cache.decision_id).
            # Without it, writes sit in one uncommitted transaction → dedup is blind →
            # the priority queue re-fetches and re-solves the same spots forever.
            _tsdb_conn = timescale.connect(_tsdb_dsn_from_env(), autocommit=True)

        # Without a Milvus client, _inject and the LOO gate get milvus=None and every
        # solver node is written to TSDB only — write-only, never retrievable (gap #1).
        if _milvus is None:
            from src.db.milvus import connect_from_env

            _milvus = connect_from_env()

        self._ensure_retention_policy(_tsdb_conn)

        solver = _solver or self._default_solver()
        palette_lookup = self._get_palette_lookup()
        cfg = self._cfg

        def _run_cadence_checks() -> bool:
            """Checkpoint / disk-watchdog / LOO / suite cadence (keyed off spots_completed).

            Returns True when the caller must stop the run (disk floor breach or LOO pause).
            """
            nonlocal last_eval_suite_at, last_loo_tvd
            if spots_completed % cfg.checkpoint_every == 0 and spots_completed > 0:
                try:
                    _check_disk(floor_gb=cfg.disk_floor_gb)
                except RuntimeError as exc:
                    log.error("solver.queue_driver.disk_floor_halt", error=str(exc))
                    write_checkpoint(
                        ckpt_path,
                        SolverQueueCheckpoint(
                            run_id=self._run_id,
                            run_epoch=run_epoch,
                            spots_submitted=spots_submitted,
                            spots_completed=spots_completed,
                            spots_failed=spots_failed,
                            last_obs_id=last_obs_id or "",
                            last_eval_suite_at=last_eval_suite_at,
                        ),
                    )
                    self._stop_requested = True
                    return True

                write_checkpoint(
                    ckpt_path,
                    SolverQueueCheckpoint(
                        run_id=self._run_id,
                        run_epoch=run_epoch,
                        spots_submitted=spots_submitted,
                        spots_completed=spots_completed,
                        spots_failed=spots_failed,
                        last_obs_id=last_obs_id or "",
                        last_eval_suite_at=last_eval_suite_at,
                    ),
                )
                log.info(
                    "solver.queue_driver.heartbeat",
                    run_id=self._run_id,
                    spots_completed=spots_completed,
                    spots_failed=spots_failed,
                )

            if spots_completed % cfg.eval_loo_every == 0 and spots_completed > 0:
                try:
                    from src.eval.loo_gate import run_loo_gate

                    loo_result = run_loo_gate(_tsdb_conn=_tsdb_conn, _milvus=_milvus)
                    log.info(
                        "solver.queue_driver.loo_gate",
                        spots_completed=spots_completed,
                        tier1_tvd=loo_result.tier1_tvd,
                        passed=loo_result.passed,
                    )
                    from src._config import load_eval_config

                    eval_cfg = load_eval_config()
                    if _should_pause_on_loo(loo_result, tvd_floor=eval_cfg.tvd_floor, last_tvd=last_loo_tvd):
                        log.warning(
                            "solver.queue_driver.auto_pause",
                            tier1_tvd=loo_result.tier1_tvd,
                            last_tvd=last_loo_tvd,
                        )
                        self._stop_requested = True
                        return True
                    last_loo_tvd = loo_result.tier1_tvd
                except Exception as exc:
                    log.warning("solver.queue_driver.loo_gate_failed", error=str(exc))

            if spots_completed % cfg.eval_suite_every == 0 and spots_completed > 0:
                try:
                    from src.eval.suite import run_suite

                    run_suite(_tsdb_conn=_tsdb_conn, _milvus=_milvus)
                    last_eval_suite_at = spots_completed
                except Exception as exc:
                    log.warning("solver.queue_driver.suite_failed", error=str(exc))
            return False

        try:
            while not self._stop_requested:
                batch = self._fetch_batch(_tsdb_conn, limit=cfg.checkpoint_every * 2)

                if not batch:
                    self._on_drain(_tsdb_conn, _sim_runner=_sim_runner)
                    batch = self._fetch_batch(_tsdb_conn, limit=cfg.checkpoint_every * 2)
                    if not batch:
                        log.info("solver.queue_driver.queue_empty_after_resim", run_id=self._run_id)
                        break

                last_obs_id = batch[-1]["obs_id"]
                groups, multiway_members = _group_by_flop_spot(batch)

                for member in multiway_members:
                    log.info(
                        "solver.queue_driver.multiway_skip",
                        obs_id=member["obs_id"],
                        decision_id=member.get("decision_id"),
                        cluster_key=member["cluster_key"],
                    )
                    _handle_multiway_skip(member, conn=_tsdb_conn)
                    spots_failed += 1

                with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
                    futures = {
                        pool.submit(_solve_group, members, solver, cfg): members
                        for members in groups.values()
                    }
                    spots_submitted += sum(len(m) for m in groups.values())

                    for fut in as_completed(futures):
                        members = futures[fut]
                        try:
                            result = fut.result()
                        except subprocess.TimeoutExpired as exc:
                            for member in members:
                                log.warning(
                                    "solver.queue_driver.spot_timed_out",
                                    obs_id=member["obs_id"],
                                    decision_id=member.get("decision_id"),
                                    cluster_key=member["cluster_key"],
                                    timeout_s=cfg.timeout_s,
                                    error=str(exc),
                                )
                                _handle_timeout_skip(member, conn=_tsdb_conn)
                                spots_failed += 1
                            continue
                        except (NoStrategyError, SolverParseError) as exc:
                            for member in members:
                                log.warning(
                                    "solver.queue_driver.spot_failed",
                                    obs_id=member["obs_id"],
                                    cluster_key=member["cluster_key"],
                                    error=str(exc),
                                )
                                _handle_solve_failed(member, conn=_tsdb_conn)
                                spots_failed += 1
                            # Marked covered above; a deterministic unsolvable/unmappable spot
                            # is not DB ill-health, so it must not trip the db-failure breaker.
                            continue
                        except Exception as exc:
                            log.error(
                                "solver.queue_driver.group_solve_error",
                                obs_id=members[0]["obs_id"] if members else "",
                                error=str(exc),
                            )
                            spots_failed += len(members)
                            consecutive_db_failures += 1
                            if consecutive_db_failures >= 5:
                                log.error(
                                    "solver.queue_driver.circuit_broken",
                                    run_id=self._run_id,
                                    consecutive_db_failures=consecutive_db_failures,
                                )
                                self._stop_requested = True
                                break
                            continue

                        if result.get("skipped_multiway"):
                            for member in result["members"]:
                                log.info(
                                    "solver.queue_driver.multiway_skip",
                                    obs_id=member["obs_id"],
                                    decision_id=member.get("decision_id"),
                                    cluster_key=member["cluster_key"],
                                )
                                _handle_multiway_skip(member, conn=_tsdb_conn)
                                spots_failed += 1
                            continue

                        if result.get("skipped_placeholder"):
                            for member in result["members"]:
                                log.info(
                                    "solver.queue_driver.placeholder_skip",
                                    obs_id=member["obs_id"],
                                    decision_id=member.get("decision_id"),
                                    cluster_key=member["cluster_key"],
                                )
                                _handle_placeholder_skip(member, conn=_tsdb_conn)
                                spots_failed += 1
                            continue

                        member_rows = zip(
                            result["members"],
                            result["results"],
                            result.get("raw_results") or [None] * len(result["results"]),
                            result.get("nav_lines") or [None] * len(result["results"]),
                            strict=True,
                        )
                        for member, member_result, raw_result, nav_line in member_rows:
                            try:
                                if member_result is None:
                                    log.warning(
                                        "solver.queue_driver.nav_failed",
                                        obs_id=member["obs_id"],
                                        decision_id=member.get("decision_id"),
                                        cluster_key=member["cluster_key"],
                                    )
                                    _handle_nav_failed(member, conn=_tsdb_conn)
                                    spots_failed += 1
                                    continue
                                member_street = (member.get("felt_snapshot") or {}).get("street") or ""
                                range_narrowing = "none" if member_street == "flop" else "flop_rooted"
                                _, member_context = _build_spot(member, palette_lookup, cfg)
                                _inject_node(
                                    member,
                                    result["spot"],
                                    member_result,
                                    member_context,
                                    conn=_tsdb_conn,
                                    milvus=_milvus,
                                    range_narrowing=range_narrowing,
                                    nav_ok=True,
                                )
                                if raw_result is not None and nav_line is not None:
                                    _persist_member_harvest(
                                        member,
                                        raw_result,
                                        street=member_street,
                                        hero_seat=int(nav_line["hero_player"]),
                                        conn=_tsdb_conn,
                                    )
                                spots_completed += 1
                                consecutive_db_failures = 0
                            except Exception as exc:
                                log.error(
                                    "solver.queue_driver.db_error",
                                    obs_id=member["obs_id"],
                                    error=str(exc),
                                )
                                spots_failed += 1
                                consecutive_db_failures += 1
                                if consecutive_db_failures >= 5:
                                    log.error(
                                        "solver.queue_driver.circuit_broken",
                                        run_id=self._run_id,
                                        consecutive_db_failures=consecutive_db_failures,
                                    )
                                    self._stop_requested = True
                                    break

                            if _run_cadence_checks():
                                break

                        if self._stop_requested:
                            break

                if self._stop_requested:
                    break

        finally:
            write_checkpoint(
                ckpt_path,
                SolverQueueCheckpoint(
                    run_id=self._run_id,
                    run_epoch=run_epoch,
                    spots_submitted=spots_submitted,
                    spots_completed=spots_completed,
                    spots_failed=spots_failed,
                    last_obs_id=last_obs_id or "",
                    last_eval_suite_at=last_eval_suite_at,
                ),
            )

        duration_s = (datetime.now(UTC) - start_epoch).total_seconds()
        # Only snapshot when spots actually landed — empty versions are meaningless.
        if spots_completed > 0:
            corpus_versions.snapshot(
                label=f"solver-{self._run_id}",
                source="solver",
                _tsdb_conn=_tsdb_conn,
            )
        return {
            "run_id": self._run_id,
            "spots_submitted": spots_submitted,
            "spots_completed": spots_completed,
            "spots_failed": spots_failed,
            "duration_s": duration_s,
        }

    @staticmethod
    def _default_solver() -> PostflopCliBackend:
        import os

        binary = Path(
            os.environ.get("POSTFLOP_CLI_BIN", "~/postflop-cli/target/release/postflop-cli")
        ).expanduser()
        return PostflopCliBackend(binary_path=binary)


_CLUSTER_KEY_POT_TYPE_RE = re.compile(r"(?:^|\|)pot_type=([^|]+)")

_RAISE_VERB_PREFIXES = ("open", "raise", "3b", "4b", "5b")

_PLACEHOLDER_RANGE = "AA-22,AKs-A2s"


def _is_placeholder_range(range_str: str) -> bool:
    """Return True if range_str is the canonical placeholder for uncovered spots."""
    return range_str == _PLACEHOLDER_RANGE


def _handle_placeholder_skip(obs_row: dict, *, conn: Any) -> None:
    """Record a skip entry for a placeholder spot so the dedup query excludes it next pass."""
    entry = SolverCacheEntry(
        cluster_key=obs_row.get("cluster_key") or "",
        decision_id=obs_row.get("decision_id"),
        action_dist={},
        exploitability_pct=0.0,
        solver_version="skipped_placeholder",
        spot_features=None,
    )
    persist_solve(entry, _tsdb_conn=conn)


def _handle_timeout_skip(obs_row: dict, *, conn: Any) -> None:
    """Record a marker for a spot that exceeded the wall-clock cap so resumes don't re-grind it.

    Without this, a timed-out solve persists nothing → no solver_cache.decision_id row →
    the priority queue re-fetches and burns the full cap on it every resume. The marker
    (solver_version='timed_out') dedups it out and is SQL-countable / targetable for a
    later higher-cap retry pass.
    """
    entry = SolverCacheEntry(
        cluster_key=obs_row.get("cluster_key") or "",
        decision_id=obs_row.get("decision_id"),
        action_dist={},
        exploitability_pct=0.0,
        solver_version="timed_out",
        spot_features=None,
    )
    persist_solve(entry, _tsdb_conn=conn)


def _pot_type_from_cluster_key(cluster_key: str) -> str | None:
    """Extract pot_type token from a cluster_key like 'pot_type=3bet|...'."""
    m = _CLUSTER_KEY_POT_TYPE_RE.search(cluster_key)
    return m.group(1) if m else None


def _is_preflop_raise(verb: str) -> bool:
    """Return True if the verb represents a preflop raise (open/raise/3bet/4bet/5bet)."""
    v = verb.lower()
    return any(v.startswith(p) for p in _RAISE_VERB_PREFIXES)


def _derive_preflop_info(
    action_sequence: list[str],
    hero_pos: str,
) -> tuple[str, str, str, str, list[str]]:
    """Infer pot_type, villain_pos, opener_pos, bettor_pos, preflop tokens from action_sequence."""  # long-ok
    # Preflop slice: drop everything from the first "/" delimiter on, so postflop
    # raises don't leak into opener/bettor/pot_type derivation.
    preflop_seq: list[str] = []
    for tok in action_sequence:
        if tok == _STREET_DELIM:
            break
        preflop_seq.append(tok)
    action_seq_tuple = tuple(preflop_seq)
    parsed = _parse_action_tokens(action_seq_tuple)
    pot_type = _infer_pot_type(action_seq_tuple)

    raiser_positions: list[str] = []
    for tok_dict in parsed:
        if tok_dict.get("action") == "raise":
            raiser_positions.append(tok_dict["pos"])

    opener = raiser_positions[0] if raiser_positions else None
    bettor = (
        raiser_positions[-1]
        if len(raiser_positions) >= 2
        else raiser_positions[0]
        if raiser_positions
        else None
    )  # long-ok

    # Villain must still be in the hand: a folded raiser (limp-raise pots) gives the
    # wrong IP/OOP seat downstream and a dead player's range.
    last_action: dict[str, str] = {}
    for tok_dict in parsed:
        last_action[tok_dict["pos"]] = tok_dict.get("action", "")
    survivors = [p for p, a in last_action.items() if a != "fold" and p != hero_pos]
    aggressor_survivors = [p for p in raiser_positions if p in survivors]

    if aggressor_survivors:
        villain = aggressor_survivors[-1]
    elif survivors:
        villain = survivors[-1]
    elif bettor and bettor != hero_pos:
        villain = bettor
    elif opener and opener != hero_pos:
        villain = opener
    else:
        villain = "BB" if hero_pos != "BB" else "BTN"

    n_raises_needed = {"srp": 1, "3bet": 2, "4bet": 3}.get(pot_type, 1)
    preflop_tokens: list[str] = []
    raises_seen = 0
    in_preflop_close = False
    for raw_tok, tok_dict in zip(action_sequence, parsed, strict=False):
        action = tok_dict.get("action", "")
        if action == "raise":
            raises_seen += 1
            preflop_tokens.append(raw_tok)
            if raises_seen >= n_raises_needed:
                in_preflop_close = True
        elif action in ("call", "fold") and raises_seen > 0:
            preflop_tokens.append(raw_tok)
            if in_preflop_close and action == "call":
                break
        elif action in ("bet", "check") and raises_seen > 0:
            break
        elif raises_seen == 0:
            preflop_tokens.append(raw_tok)

    return pot_type, villain, opener or villain, bettor or villain, preflop_tokens


def _build_spot(obs_row: dict, palette_lookup: dict, cfg: SolverQueueConfig) -> tuple[SolverSpot, dict]:
    """Build SolverSpot + full_context dict from an obs row's felt_snapshot (D-16 source)."""
    felt = obs_row.get("felt_snapshot") or {}
    cluster_key = obs_row.get("cluster_key") or ""
    bet_sizes = list(cfg.bet_sizes)

    hero_pos = felt.get("hero_position") or felt.get("hero_pos") or "BTN"
    street = felt.get("street") or "flop"
    board_cards = felt.get("board_cards") or []
    hero_hole = felt.get("hero_hole_cards") or []
    pot_bb = felt.get("pot_size_bb")
    stack_bb = felt.get("effective_stack_bb")
    if not pot_bb or not stack_bb:
        raise ValueError(
            f"missing/zero pot or stack for obs_id={obs_row.get('obs_id')!r} "
            f"cluster_key={cluster_key!r} (pot_bb={pot_bb}, stack_bb={stack_bb}) — "
            "refusing to solve a postflop spot on fabricated pot/stack"
        )
    prev_bet_bb = felt.get("hero_facing_bet_bb")
    n_players = int(felt.get("opponents_remaining") or 1) + 1
    action_line_full = felt.get("action_sequence") or []

    pot_type_from_ck = _pot_type_from_cluster_key(cluster_key)
    if pot_type_from_ck:
        pot_type = pot_type_from_ck
        _, villain_pos, opener_pos, bettor_pos, preflop_action_seq = _derive_preflop_info(
            action_line_full, hero_pos
        )
    else:
        pot_type, villain_pos, opener_pos, bettor_pos, preflop_action_seq = _derive_preflop_info(
            action_line_full, hero_pos
        )

    range_ip, range_oop = _resolve_ranges(
        obs_spot_features=felt,
        hero_pos=hero_pos,
        villain_pos=villain_pos,
        pot_type=pot_type,
        palette_lookup=palette_lookup,
        opener_pos=opener_pos,
        bettor_pos=bettor_pos,
    )

    pot = round(float(pot_bb) * 100)
    effective_stack = round(float(stack_bb) * 100)
    prev_bet = round(float(prev_bet_bb) * 100) if prev_bet_bb is not None else None

    # n_workers subprocesses solve concurrently under one container mem_limit, so
    # the per-spot gate budget is the total pool split across workers. 0 → no gate.
    memory_budget_mb = cfg.memory_budget_mb // max(1, cfg.n_workers) if cfg.memory_budget_mb > 0 else None

    spot = SolverSpot(
        pot=pot,
        effective_stack=effective_stack,
        board=list(board_cards),
        range_ip=range_ip,
        range_oop=range_oop,
        prev_bet=prev_bet,
        bet_sizes_flop_oop=bet_sizes,
        bet_sizes_flop_ip=bet_sizes,
        bet_sizes_turn_oop=bet_sizes,
        bet_sizes_turn_ip=bet_sizes,
        bet_sizes_river_oop=bet_sizes,
        bet_sizes_river_ip=bet_sizes,
        max_iterations=cfg.max_iterations,
        target_exploitability_pct=cfg.target_exploitability_pct,
        memory_budget_mb=memory_budget_mb,
    )

    full_context_partial: dict[str, Any] = {
        "board": list(board_cards),
        "hero_hole": list(hero_hole),
        "pot": pot,
        "effective_stack": effective_stack,
        "prev_bet": prev_bet,
        "action_line_full": action_line_full,
        "preflop_action_seq": preflop_action_seq,
        "street": street,
        "hero_pos": hero_pos,
        "villain_pos": villain_pos,
        "pot_type": pot_type,
        "n_players_at_street": n_players,
        "range_ip": range_ip,
        "range_oop": range_oop,
        "solver_settings": {
            "target_exploitability_pct": cfg.target_exploitability_pct,
            "max_iterations": cfg.max_iterations,
            "timeout_s": cfg.timeout_s,
            "bet_sizes_flop_oop": bet_sizes,
            "bet_sizes_flop_ip": bet_sizes,
            "bet_sizes_turn_oop": bet_sizes,
            "bet_sizes_turn_ip": bet_sizes,
            "bet_sizes_river_oop": bet_sizes,
            "bet_sizes_river_ip": bet_sizes,
            "solver_version": "postflop-cli",
        },
    }
    return spot, full_context_partial


def _solve_one_spot(obs_row: dict, solver: PostflopCliBackend, cfg: SolverQueueConfig) -> dict:
    """Worker submitted to ProcessPoolExecutor. Propagates solver exceptions to as_completed.

    Returns {"skipped_placeholder": True} for placeholder-range spots so the main
    loop can record a skip entry without re-trying the solve indefinitely.
    """
    palette_lookup = build_range_lookup(PALETTE_DIR) if PALETTE_DIR.exists() else {}
    spot, full_context_partial = _build_spot(obs_row, palette_lookup, cfg)

    range_ip = full_context_partial.get("range_ip", "")
    range_oop = full_context_partial.get("range_oop", "")
    if _is_placeholder_range(range_ip) or _is_placeholder_range(range_oop):
        return {
            "spot": spot,
            "solver_result": None,
            "full_context_partial": full_context_partial,
            "obs_row": obs_row,
            "skipped_placeholder": True,
        }

    solver_result = solver.solve(spot, timeout_s=cfg.timeout_s)
    return {
        "spot": spot,
        "solver_result": solver_result,
        "full_context_partial": full_context_partial,
        "obs_row": obs_row,
        "skipped_placeholder": False,
    }


def _flop_start_stack_by_hand(batch: list[dict[str, Any]]) -> dict[str, float]:
    """Map hand_id -> flop-start effective stack from each hand's flop DP in the batch."""
    out: dict[str, float] = {}
    for obs in batch:
        felt = obs.get("felt_snapshot") or {}
        if (felt.get("street") or "") != "flop":
            continue
        hand_id = obs.get("hand_id") or ""
        stack = felt.get("effective_stack_bb")
        if hand_id and stack is not None:
            out[hand_id] = float(stack)
    return out


def _group_by_flop_spot(
    batch: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, tuple[str, str, str, str, str], int], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Partition a batch into flop-spot groups + a multiway-skip list.

    Each group key (canonical_flop, range_config, flop_start_stack) is one harvest worker
    unit. Multiway (3+ player) DPs the HU solver cannot model are returned separately to be
    skipped+flagged. The per-hand flop-start stack (each hand's flop DP) keys the shared spot.
    """  # long-ok
    flop_start_by_hand = _flop_start_stack_by_hand(batch)
    groups: dict[tuple[str, tuple[str, str, str, str, str], int], list[dict[str, Any]]] = {}
    multiway_members: list[dict[str, Any]] = []
    for obs in batch:
        felt = obs.get("felt_snapshot") or {}
        if is_multiway(felt):
            multiway_members.append(obs)
            continue
        hero_pos = felt.get("hero_position") or felt.get("hero_pos") or "BTN"
        action_seq = felt.get("action_sequence") or []
        key = flop_spot_key(
            felt,
            action_seq,
            hero_pos,
            flop_start_stack_by_hand=flop_start_by_hand,
            hand_id=obs.get("hand_id") or None,
        )
        groups.setdefault(key, []).append(obs)
    return groups, multiway_members


def _build_group(
    group_members: list[dict[str, Any]],
    palette_lookup: dict[str, Any],
    cfg: SolverQueueConfig,
) -> tuple[SolverSpot, list[dict[str, object]], list[dict[str, Any]]]:
    """Build the shared flop-rooted SolverSpot + per-member nav_lines for one group.

    Returns (shared_spot, nav_lines, ordered_members) with nav_lines[i] parallel to
    ordered_members[i]. The spot is forced flop-rooted (3-card board + flop-start pot/stack)
    so the binary solves from the flop; each member's navigated turn/river node is read off
    that single solve.
    """  # long-ok
    flop_dp = next(
        (m for m in group_members if ((m.get("felt_snapshot") or {}).get("street") or "") == "flop"),
        None,
    )
    rep = flop_dp or group_members[0]
    spot, _ = _build_spot(rep, palette_lookup, cfg)

    rep_felt = rep.get("felt_snapshot") or {}
    flop_cards = list((rep_felt.get("board_cards") or [])[:3])

    flop_start_by_hand = _flop_start_stack_by_hand(group_members)
    hero_pos = rep_felt.get("hero_position") or rep_felt.get("hero_pos") or "BTN"
    flop_stack_bb = flop_start_stack_bb(rep_felt, flop_start_by_hand, rep.get("hand_id") or None)
    flop_pot_felt = (flop_dp or rep).get("felt_snapshot") or {}
    flop_pot_bb = float(flop_pot_felt.get("pot_size_bb") or 0.0)

    # Flop-root the shared spot: a 3-card board makes the binary solve from the flop and
    # deal turn/river per nav_line; pot/stack are the flop-start values shared by all members.
    spot = msgspec.structs.replace(
        spot,
        board=flop_cards,
        effective_stack=round(flop_stack_bb * 100),
        pot=round(flop_pot_bb * 100),
    )

    ordered_members: list[dict[str, Any]] = []
    nav_lines: list[dict[str, object]] = []
    for member in group_members:
        felt = member.get("felt_snapshot") or {}
        m_hero_pos = felt.get("hero_position") or felt.get("hero_pos") or hero_pos
        action_seq = felt.get("action_sequence") or []
        _, villain_pos, _, _, _ = _derive_preflop_info(action_seq, m_hero_pos)
        board_cards = list(felt.get("board_cards") or [])
        steps, hero_player, turn_card, river_card = parse_nav_line(
            action_seq, m_hero_pos, villain_pos, board_cards
        )
        nav_lines.append(
            {
                "steps": steps,
                "hero_player": hero_player,
                "turn_card": turn_card,
                "river_card": river_card,
            }
        )
        ordered_members.append(member)
    return spot, nav_lines, ordered_members


def _solve_group(
    group_members: list[dict[str, Any]], solver: PostflopCliBackend, cfg: SolverQueueConfig
) -> dict[str, Any]:
    """Pool worker: build the shared flop spot + nav_lines, run ONE solve_harvest call.

    Returns {"members": ordered_members, "results": [SolverResult|None], "spot": spot}.
    A placeholder-range group is returned with skipped_placeholder=True (never solved).
    Propagates solver exceptions to as_completed (mirrors _solve_one_spot).
    """  # long-ok
    palette_lookup = build_range_lookup(PALETTE_DIR) if PALETTE_DIR.exists() else {}
    try:
        spot, nav_lines, ordered_members = _build_group(group_members, palette_lookup, cfg)
    except MultiwayNavError:
        # 3+-position postflop line the HU solver cannot navigate. Skip gracefully so it
        # is marked covered and never trips the db-failure circuit breaker.
        return {"members": group_members, "results": [], "spot": None, "skipped_multiway": True}
    if _is_placeholder_range(spot.range_ip) or _is_placeholder_range(spot.range_oop):
        return {"members": ordered_members, "results": [], "spot": spot, "skipped_placeholder": True}
    parsed_top, raw_results = solver.solve_harvest_raw(spot, nav_lines, timeout_s=cfg.timeout_s)
    results = [raw_harvest_to_solver_result(r, parsed_top) for r in raw_results]
    return {
        "members": ordered_members,
        "results": results,
        "raw_results": raw_results,
        "nav_lines": nav_lines,
        "spot": spot,
        "skipped_placeholder": False,
    }


def _persist_member_harvest(
    member_obs: dict[str, Any],
    raw_result: dict[str, Any],
    *,
    street: str,
    hero_seat: int,
    conn: Any,
) -> None:
    """Persist both seats' range grids for one nav_ok member (the ✓ study artifact).

    One solve feeds both the kNN corpus row and the range-viewer contract;
    hero_action stays None (the priority queue does not fetch action_taken).
    """
    hand_id = member_obs.get("hand_id") or ""
    decision_id = member_obs.get("decision_id") or member_obs["obs_id"]
    if not hand_id:
        log.warning("solver.queue_driver.harvest_skip_no_hand_id", decision_id=decision_id)
        return

    hero_payload, villain_payload = build_seat_payloads(
        raw_result,
        street=street,
        hero_seat=hero_seat,
        hero_action=None,
        decision_id=decision_id,
    )
    persist_harvest_range(
        HarvestRangeRow(hand_id=hand_id, decision_id=decision_id, seat=hero_seat, payload=hero_payload),
        _tsdb_conn=conn,
    )
    persist_harvest_range(
        HarvestRangeRow(
            hand_id=hand_id, decision_id=decision_id, seat=1 - hero_seat, payload=villain_payload
        ),
        _tsdb_conn=conn,
    )


def _handle_multiway_skip(obs_row: dict[str, Any], *, conn: Any) -> None:
    """Record a multiway_hu_unsupported marker so the HU solver never re-grinds it.

    A 3+ player postflop DP can't be navigated on the crate's 2-seat tree; mirror the
    placeholder/timeout skip pattern so the per-DP dedup excludes it on the next pass.
    """
    entry = SolverCacheEntry(
        cluster_key=obs_row.get("cluster_key") or "",
        decision_id=obs_row.get("decision_id"),
        action_dist={},
        exploitability_pct=0.0,
        solver_version="multiway_hu_unsupported",
        spot_features={"solver_settings": {"range_narrowing": "multiway_hu_unsupported", "nav_ok": False}},
    )
    persist_solve(entry, _tsdb_conn=conn)


def _handle_nav_failed(obs_row: dict[str, Any], *, conn: Any) -> None:
    """Record a nav_failed marker for a member the binary could not navigate to.

    range_narrowing stays flop_rooted (it WAS a flop-rooted attempt that failed nav, not a
    new enum). The solver_version='nav_failed' marker dedups it out and is SQL-countable for
    a later fix pass — never silently dropped.
    """  # long-ok
    entry = SolverCacheEntry(
        cluster_key=obs_row.get("cluster_key") or "",
        decision_id=obs_row.get("decision_id"),
        action_dist={},
        exploitability_pct=0.0,
        solver_version="nav_failed",
        spot_features={"solver_settings": {"range_narrowing": "flop_rooted", "nav_ok": False}},
    )
    persist_solve(entry, _tsdb_conn=conn)


def _handle_solve_failed(obs_row: dict[str, Any], *, conn: Any) -> None:
    """Record a solve_failed marker for a member whose solve/parse raised deterministically.

    Without this, a spot that always raises NoStrategyError / SolverParseError has no
    solver_cache.decision_id row → the per-DP dedup re-fetches and re-grinds it every pass
    forever. The solver_version='solve_failed' marker dedups it out and is SQL-countable for
    a later fix pass — never silently dropped.
    """
    entry = SolverCacheEntry(
        cluster_key=obs_row.get("cluster_key") or "",
        decision_id=obs_row.get("decision_id"),
        action_dist={},
        exploitability_pct=0.0,
        solver_version="solve_failed",
        spot_features=None,
    )
    persist_solve(entry, _tsdb_conn=conn)


def _upsert_corpus_node(
    *,
    decision_id: str,
    cluster_key: str,
    street: str,
    pot: float,
    solved_stack: float,
    action_dist: dict[str, float],
    exploitability_pct: float,
    raw_embedding: list[float],
    milvus: Any,
    raise_on_failure: bool = False,
    added_at_override: int | None = None,
) -> bool:
    """Upsert one per-DP corpus node into Milvus. Returns False on skip/swallowed failure.

    Sizing fields use the -1 sentinels: solver action_dist keys are already canonical
    buckets, so snap_postflop_action passes them through and never reads the sizes.
    added_at_override lets backfill stamp the historical solve time; default = now.
    """
    hard_filter = _parse_cluster_key(cluster_key)
    dominant_action = max(action_dist, key=action_dist.__getitem__)
    spr_x100 = round(solved_stack / pot * 100) if pot else -1

    collection = (
        "preflop_decisions"
        if hard_filter.get("street_class", "").lower() == "preflop"
        else "postflop_decisions"
    )

    embedding = _normalize_for_corpus(raw_embedding, collection)
    if embedding is None:
        log.warning(
            "solver.queue_driver.inject_skip_no_embedding",
            decision_id=decision_id,
            cluster_key=cluster_key,
        )
        return False

    node_row = {
        "decision_id": decision_id,
        "embedding": embedding,
        "street_class": hard_filter.get("street_class", ""),
        "street": street,
        "pot_type": hard_filter.get("pot_type", ""),
        "hero_pos_rel": hard_filter.get("hero_pos_rel", ""),
        "n_players_active": int(hard_filter.get("n_players_active", 0)),
        "spr_x100": spr_x100,
        "hero_action_type": dominant_action,
        "hero_action_size_pot_frac": -1.0,
        "raise_ratio": -1.0,
        "hero_action_allin": False,
        "active": True,
        "confidence": 1.0,
        "gto_score": max(0.0, 1.0 - (exploitability_pct / 100.0)),
        "feature_spec_version": _FEATURE_SPEC_VERSION,
        "action_dist": json.dumps(action_dist),
        "added_at": (
            added_at_override if added_at_override is not None else int(datetime.now(tz=UTC).timestamp())
        ),
        "removed_at": 0,
    }

    try:
        milvus.upsert(collection_name=collection, data=[node_row], timeout=_MILVUS_RPC_TIMEOUT_S)
    except Exception as exc:
        if raise_on_failure:
            raise
        log.warning(
            "solver.queue_driver.milvus_upsert_failed",
            decision_id=decision_id,
            cluster_key=cluster_key,
            error=str(exc),
        )
        return False
    return True


def _solved_stack(solver_result: Any, full_context: dict[str, Any]) -> float:
    # When the memory gate caps the stack, file the node under its solved SPR, not
    # the deep SPR it no longer represents. Non-int/absent (==0) → requested stack.
    final_stack = getattr(solver_result, "final_effective_stack", 0)
    return (
        final_stack
        if isinstance(final_stack, int) and final_stack > 0
        else (full_context.get("effective_stack") or 0)
    )


def _inject(
    obs_row: dict,
    spot: SolverSpot,
    solver_result: Any,
    *,
    conn: Any,
    milvus: Any,
    palette_lookup: dict,
    _patch_engine: Any = None,
) -> None:
    """D-16: persist_solve(spot_features=full_context, decision_id) BEFORE per-DP Milvus upsert.

    Writes one Milvus row per decision point (NOT via PatchEngine — solver-distillation
    is corpus population, not a patch to the strategy_nodes override layer).
    Fails loud on empty action_line_full.
    """  # long-ok
    cfg_path = DEFAULT_CONFIG_PATH
    cfg = load_toml_config(cfg_path, AutoLoopConfig).solver_queue
    palette_lookup = palette_lookup or (build_range_lookup(PALETTE_DIR) if PALETTE_DIR.exists() else {})
    _, full_context = _build_spot(obs_row, palette_lookup, cfg)

    full_context["action_dist"] = solver_result.action_dist
    full_context["exploitability_pct"] = solver_result.exploitability_pct

    settings = full_context.get("solver_settings")
    if isinstance(settings, dict):
        settings["distortion"] = solver_result.distortion
        settings["applied_prune"] = solver_result.applied_prune
        settings["applied_n_sizes"] = solver_result.applied_n_sizes
        settings["final_effective_stack"] = solver_result.final_effective_stack
        settings["mem_estimate_mb"] = solver_result.mem_estimate_mb

    action_line_full = full_context.get("action_line_full")
    if not action_line_full:
        raise ValueError(
            f"action_line_full is None/empty for obs_id={obs_row['obs_id']!r} "
            f"cluster_key={obs_row['cluster_key']!r} — "
            "cannot persist solve context without full betting line (D-16 / T-15-07)"
        )

    decision_id = obs_row.get("decision_id") or obs_row["obs_id"]
    obs_embedding = obs_row.get("embedding") or []

    persist_solve(
        SolverCacheEntry(
            cluster_key=obs_row["cluster_key"],
            decision_id=decision_id,
            action_dist=solver_result.action_dist,
            exploitability_pct=solver_result.exploitability_pct,
            spot_features=full_context,
            solver_version="postflop-cli",
        ),
        _tsdb_conn=conn,
    )

    cluster_key = obs_row["cluster_key"]
    raw_embedding = list(obs_embedding) if not isinstance(obs_embedding, list) else obs_embedding
    _upsert_corpus_node(
        decision_id=decision_id,
        cluster_key=cluster_key,
        street=full_context.get("street") or "",
        pot=full_context.get("pot") or 0,
        solved_stack=_solved_stack(solver_result, full_context),
        action_dist=solver_result.action_dist,
        exploitability_pct=solver_result.exploitability_pct,
        raw_embedding=raw_embedding,
        milvus=milvus,
    )

    log.info(
        "solver.queue_driver.inject_complete",
        decision_id=decision_id,
        cluster_key=cluster_key,
        dominant_action=max(solver_result.action_dist, key=solver_result.action_dist.__getitem__),
    )


def _inject_node(
    member_obs: dict[str, Any],
    spot: SolverSpot,
    solver_result: SolverResult,
    full_context: dict[str, Any],
    *,
    conn: Any,
    milvus: Any,
    range_narrowing: str,
    nav_ok: bool,
) -> None:
    """Persist one navigated member node from a group solve (no re-solve).

    Takes a PRE-BUILT SolverResult + per-member full_context, stamps range_narrowing/nav_ok
    + gate provenance into solver_settings, persists the solver_cache row keyed by the
    member's decision_id, then upserts one per-member Milvus node (spr from the solved stack).
    """  # long-ok
    full_context["action_dist"] = solver_result.action_dist
    full_context["exploitability_pct"] = solver_result.exploitability_pct

    settings = full_context.get("solver_settings")
    if isinstance(settings, dict):
        settings["distortion"] = solver_result.distortion
        settings["applied_prune"] = solver_result.applied_prune
        settings["applied_n_sizes"] = solver_result.applied_n_sizes
        settings["final_effective_stack"] = solver_result.final_effective_stack
        settings["mem_estimate_mb"] = solver_result.mem_estimate_mb
        settings["range_narrowing"] = range_narrowing
        settings["nav_ok"] = nav_ok

    action_line_full = full_context.get("action_line_full")
    if not action_line_full:
        raise ValueError(
            f"action_line_full is None/empty for obs_id={member_obs['obs_id']!r} "
            f"cluster_key={member_obs['cluster_key']!r} — "
            "cannot persist solve context without full betting line (D-16 / T-15-07)"
        )

    decision_id = member_obs.get("decision_id") or member_obs["obs_id"]
    obs_embedding = member_obs.get("embedding") or []

    persist_solve(
        SolverCacheEntry(
            cluster_key=member_obs["cluster_key"],
            decision_id=decision_id,
            action_dist=solver_result.action_dist,
            exploitability_pct=solver_result.exploitability_pct,
            spot_features=full_context,
            solver_version="postflop-cli",
        ),
        _tsdb_conn=conn,
    )

    cluster_key = member_obs["cluster_key"]
    raw_embedding = list(obs_embedding) if not isinstance(obs_embedding, list) else obs_embedding
    _upsert_corpus_node(
        decision_id=decision_id,
        cluster_key=cluster_key,
        street=full_context.get("street") or "",
        pot=full_context.get("pot") or 0,
        solved_stack=_solved_stack(solver_result, full_context),
        action_dist=solver_result.action_dist,
        exploitability_pct=solver_result.exploitability_pct,
        raw_embedding=raw_embedding,
        milvus=milvus,
    )

    log.info(
        "solver.queue_driver.inject_node_complete",
        decision_id=decision_id,
        cluster_key=cluster_key,
        range_narrowing=range_narrowing,
        dominant_action=max(solver_result.action_dist, key=solver_result.action_dist.__getitem__),
    )
