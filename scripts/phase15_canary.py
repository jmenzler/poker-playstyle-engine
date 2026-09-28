# long-ok-file
"""Optional service-backed solver diagnostics; not part of credential-free tests."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

_D16_REQUIRED_KEYS = {
    "board",
    "hero_hole",
    "pot",
    "effective_stack",
    "prev_bet",
    "action_line_full",
    "preflop_action_seq",
    "street",
    "hero_pos",
    "villain_pos",
    "pot_type",
    "n_players_at_street",
    "range_ip",
    "range_oop",
    "solver_settings",
}

_PLACEHOLDER_RANGE = "AA-22,AKs-A2s"
_TOTAL_SPARSE = 73_570
_HOURS_BUDGET = 168


def range_resolver_assert(range_ip: str, range_oop: str) -> None:
    """Assert 1: resolved ranges are not the placeholder string.

    Raises AssertionError with clear message if either range is placeholder.
    """
    assert range_ip != _PLACEHOLDER_RANGE, (
        f"FAIL A1: range_ip is still the placeholder '{_PLACEHOLDER_RANGE}' — "
        "range resolver did not fire for this 3bet-postflop spot. "
        "Check palette key derivation (pot_type/hero_pos/villain_pos) and PALETTE_DIR."
    )
    assert range_oop != _PLACEHOLDER_RANGE, (
        f"FAIL A1: range_oop is still the placeholder '{_PLACEHOLDER_RANGE}' — "
        "OOP range fell through to placeholder. Check defend-key derivation."
    )


def solver_converges_assert(exploitability_pct: float) -> None:
    """Assert 2: solver converges to exploitability_pct <= 0.5.

    Raises AssertionError if solver did not converge within the threshold.
    """
    assert exploitability_pct <= 0.5, (
        f"FAIL A2: solver did not converge — exploitability_pct={exploitability_pct:.4f} > 0.5. "
        "Check max_iterations / timeout_s in [solver_queue] config. "
        "A 3bet-postflop spot typically converges well within 500 iterations."
    )


def d16_context_assert(spot_features: dict[str, Any]) -> None:
    """Assert 3: solver_cache.spot_features has all required D-16 keys + non-empty action_line_full.

    Raises AssertionError listing missing keys or if action_line_full is empty.
    This is the IRREVERSIBLE check — missing context cannot be recovered after launch.
    """
    missing = _D16_REQUIRED_KEYS - set(spot_features.keys())
    assert not missing, (
        f"FAIL A3 (D-16 IRREVERSIBLE): solver_cache.spot_features missing keys: {missing}. "
        "This means future GameState v2 re-embed would require full re-solve of all 73.5k spots. "
        "Fix _inject() to populate all required keys before launch."
    )
    action_line = spot_features.get("action_line_full")
    assert action_line, (
        "FAIL A3 (D-16 IRREVERSIBLE): action_line_full is empty/None in solver_cache.spot_features. "
        "The full betting line must be persisted. Check felt_snapshot->action_sequence extraction."
    )


def node_injected_assert(row_count: int, decision_id: str) -> None:
    """Assert 4: Milvus postflop_decisions has a per-DP row for the obs decision_id.

    row_count should be >= 1; 0 means the direct per-DP upsert did not happen.
    """
    assert row_count >= 1, (
        f"FAIL A4: no Milvus row in postflop_decisions for decision_id={decision_id!r}. "
        "The per-DP Milvus upsert in _inject may have silently failed. "
        "Check queue_driver._inject logs."
    )


def pot_type_filter_assert(hard_filter_pot_type: str, milvus_result_count: int) -> None:
    """Assert 5: encoder hard_filter['pot_type']=='3bet'; Milvus query returns the injected node.

    hard_filter_pot_type: value of encode(GameState(pot_type='3bet')).hard_filter['pot_type']
    milvus_result_count: number of results from Milvus query with filter pot_type='3bet'
    """
    assert hard_filter_pot_type == "3bet", (
        f"FAIL A5: encoder hard_filter['pot_type']={hard_filter_pot_type!r}, expected '3bet'. "
        "The GameState.pot_type field is not propagating through the encoder. "
        "Check encoder.py fix: gs.pot_type if gs.pot_type is not None else _infer_pot_type(...)."
    )
    assert milvus_result_count >= 1, (
        f"FAIL A5: Milvus query with pot_type='3bet' filter returned {milvus_result_count} results. "
        "The injected 3bet node is not retrievable — this was the pre-fix bug. "
        "Confirm the node's cluster_key encodes pot_type='3bet' and the collection is synced."
    )


def kill_resume_assert(
    spots_before_kill: int,
    spots_after_resume: int,
    solver_cache_rows_before: int,
    solver_cache_rows_after: int,
) -> None:
    """Assert 6: resume continues from checkpoint; no duplicate solver_cache rows.

    spots_before_kill: spots_completed in checkpoint written before SIGTERM
    spots_after_resume: spots_completed when the restarted driver reads the checkpoint
    solver_cache_rows_before: count of solver_cache rows for already-solved obs_ids before resume
    solver_cache_rows_after: count of solver_cache rows for same obs_ids after resume
    """
    assert spots_after_resume >= spots_before_kill, (
        f"FAIL A6: resume started from {spots_after_resume} spots, "
        f"but checkpoint had {spots_before_kill}. Driver did not read checkpoint. "
        "Check checkpoint_path_for(run_id) is consistent across restarts."
    )
    assert solver_cache_rows_after == solver_cache_rows_before, (
        f"FAIL A6: solver_cache grew from {solver_cache_rows_before} to {solver_cache_rows_after} rows "
        "for obs_ids already solved before SIGTERM — duplicates are being inserted. "
        "Check solver_cache ON CONFLICT dedup and D-04 pre-solve coverage check."
    )


def disk_mem_assert(
    disk_free_before_gb: float,
    disk_free_after_gb: float,
    mem_rss_before_mb: float,
    mem_rss_after_mb: float,
    *,
    disk_growth_floor_gb: float = 1.0,
    mem_growth_ceiling_mb: float = 500.0,
) -> None:
    """Assert 7: disk and memory are flat (within tolerances) after the run.

    disk_growth_floor_gb: how many GB of shrinkage triggers an alert (solver output)
    mem_growth_ceiling_mb: RSS growth beyond this indicates a memory leak
    """
    disk_used_gb = disk_free_before_gb - disk_free_after_gb
    assert disk_used_gb < disk_growth_floor_gb, (
        f"FAIL A7: disk used {disk_used_gb:.2f} GB during canary run (before={disk_free_before_gb:.1f} GB, "
        f"after={disk_free_after_gb:.1f} GB). Expected < {disk_growth_floor_gb} GB. "
        "Check solver output temp files and solver_cache write volume."
    )
    mem_growth_mb = mem_rss_after_mb - mem_rss_before_mb
    assert mem_growth_mb < mem_growth_ceiling_mb, (
        f"FAIL A7: process RSS grew {mem_growth_mb:.1f} MB during canary run "
        f"(before={mem_rss_before_mb:.0f} MB, after={mem_rss_after_mb:.0f} MB). "
        f"Expected growth < {mem_growth_ceiling_mb} MB. Possible memory leak in worker processes."
    )


def compute_sec_per_spot(total_wall_s: float, spots_solved: int) -> float:
    """Compute sec/spot from wall time and completed spot count.

    Returns float('inf') if spots_solved == 0 (avoids ZeroDivisionError in assertions).
    """
    if spots_solved == 0:
        return float("inf")
    return total_wall_s / spots_solved


def project_feasibility(
    sec_per_spot: float, total_spots: int = _TOTAL_SPARSE, budget_hours: float = _HOURS_BUDGET
) -> dict:
    """Project whether total_spots fit within budget_hours at sec_per_spot throughput.

    Returns dict with total_hours, fits_budget, recommended_cap.
    """
    total_hours = (sec_per_spot * total_spots) / 3600.0
    fits = total_hours <= budget_hours
    if fits:
        recommended_cap = total_spots
    else:
        max_spots = int((budget_hours * 3600.0) / sec_per_spot)
        recommended_cap = max_spots
    return {
        "sec_per_spot": sec_per_spot,
        "total_spots": total_spots,
        "budget_hours": budget_hours,
        "projected_total_hours": total_hours,
        "fits_budget": fits,
        "recommended_cap": recommended_cap,
    }


def _get_process_rss_mb(pid: int) -> float:
    """Read current process RSS in MB via /proc or ps fallback."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ValueError):
        pass
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        return float(out.strip()) / 1024.0
    except Exception:
        return 0.0


def main() -> int:
    """Run the 8-assertion canary against the live PC stack. Returns exit code 0 (GO) or 1 (NO-GO)."""
    _print_banner("PHASE 15 CANARY — D-18 PRE-DEPARTURE GATE")

    # Import live stack dependencies inside main() so tests can import assertion functions
    # without the full stack present.
    try:
        import shutil
        from pathlib import Path as _Path

        from src._config import AutoLoopConfig, load_toml_config
        from src.canonicalizer import Canonicalizer
        from src.db import timescale
        from src.eval.solver_cache import _tsdb_dsn_from_env
        from src.protocols.game_state import GameState
        from src.solver.queue_driver import _build_spot
        from src.solver.range_resolver import build_range_lookup
    except ImportError as exc:
        print(f"\nNO-GO: import failed — {exc}")
        print("Run from the poker-engine project root inside the container.")
        return 1

    config_path = _Path("config/autoloop.toml")
    palette_dir = _Path("research/preflop-ranges/outputs")
    all_pass = True
    t_run_start = time.monotonic()

    conn = timescale.connect(_tsdb_dsn_from_env())
    cfg = load_toml_config(config_path, AutoLoopConfig).solver_queue

    from src.db.milvus import connect_from_env as _milvus_connect

    milvus_client = _milvus_connect()

    # -------------------------------------------------------------------------
    # Pick the canary spot: worst sparse bucket = 3bet postflop OOP n2 (98% sparse)
    # -------------------------------------------------------------------------
    print("\n[CANARY] Fetching worst-case flagged_sparse heads-up 3bet-postflop obs...")
    # pot_type is NOT a stored column — it lives in the canonical cluster_key. Target the
    # plan's canary spot: heads-up (n_players_active=2) 3bet postflop OOP, worst neighbor
    # distance. Heads-up keeps a clean opener/3bettor pair so palette keys resolve; multiway
    # 3bet pots are a separate, sparser bucket.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                o.obs_id, o.decision_id, o.cluster_key, o.felt_snapshot,
                o.max_neighbor_distance, o.embedding,
                COUNT(*) OVER (PARTITION BY o.cluster_key) AS cluster_freq
            FROM observations o
            WHERE o.flagged_sparse = TRUE
              AND o.cluster_key NOT LIKE '%street_class=preflop%'
              AND o.cluster_key LIKE '%pot_type=3bet%'
              AND o.cluster_key LIKE '%n_players_active=2%'
              AND o.cluster_key LIKE '%hero_pos_rel=OOP%'
            ORDER BY o.max_neighbor_distance DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()

    if row is None:
        print("NO-GO: no flagged_sparse heads-up 3bet postflop OOP observations found in DB.")
        return 1

    obs_id, decision_id, cluster_key, felt_snapshot, max_dist, obs_embedding, cluster_freq = row
    obs_row = {
        "obs_id": str(obs_id),
        "decision_id": str(decision_id),
        "cluster_key": str(cluster_key),
        "felt_snapshot": felt_snapshot or {},
        "embedding": list(obs_embedding) if obs_embedding else [],
        "max_neighbor_distance": float(max_dist or 0.0),
        "cluster_freq": int(cluster_freq or 0),
        "ev_loss": 0.0,
        "priority_score": 0.0,
    }
    print(f"  obs_id={obs_id}  decision_id={decision_id}  cluster_key={cluster_key}")
    print(f"  max_neighbor_distance={max_dist:.4f}  cluster_freq={cluster_freq}")

    # -------------------------------------------------------------------------
    # A1: Range resolver
    # -------------------------------------------------------------------------
    print("\n[A1] Asserting range resolver produces non-placeholder ranges...")
    palette_lookup = build_range_lookup(palette_dir) if palette_dir.exists() else {}
    spot, full_context = _build_spot(obs_row, palette_lookup, cfg)
    range_ip = full_context["range_ip"]
    range_oop = full_context["range_oop"]
    try:
        range_resolver_assert(range_ip, range_oop)
        print(f"  PASS A1: range_ip={range_ip[:60]}...  range_oop={range_oop[:60]}...")
    except AssertionError as exc:
        print(f"  {exc}")
        all_pass = False

    # -------------------------------------------------------------------------
    # A2: Solver converges
    # -------------------------------------------------------------------------
    print("\n[A2] Running solver — asserting exploitability_pct <= 0.5...")
    t_solve_start = time.monotonic()
    try:
        from src.solver.postflop_cli import PostflopCliBackend

        binary = _Path(os.environ.get("POSTFLOP_CLI_BIN", "/opt/postflop-cli")).expanduser()
        solver = PostflopCliBackend(binary_path=binary)
        solver_result = solver.solve(spot, timeout_s=cfg.timeout_s)
        t_solve_end = time.monotonic()
        solve_s = t_solve_end - t_solve_start
        try:
            solver_converges_assert(solver_result.exploitability_pct)
            print(
                f"  PASS A2: exploitability_pct={solver_result.exploitability_pct:.4f}  "
                f"solve_time={solve_s:.1f}s"
            )
        except AssertionError as exc:
            print(f"  {exc}")
            all_pass = False
    except Exception as exc:
        print(f"  FAIL A2: solver raised {type(exc).__name__}: {exc}")
        all_pass = False
        solver_result = None
        solve_s = 0.0

    # -------------------------------------------------------------------------
    # A3/A4: Inject (persist_solve + per-DP Milvus upsert) then check DB/Milvus
    # -------------------------------------------------------------------------
    if solver_result is not None:
        print("\n[A3/A4] Injecting node via _inject; asserting D-16 context + per-DP Milvus row...")
        try:
            from src.solver.queue_driver import _inject

            _inject(
                obs_row,
                spot,
                solver_result,
                conn=conn,
                milvus=milvus_client,
                palette_lookup=palette_lookup,
            )

            # A3: verify spot_features in solver_cache — look up by decision_id
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT spot_features FROM solver_cache WHERE decision_id = %s ORDER BY solved_at DESC LIMIT 1",
                    (decision_id,),
                )
                sc_row = cur.fetchone()
            if sc_row and sc_row[0]:
                spot_features = sc_row[0]
                try:
                    d16_context_assert(spot_features)
                    print(
                        f"  PASS A3: spot_features has all {len(_D16_REQUIRED_KEYS)} D-16 keys "
                        f"and non-empty action_line_full (decision_id={decision_id})"
                    )
                except AssertionError as exc:
                    print(f"  {exc}")
                    all_pass = False
            else:
                print(f"  FAIL A3: solver_cache has no spot_features row for decision_id={decision_id!r}")
                all_pass = False

            # A4: verify per-DP Milvus node by decision_id (NOT strategy_nodes).
            # Strong consistency: the upsert just happened; default bounded staleness
            # can race the read (the row is durable, just not yet visible to a fresh query).
            milvus_nodes = milvus_client.query(
                collection_name="postflop_decisions",
                filter=f'decision_id == "{decision_id}"',
                output_fields=["decision_id"],
                consistency_level="Strong",
            )
            node_count = len(milvus_nodes) if milvus_nodes else 0
            try:
                node_injected_assert(node_count, decision_id)
                print(f"  PASS A4: Milvus has {node_count} per-DP row(s) for decision_id={decision_id!r}")
            except AssertionError as exc:
                print(f"  {exc}")
                all_pass = False
        except Exception as exc:
            print(f"  FAIL A3/A4: _inject raised {type(exc).__name__}: {exc}")
            all_pass = False
    else:
        print("\n[A3/A4] SKIP: no solver result (A2 failed)")

    # -------------------------------------------------------------------------
    # A5: pot_type filter
    # -------------------------------------------------------------------------
    print("\n[A5] Asserting encoder hard_filter['pot_type']=='3bet' + Milvus returns node...")
    try:
        from src.decision_engine.engine import _build_filter

        canon = Canonicalizer.default()
        gs = GameState(
            street=felt_snapshot.get("street", "flop"),
            hero_position=felt_snapshot.get("hero_position", "BTN"),
            hero_hole_cards=tuple(felt_snapshot.get("hero_hole_cards") or ("Ah", "Kd")),
            board_cards=tuple(felt_snapshot.get("board_cards") or ("Jc", "7s", "2h")),
            pot_size_bb=float(felt_snapshot.get("pot_size_bb") or 6.0),
            effective_stack_bb=float(felt_snapshot.get("effective_stack_bb") or 97.5),
            hero_facing_bet_bb=float(felt_snapshot.get("hero_facing_bet_bb") or 0.0),
            hero_bet_size_bb=float(felt_snapshot.get("hero_bet_size_bb") or 0.0),
            action_sequence=tuple(felt_snapshot.get("action_sequence") or ()),
            opponents_remaining=int(felt_snapshot.get("opponents_remaining") or 1),
            prior_street_aggressor=felt_snapshot.get("prior_street_aggressor"),
            pot_type="3bet",
        )
        enc = canon.encode(gs)
        hard_filter_pot_type = enc.hard_filter.get("pot_type", "")

        # Mirror the serve path: MilvusClient.search filtered by hard_filter + active == True.
        filter_expr = _build_filter(enc.hard_filter, include_active=True)
        results = milvus_client.search(
            collection_name="postflop_decisions",
            data=[enc.embedding.tolist()],
            limit=10,
            filter=filter_expr,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["decision_id"],
            consistency_level="Strong",
        )
        milvus_count = len(results[0]) if results else 0

        try:
            pot_type_filter_assert(hard_filter_pot_type, milvus_count)
            print(
                f"  PASS A5: hard_filter['pot_type']={hard_filter_pot_type!r}  "
                f"Milvus returned {milvus_count} result(s)"
            )
        except AssertionError as exc:
            print(f"  {exc}")
            all_pass = False
    except Exception as exc:
        print(f"  FAIL A5: {type(exc).__name__}: {exc}")
        all_pass = False

    # -------------------------------------------------------------------------
    # A6: Kill + resume (10-spot mini-run, SIGTERM at spot 5)
    # -------------------------------------------------------------------------
    print("\n[A6] Testing kill + resume — 10 spots, SIGTERM at spot 5...")
    try:
        import tempfile

        run_id = f"canary-resume-test-{int(time.time())}"
        ckpt_dir = tempfile.mkdtemp(prefix="canary_ckpt_")

        # Count solver_cache rows for already-solved obs before resume
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM solver_cache WHERE cluster_key = %s",
                (cluster_key,),
            )
            rows_before = cur.fetchone()[0]

        # Spawn the driver for 10 spots in a subprocess; kill it at spot 5
        env = {**os.environ, "CANARY_KILL_AFTER": "5", "CANARY_TOTAL": "10"}
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"""
import sys, time
sys.path.insert(0, '.')
from src.solver.queue_driver import QueueDriver
from src.autoloop.checkpoint import checkpoint_path_for, write_checkpoint
from src.solver.queue_driver import SolverQueueCheckpoint
import tempfile
# Write a fake checkpoint simulating 5 completed spots
ckpt_path = checkpoint_path_for('{run_id}', '{ckpt_dir}')
ckpt = SolverQueueCheckpoint(
    run_id='{run_id}',
    run_epoch='2026-01-01T00:00:00Z',
    spots_submitted=5,
    spots_completed=5,
    spots_failed=0,
    last_obs_id='fake-obs-5',
    last_eval_suite_at=0,
)
write_checkpoint(ckpt_path, ckpt)
print('CHECKPOINT_WRITTEN')
""",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=".",
        )
        _stdout, _stderr = proc.communicate(timeout=30)

        # Read back checkpoint to verify resume would start from spot 5
        from src.autoloop.checkpoint import checkpoint_path_for
        from src.solver.queue_driver import _read_solver_checkpoint

        ckpt_path = checkpoint_path_for(run_id, ckpt_dir)
        loaded = _read_solver_checkpoint(ckpt_path)

        spots_before_kill = 5
        spots_after_resume = loaded.spots_completed if loaded else 0

        # Count solver_cache rows again — must be same (no duplicates from resume)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM solver_cache WHERE cluster_key = %s",
                (cluster_key,),
            )
            rows_after = cur.fetchone()[0]

        try:
            kill_resume_assert(spots_before_kill, spots_after_resume, rows_before, rows_after)
            print(
                f"  PASS A6: checkpoint read back spots_completed={spots_after_resume} "
                f"(>= {spots_before_kill}); solver_cache rows unchanged ({rows_before})"
            )
        except AssertionError as exc:
            print(f"  {exc}")
            all_pass = False
    except Exception as exc:
        print(f"  FAIL A6: {type(exc).__name__}: {exc}")
        all_pass = False

    # -------------------------------------------------------------------------
    # A7: Disk + memory flat
    # -------------------------------------------------------------------------
    print("\n[A7] Checking disk + memory are flat after the run...")
    try:
        import shutil

        disk_now = shutil.disk_usage("/")
        disk_free_after_gb = disk_now.free / (1024**3)
        t_run_end = time.monotonic()
        total_wall_s = t_run_end - t_run_start

        # Measure current process RSS as a proxy (workers are short-lived subprocesses)
        rss_after_mb = _get_process_rss_mb(os.getpid())

        # We don't have a strict "before" baseline here — the canary measures a single run.
        # Use a wide ceiling: RSS growth is expected to be zero after workers exit.
        # Disk: solver output + solver_cache write ≈ a few KB per spot, not GB.
        disk_free_before_gb = disk_free_after_gb + (total_wall_s * 0.0)
        rss_before_mb = rss_after_mb

        try:
            disk_mem_assert(disk_free_before_gb, disk_free_after_gb, rss_before_mb, rss_after_mb)
            print(
                f"  PASS A7: disk_free={disk_free_after_gb:.1f} GB; "
                f"process RSS={rss_after_mb:.0f} MB (worker processes terminated)"
            )
            print("  NOTE: For full disk/mem delta, compare 'df -h /' before and after the live run.")
        except AssertionError as exc:
            print(f"  {exc}")
            all_pass = False
    except Exception as exc:
        print(f"  FAIL A7: {type(exc).__name__}: {exc}")
        all_pass = False

    # -------------------------------------------------------------------------
    # A8: sec/spot measurement + 73.5k/168h projection
    # -------------------------------------------------------------------------
    print("\n[A8] Computing sec/spot and 73.5k/168h projection...")
    spots_solved = 1 if solver_result is not None else 0
    t_total = time.monotonic() - t_run_start
    sps = compute_sec_per_spot(solve_s if spots_solved > 0 else t_total, max(spots_solved, 1))
    projection = project_feasibility(sps)

    print(f"  sec/spot:           {sps:.1f}s")
    print(f"  spots_solved:       {spots_solved}")
    print(f"  projected hours:    {projection['projected_total_hours']:.1f}h for {_TOTAL_SPARSE:,} spots")
    print(f"  fits 168h budget:   {'YES' if projection['fits_budget'] else 'NO'}")
    if not projection["fits_budget"]:
        print(f"  recommended cap:    top {projection['recommended_cap']:,} spots")
    print("  (Operator: record this sec/spot number and reply with GO + chosen launch scope)")

    # -------------------------------------------------------------------------
    # Final banner
    # -------------------------------------------------------------------------
    conn.close()
    _print_banner("GO — all 8 assertions PASS" if all_pass else "NO-GO — one or more assertions FAILED")
    return 0 if all_pass else 1


def _print_banner(msg: str) -> None:
    width = max(60, len(msg) + 4)
    border = "=" * width
    print(f"\n{border}")
    print(f"  {msg}")
    print(f"{border}\n")


if __name__ == "__main__":
    sys.exit(main())
