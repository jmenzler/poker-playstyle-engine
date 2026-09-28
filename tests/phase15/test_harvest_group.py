# rot-allow-file
# long-ok-file
"""solve_harvest backend + flop-spot group queue dispatch.

Backend (`harvest_backend`): one solve_harvest call -> N per-line results; deep
pot_at_node snap; nav_ok=false -> None. Group (`group`): fetch -> group-by-flop-spot
-> one harvest call -> N injects, multiway skip+flag, nav_failed marker, decision_id dedup.
The binary subprocess and the solver backend are mocked (real PC integration is Wave 3).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Task 1 — PostflopCliBackend.solve_harvest (harvest_backend)


def _completed(stdout: dict) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["postflop-cli"], returncode=0, stdout=json.dumps(stdout), stderr=""
    )


def _harvest_result(
    *, action_dist: list, pot_at_node: int, nav_ok: bool, to_call: int = 0, villain_bet_total: int = 0
) -> dict:
    return {
        "action_dist": action_dist,
        "ev_detail": [],
        "combos": [],
        "actions": [p[0] for p in action_dist],
        "root_player": 1,
        "final_effective_stack": 1800,
        "pot_at_node": pot_at_node,
        "to_call": to_call,
        "villain_bet_total": villain_bet_total,
        "villain_live_combos": 7,
        "nav_ok": nav_ok,
    }


def _harvest_output(results: list[dict]) -> dict:
    return {
        "exploitability_pct": 0.8,
        "time_ms": 1234,
        "distortion": "none",
        "applied_prune": 0.0,
        "applied_n_sizes": 3,
        "final_effective_stack": 1800,
        "mem_estimate_mb": 512,
        "results": results,
    }


def test_harvest_backend_snaps_against_pot_at_node_and_handles_nav_failure() -> None:
    """A deep BET 900 with pot_at_node 1800 snaps to bet_50; nav_ok=false -> None.

    spot.pot is 200 (flop-start). 900/200 = 4.5 would raise SolverParseError; the snap
    MUST use the result's pot_at_node 1800 (900/1800 = 0.5 -> bet_50) instead.
    """
    from src.solver.postflop_cli import PostflopCliBackend, SolverResult, SolverSpot

    spot = SolverSpot(
        pot=200,
        effective_stack=18000,
        board=["Ah", "7c", "2d"],
        range_ip="AKo,AKs",
        range_oop="22+",
    )
    nav_lines: list[dict[str, object]] = [
        {
            "steps": [{"seat": 0, "kind": "check", "frac": None}],
            "hero_player": 1,
            "turn_card": "Td",
            "river_card": None,
        },
        {
            "steps": [{"seat": 0, "kind": "bet", "frac": 0.5}],
            "hero_player": 1,
            "turn_card": "Td",
            "river_card": "3s",
        },
    ]
    out = _harvest_output(
        [
            _harvest_result(action_dist=[["BET 900", 0.7], ["CHECK", 0.3]], pot_at_node=1800, nav_ok=True),
            _harvest_result(action_dist=[], pot_at_node=0, nav_ok=False),
        ]
    )

    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["input"] = kwargs["input"]
        return _completed(out)

    with patch("src.solver.postflop_cli.subprocess.run", side_effect=fake_run):
        results = backend.solve_harvest(spot, nav_lines)

    assert len(results) == 2
    assert isinstance(results[0], SolverResult)
    assert results[1] is None
    assert "bet_50" in results[0].action_dist, (
        f"BET 900 / pot_at_node 1800 must snap to bet_50 (not 900/200=4.5 -> error). "
        f"Got {results[0].action_dist}"
    )

    payload = json.loads(captured["input"])
    assert payload["mode"] == "solve_harvest"
    assert isinstance(payload["nav_lines"], list)
    assert len(payload["nav_lines"]) == 2


def test_harvest_all_zero_action_dist_is_degenerate_nav_not_error() -> None:
    """Hero's range carries no weight at the node -> None (nav_failed path), not
    a SolverParseError that loops the whole group as spot_failed."""
    from src.solver.postflop_cli import raw_harvest_to_solver_result

    raw = _harvest_result(action_dist=[["CHECK", 0.0], ["BET 100", 0.0]], pot_at_node=200, nav_ok=True)
    assert raw_harvest_to_solver_result(raw, _harvest_output([raw])) is None


def test_harvest_backend_maps_raise_at_facing_bet_node() -> None:
    """A facing-bet node: RAISE TO 1200 over villain 600 (to_call 600) → increment
    ratio 1.0 → raise_min. Requires the binary's to_call/villain_bet_total fields."""
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    spot = SolverSpot(pot=200, effective_stack=18000, board=["Ah", "7c", "2d"], range_ip="AA", range_oop="KK")
    nav_lines: list[dict[str, object]] = [
        {
            "steps": [{"seat": 0, "kind": "bet", "frac": 0.5}],
            "hero_player": 1,
            "turn_card": None,
            "river_card": None,
        },
    ]
    out = _harvest_output(
        [
            _harvest_result(
                action_dist=[["RAISE 1200", 0.4], ["CALL", 0.4], ["FOLD", 0.2]],
                pot_at_node=1400,
                nav_ok=True,
                to_call=600,
                villain_bet_total=600,
            ),
        ]
    )
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=_completed(out)):
        results = backend.solve_harvest(spot, nav_lines)

    assert results[0] is not None
    assert results[0].action_dist == {"raise_min": 0.4, "call": 0.4, "fold": 0.2}


def test_harvest_backend_returns_one_result_per_nav_line() -> None:
    """results length == nav_lines length, including None for nav failures."""
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    spot = SolverSpot(pot=200, effective_stack=18000, board=["Ah", "7c", "2d"], range_ip="AA", range_oop="KK")
    nav_lines: list[dict[str, object]] = [
        {"steps": [], "hero_player": 0, "turn_card": None, "river_card": None},
        {"steps": [], "hero_player": 0, "turn_card": None, "river_card": None},
        {"steps": [], "hero_player": 0, "turn_card": None, "river_card": None},
    ]
    out = _harvest_output(
        [
            _harvest_result(action_dist=[["CHECK", 1.0]], pot_at_node=200, nav_ok=True),
            _harvest_result(action_dist=[], pot_at_node=0, nav_ok=False),
            _harvest_result(action_dist=[["BET 100", 1.0]], pot_at_node=200, nav_ok=True),
        ]
    )
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=_completed(out)):
        results = backend.solve_harvest(spot, nav_lines)

    assert len(results) == 3
    assert results[1] is None
    assert results[0] is not None and "check" in results[0].action_dist
    assert results[2] is not None and "bet_50" in results[2].action_dist


def test_harvest_backend_nonzero_exit_raises() -> None:
    """A non-zero binary exit raises NoStrategyError (mirrors solve())."""
    from src._errors import NoStrategyError
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    spot = SolverSpot(pot=200, effective_stack=18000, board=["Ah", "7c", "2d"], range_ip="AA", range_oop="KK")
    crashed = subprocess.CompletedProcess(args=["postflop-cli"], returncode=1, stdout="", stderr="boom")
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=crashed):
        with pytest.raises(NoStrategyError):
            backend.solve_harvest(
                spot, [{"steps": [], "hero_player": 0, "turn_card": None, "river_card": None}]
            )


# Task 2 — flop-spot group queue dispatch (group)

_SRP_FLOP_FELT: dict = {
    "street": "flop",
    "board_cards": ["Jc", "7s", "2h"],
    "pot_size_bb": 5.5,
    "hero_position": "BTN",
    "action_sequence": ["BTN:open_2.5", "BB:call", "BB:check"],
    "hero_hole_cards": ["Ah", "Kd"],
    "hero_bet_size_bb": 0.0,
    "effective_stack_bb": 97.5,
    "hero_facing_bet_bb": 0.0,
    "opponents_remaining": 1,
}

_SRP_TURN_FELT: dict = {
    "street": "turn",
    "board_cards": ["Jc", "7s", "2h", "Td"],
    "pot_size_bb": 11.0,
    "hero_position": "BTN",
    "action_sequence": ["BTN:open_2.5", "BB:call", "BB:check", "BTN:bet_half_pot", "BB:call", "BB:check"],
    "hero_hole_cards": ["Ah", "Kd"],
    "hero_bet_size_bb": 0.0,
    "effective_stack_bb": 92.0,
    "hero_facing_bet_bb": 0.0,
    "opponents_remaining": 1,
}

_MULTIWAY_FELT: dict = {
    "street": "flop",
    "board_cards": ["Jc", "7s", "2h"],
    "pot_size_bb": 9.0,
    "hero_position": "BTN",
    "action_sequence": ["CO:open_2.5", "BTN:call", "BB:call", "BB:check"],
    "hero_hole_cards": ["Ah", "Kd"],
    "hero_bet_size_bb": 0.0,
    "effective_stack_bb": 95.0,
    "hero_facing_bet_bb": 0.0,
    "opponents_remaining": 2,
}


def _member(obs_id: str, decision_id: str, felt: dict, hand_id: str = "hand-1") -> dict:
    return {
        "obs_id": obs_id,
        "decision_id": decision_id,
        "cluster_key": "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        "embedding": [0.2] * 80,
        "max_neighbor_distance": 0.8,
        "felt_snapshot": felt,
        "cluster_freq": 30,
        "ev_loss": 0.7,
        "priority_score": 30 * 0.7 * 0.8,
        "hand_id": hand_id,
    }


def _real_result(action_dist: dict) -> object:
    from src.solver.postflop_cli import SolverResult

    return SolverResult(
        action_dist=action_dist,
        exploitability_pct=0.5,
        solve_time_ms=100,
        distortion="none",
        final_effective_stack=9200,
    )


def _raw_top() -> dict:
    return {
        "exploitability_pct": 0.5,
        "time_ms": 100,
        "distortion": "none",
        "applied_prune": 0.0,
        "applied_n_sizes": 0,
        "final_effective_stack": 9200,
        "mem_estimate_mb": 0,
    }


def _raw_ok(action_dist_pairs: list) -> dict:
    n_a = len(action_dist_pairs)
    return {
        "nav_ok": True,
        "action_dist": action_dist_pairs,
        "pot_at_node": 200,
        "actions": [p[0] for p in action_dist_pairs],
        "final_effective_stack": 9200,
        "hero_grid": {
            "combos": ["AsAh"],
            "weights": [1.0],
            "equity": [0.5],
            "strategy": [1.0 / n_a] * n_a,
            "ev_detail": [0.0] * n_a,
        },
        "villain_grid": {"combos": ["KsKh"], "weights": [1.0], "equity": [0.5]},
    }


_RAW_NAV_FAILED = {"nav_ok": False, "action_dist": [], "pot_at_node": 0}


def _cfg():
    from pathlib import Path as _P

    from src._config import AutoLoopConfig, load_toml_config

    return load_toml_config(_P("config/autoloop.toml"), AutoLoopConfig).solver_queue


def test_group_by_flop_spot_collapses_hu_members_and_splits_multiway() -> None:
    """Flop DP + turn DP sharing a flop-spot land in one group; multiway goes to the skip list."""
    from src.solver.queue_driver import _group_by_flop_spot

    flop = _member("obs-flop", "dp-flop", _SRP_FLOP_FELT, hand_id="hand-1")
    turn = _member("obs-turn", "dp-turn", _SRP_TURN_FELT, hand_id="hand-1")
    multiway = _member("obs-mw", "dp-mw", _MULTIWAY_FELT, hand_id="hand-2")

    groups, multiway_members = _group_by_flop_spot([flop, turn, multiway])

    assert len(groups) == 1, (
        f"flop + turn on the same flop-spot must collapse to ONE group, got {len(groups)}"
    )
    members = next(iter(groups.values()))
    assert {m["decision_id"] for m in members} == {"dp-flop", "dp-turn"}
    assert len(multiway_members) == 1
    assert multiway_members[0]["decision_id"] == "dp-mw"


def test_solve_group_dispatches_one_harvest_call() -> None:
    """A 2-member group fires exactly ONE solve_harvest with 2 nav_lines."""
    from src.solver.queue_driver import _solve_group

    flop = _member("obs-flop", "dp-flop", _SRP_FLOP_FELT)
    turn = _member("obs-turn", "dp-turn", _SRP_TURN_FELT)

    fake_solver = MagicMock()
    fake_solver.solve_harvest_raw.return_value = (
        _raw_top(),
        [_raw_ok([["CHECK", 1.0]]), _raw_ok([["BET 100", 1.0]])],
    )

    out = _solve_group([flop, turn], fake_solver, _cfg())

    assert fake_solver.solve_harvest_raw.call_count == 1, "exactly one harvest call per group"
    call = fake_solver.solve_harvest_raw.call_args
    nav_lines = call.args[1]
    assert len(nav_lines) == 2, "one nav_line per member"
    assert len(out["members"]) == 2
    assert len(out["results"]) == 2
    assert len(out["raw_results"]) == 2, "raw harvest dicts ride along for range persistence"
    assert len(out["nav_lines"]) == 2
    spot = call.args[0]
    assert len(spot.board) == 3, "the shared spot must be flop-rooted (3-card board)"


def _mock_conn() -> MagicMock:
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


def test_inject_node_records_range_narrowing_provenance() -> None:
    """A turn member persists range_narrowing=flop_rooted; a flop member persists none."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _build_spot, _inject_node

    cfg = _cfg()
    flop = _member("obs-flop", "dp-flop", _SRP_FLOP_FELT)
    turn = _member("obs-turn", "dp-turn", _SRP_TURN_FELT)
    spot = SolverSpot(pot=550, effective_stack=9750, board=["Jc", "7s", "2h"], range_ip="AA", range_oop="KK")

    persisted: list = []

    def cap_persist(entry, *, _tsdb_conn=None):
        persisted.append(entry)

    for member, narrowing in ((flop, "none"), (turn, "flop_rooted")):
        _, ctx = _build_spot(member, {}, cfg)
        mock_milvus = MagicMock()
        with patch("src.solver.queue_driver.persist_solve", cap_persist):
            _inject_node(
                member,
                spot,
                _real_result({"check": 1.0}),
                ctx,
                conn=_mock_conn(),
                milvus=mock_milvus,
                range_narrowing=narrowing,
                nav_ok=True,
            )
        mock_milvus.upsert.assert_called_once()

    assert len(persisted) == 2
    by_dp = {e.decision_id: e for e in persisted}
    assert by_dp["dp-turn"].spot_features["solver_settings"]["range_narrowing"] == "flop_rooted"
    assert by_dp["dp-turn"].spot_features["solver_settings"]["nav_ok"] is True
    assert by_dp["dp-flop"].spot_features["solver_settings"]["range_narrowing"] == "none"


def test_handle_multiway_skip_flags_unsupported() -> None:
    """Multiway skip persists solver_version=multiway_hu_unsupported + range_narrowing flag."""
    from src.solver.queue_driver import _handle_multiway_skip

    persisted: list = []

    def cap_persist(entry, *, _tsdb_conn=None):
        persisted.append(entry)

    member = _member("obs-mw", "dp-mw", _MULTIWAY_FELT)
    with patch("src.solver.queue_driver.persist_solve", cap_persist):
        _handle_multiway_skip(member, conn=_mock_conn())

    assert len(persisted) == 1
    entry = persisted[0]
    assert entry.solver_version == "multiway_hu_unsupported"
    assert entry.decision_id == "dp-mw"
    assert entry.spot_features["solver_settings"]["range_narrowing"] == "multiway_hu_unsupported"
    assert entry.spot_features["solver_settings"]["nav_ok"] is False


def test_handle_nav_failed_records_marker_no_milvus() -> None:
    """nav_failed persists solver_version=nav_failed, range_narrowing=flop_rooted, nav_ok=False."""
    from src.solver.queue_driver import _handle_nav_failed

    persisted: list = []

    def cap_persist(entry, *, _tsdb_conn=None):
        persisted.append(entry)

    member = _member("obs-nav", "dp-nav", _SRP_TURN_FELT)
    with patch("src.solver.queue_driver.persist_solve", cap_persist):
        _handle_nav_failed(member, conn=_mock_conn())

    assert len(persisted) == 1
    entry = persisted[0]
    assert entry.solver_version == "nav_failed"
    assert entry.decision_id == "dp-nav"
    assert entry.spot_features["solver_settings"]["range_narrowing"] == "flop_rooted"
    assert entry.spot_features["solver_settings"]["nav_ok"] is False


def test_handle_solve_failed_records_marker() -> None:
    """A deterministically-failing solve persists solver_version=solve_failed so the per-DP
    dedup excludes it next pass (otherwise it is re-fetched and re-solved forever)."""
    from src.solver.queue_driver import _handle_solve_failed

    persisted: list = []

    def cap_persist(entry, *, _tsdb_conn=None):
        persisted.append(entry)

    member = _member("obs-fail", "dp-fail", _SRP_TURN_FELT)
    with patch("src.solver.queue_driver.persist_solve", cap_persist):
        _handle_solve_failed(member, conn=_mock_conn())

    assert len(persisted) == 1
    entry = persisted[0]
    assert entry.solver_version == "solve_failed"
    assert entry.decision_id == "dp-fail"
    assert entry.action_dist == {}


def test_priority_sql_dedup_at_decision_id_unchanged() -> None:
    """Grouping must NOT change the per-decision_id dedup clause in _PRIORITY_SQL."""
    from src.solver.queue_driver import _PRIORITY_SQL

    assert "sc.decision_id = o.decision_id" in _PRIORITY_SQL
    assert "o.hand_id" in _PRIORITY_SQL


def test_run_group_flow_one_harvest_two_injects() -> None:
    """End-to-end run() group flow: a 2-member group -> ONE harvest, TWO injects; a nav
    failure -> nav_failed marker + spots_failed; a multiway member -> skip+flag."""
    from src.solver.queue_driver import QueueDriver

    flop = _member("obs-flop", "dp-flop", _SRP_FLOP_FELT, hand_id="hand-1")
    turn = _member("obs-turn", "dp-turn", _SRP_TURN_FELT, hand_id="hand-1")
    multiway = _member("obs-mw", "dp-mw", _MULTIWAY_FELT, hand_id="hand-2")

    fake_solver = MagicMock()
    fake_solver.solve_harvest_raw.return_value = (
        _raw_top(),
        [_raw_ok([["CHECK", 1.0]]), _raw_ok([["BET 100", 1.0]])],
    )

    driver = QueueDriver.__new__(QueueDriver)
    from pathlib import Path as _P

    from src._config import AutoLoopConfig, load_toml_config

    driver._cfg = load_toml_config(_P("config/autoloop.toml"), AutoLoopConfig).solver_queue
    driver._run_id = "test-group-run"
    driver._stop_requested = False
    driver._palette_lookup = {}

    batches = [[flop, turn, multiway], []]

    def fake_fetch(conn, *, limit):
        return batches.pop(0) if batches else []

    inject_calls: list = []
    nav_failed_calls: list = []
    multiway_calls: list = []
    harvest_rows: list = []

    def cap_inject(member, spot, result, ctx, *, conn, milvus, range_narrowing, nav_ok):
        inject_calls.append((member["decision_id"], range_narrowing, nav_ok))

    def cap_nav_failed(obs_row, *, conn):
        nav_failed_calls.append(obs_row["decision_id"])

    def cap_multiway(obs_row, *, conn):
        multiway_calls.append(obs_row["decision_id"])

    def cap_harvest(row, *, _tsdb_conn=None):
        harvest_rows.append((row.decision_id, row.seat))

    with (
        patch.object(QueueDriver, "_fetch_batch", side_effect=fake_fetch),
        patch.object(QueueDriver, "_ensure_retention_policy"),
        patch.object(QueueDriver, "_install_signal_handlers"),
        patch("src.solver.queue_driver._inject_node", cap_inject),
        patch("src.solver.queue_driver._handle_nav_failed", cap_nav_failed),
        patch("src.solver.queue_driver._handle_multiway_skip", cap_multiway),
        patch("src.solver.queue_driver.persist_harvest_range", cap_harvest),
        patch("src.solver.queue_driver.write_checkpoint"),
        patch("src.solver.queue_driver._read_solver_checkpoint", return_value=None),
        patch("src.solver.queue_driver.checkpoint_path_for", return_value=Path("/tmp/ckpt.json")),
        patch("src.solver.queue_driver.ProcessPoolExecutor") as mock_pool,
    ):
        # Run groups serially in-process so the fake solver + patched injectors are exercised.
        from src.solver.queue_driver import _solve_group

        class _SerialPool:
            def __init__(self, *a, **k):
                self._futs = []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, members, solver, cfg):
                fut = MagicMock()
                fut.result.return_value = _solve_group(members, solver, cfg)
                self._futs.append((fut, members))
                return fut

        pool_inst = _SerialPool()
        mock_pool.return_value = pool_inst

        def fake_as_completed(futs):
            return list(futs.keys())

        with patch("src.solver.queue_driver.as_completed", fake_as_completed):
            summary = driver.run(
                _tsdb_conn=_mock_conn(),
                _milvus=MagicMock(),
                _solver=fake_solver,
                install_signals=False,
            )

    assert fake_solver.solve_harvest_raw.call_count == 1, "one harvest call for the 2-member group"
    assert len(inject_calls) == 2, f"two member injects, got {inject_calls}"
    narrowing_by_dp = {dp: rn for dp, rn, _ in inject_calls}
    assert narrowing_by_dp["dp-flop"] == "none"
    assert narrowing_by_dp["dp-turn"] == "flop_rooted"
    assert multiway_calls == ["dp-mw"], "multiway member skipped+flagged"
    assert sorted(harvest_rows) == [
        ("dp-flop", 0),
        ("dp-flop", 1),
        ("dp-turn", 0),
        ("dp-turn", 1),
    ], "both seats' range grids persist per nav_ok member"
    assert summary["spots_completed"] == 2
    assert summary["spots_failed"] == 1  # the multiway skip


def test_run_group_flow_nav_failure_routes_to_handle_nav_failed() -> None:
    """A None result for a member routes to _handle_nav_failed (spots_failed++), the other
    member still injects; the nav-failed member gets NO inject (NO Milvus upsert)."""
    from src.solver.queue_driver import QueueDriver

    flop = _member("obs-flop", "dp-flop", _SRP_FLOP_FELT, hand_id="hand-1")
    turn = _member("obs-turn", "dp-turn", _SRP_TURN_FELT, hand_id="hand-1")

    fake_solver = MagicMock()
    # turn member fails navigation -> nav_ok=false
    fake_solver.solve_harvest_raw.return_value = (_raw_top(), [_raw_ok([["CHECK", 1.0]]), _RAW_NAV_FAILED])

    driver = QueueDriver.__new__(QueueDriver)
    from pathlib import Path as _P

    from src._config import AutoLoopConfig, load_toml_config

    driver._cfg = load_toml_config(_P("config/autoloop.toml"), AutoLoopConfig).solver_queue
    driver._run_id = "test-nav-fail-run"
    driver._stop_requested = False
    driver._palette_lookup = {}

    batches = [[flop, turn], []]

    def fake_fetch(conn, *, limit):
        return batches.pop(0) if batches else []

    inject_calls: list = []
    nav_failed_calls: list = []

    def cap_inject(member, spot, result, ctx, *, conn, milvus, range_narrowing, nav_ok):
        inject_calls.append(member["decision_id"])

    def cap_nav_failed(obs_row, *, conn):
        nav_failed_calls.append(obs_row["decision_id"])

    with (
        patch.object(QueueDriver, "_fetch_batch", side_effect=fake_fetch),
        patch.object(QueueDriver, "_ensure_retention_policy"),
        patch("src.solver.queue_driver._inject_node", cap_inject),
        patch("src.solver.queue_driver._handle_nav_failed", cap_nav_failed),
        patch("src.solver.queue_driver.persist_harvest_range"),
        patch("src.solver.queue_driver.write_checkpoint"),
        patch("src.solver.queue_driver._read_solver_checkpoint", return_value=None),
        patch("src.solver.queue_driver.checkpoint_path_for", return_value=Path("/tmp/ckpt.json")),
        patch("src.solver.queue_driver.ProcessPoolExecutor") as mock_pool,
    ):
        from src.solver.queue_driver import _solve_group

        class _SerialPool:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, members, solver, cfg):
                fut = MagicMock()
                fut.result.return_value = _solve_group(members, solver, cfg)
                return fut

        mock_pool.return_value = _SerialPool()

        with patch("src.solver.queue_driver.as_completed", lambda futs: list(futs.keys())):
            summary = driver.run(
                _tsdb_conn=_mock_conn(),
                _milvus=MagicMock(),
                _solver=fake_solver,
                install_signals=False,
            )

    assert inject_calls == ["dp-flop"], "only the nav_ok member injects"
    assert nav_failed_calls == ["dp-turn"], "the nav-failed member routes to _handle_nav_failed"
    assert summary["spots_completed"] == 1
    assert summary["spots_failed"] == 1


def test_run_group_solve_parse_error_persists_skip_marker_per_member() -> None:
    """A group whose solve raises SolverParseError persists a solve_failed marker for EVERY
    member (so the per-DP dedup excludes them next pass — no infinite re-grind) and injects
    none of them."""
    from src._errors import SolverParseError
    from src.solver.queue_driver import QueueDriver

    flop = _member("obs-flop", "dp-flop", _SRP_FLOP_FELT, hand_id="hand-1")
    turn = _member("obs-turn", "dp-turn", _SRP_TURN_FELT, hand_id="hand-1")

    fake_solver = MagicMock()
    fake_solver.solve_harvest_raw.side_effect = SolverParseError("unmappable action label")

    driver = QueueDriver.__new__(QueueDriver)
    from pathlib import Path as _P

    from src._config import AutoLoopConfig, load_toml_config

    driver._cfg = load_toml_config(_P("config/autoloop.toml"), AutoLoopConfig).solver_queue
    driver._run_id = "test-solve-fail-run"
    driver._stop_requested = False
    driver._palette_lookup = {}

    batches = [[flop, turn], []]

    def fake_fetch(conn, *, limit):
        return batches.pop(0) if batches else []

    inject_calls: list = []
    persisted: list = []

    def cap_inject(member, spot, result, ctx, *, conn, milvus, range_narrowing, nav_ok):
        inject_calls.append(member["decision_id"])

    def cap_persist(entry, *, _tsdb_conn=None):
        persisted.append(entry)

    with (
        patch.object(QueueDriver, "_fetch_batch", side_effect=fake_fetch),
        patch.object(QueueDriver, "_ensure_retention_policy"),
        patch("src.solver.queue_driver._inject_node", cap_inject),
        patch("src.solver.queue_driver.persist_solve", cap_persist),
        patch("src.solver.queue_driver.persist_harvest_range"),
        patch("src.solver.queue_driver.write_checkpoint"),
        patch("src.solver.queue_driver._read_solver_checkpoint", return_value=None),
        patch("src.solver.queue_driver.checkpoint_path_for", return_value=Path("/tmp/ckpt.json")),
        patch("src.solver.queue_driver.ProcessPoolExecutor") as mock_pool,
    ):
        from src.solver.queue_driver import _solve_group

        class _SerialPool:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, members, solver, cfg):
                fut = MagicMock()
                fut.result.side_effect = lambda: _solve_group(members, solver, cfg)
                return fut

        mock_pool.return_value = _SerialPool()

        with patch("src.solver.queue_driver.as_completed", lambda futs: list(futs.keys())):
            summary = driver.run(
                _tsdb_conn=_mock_conn(),
                _milvus=MagicMock(),
                _solver=fake_solver,
                install_signals=False,
            )

    assert inject_calls == [], "a parse-error group injects nothing"
    skip_markers = {e.decision_id: e for e in persisted if e.solver_version == "solve_failed"}
    assert set(skip_markers) == {"dp-flop", "dp-turn"}, "every member gets a solve_failed marker"
    assert summary["spots_completed"] == 0
    assert summary["spots_failed"] == 2
