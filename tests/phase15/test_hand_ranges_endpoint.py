"""Endpoint tests for hand_ranges orchestration + router."""  # long-ok-file

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Shared helpers  # rot-allow


def _mock_conn(rows: list[tuple] | None = None) -> MagicMock:
    """Return a mock psycopg connection."""
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = rows or []
    cur.description = [("col",)]
    conn.cursor.return_value = cur
    txn = MagicMock()
    txn.__enter__ = MagicMock(return_value=txn)
    txn.__exit__ = MagicMock(return_value=False)
    conn.transaction.return_value = txn
    return conn


def _make_harvest_result(
    n_combos: int = 3,
    n_villain: int = 4,
    actions: list[str] | None = None,
    nav_ok: bool = True,
) -> dict[str, Any]:
    """Return a fake raw harvest result dict (as 17-01 Rust emits)."""
    if actions is None:
        actions = ["CHECK", "BET 50"]
    n_a = len(actions)
    combos = [f"combo{i}" for i in range(n_combos)]
    villain_combos = [f"vcombo{i}" for i in range(n_villain)]
    return {
        "nav_ok": nav_ok,
        "actions": actions,
        "root_player": 1,
        "pot_at_node": 100,
        "hero_grid": {
            "combos": combos,
            "weights": [0.8] * n_combos,
            "strategy": [0.6] * (n_a * n_combos),
            "ev_detail": [5.0] * (n_a * n_combos),
            "equity": [0.55] * n_combos,
        },
        "villain_grid": {
            "combos": villain_combos,
            "weights": [0.7] * n_villain,
            "equity": [0.45] * n_villain,
        },
    }


def _make_mock_solver(results: list[dict[str, Any]]) -> MagicMock:
    """Return a mock solver whose solve_harvest_raw returns (top, results)."""
    solver = MagicMock()
    solver.is_available.return_value = True
    solver.solve_harvest_raw.return_value = (
        {"exploitability_pct": 0.8},
        results,
    )
    return solver


def _make_hand_rows(n_dps: int = 1, action_taken: str = "CHECK") -> list[dict[str, Any]]:
    """Return minimal hand_rows for a HU flop hand."""
    rows = []
    for i in range(n_dps):
        decision_id = f"hand1_dp{i}"
        rows.append(
            {
                "hand_id": "hand1",
                "decision_id": decision_id,
                "action_taken": action_taken,
                "felt_snapshot": {
                    "street": "flop",
                    "board_cards": ["Ah", "Kd", "2c"],
                    "hero_position": "BTN",
                    "effective_stack_bb": 100.0,
                    "pot_size_bb": 6.0,
                    "action_sequence": ["BB:check"],
                    "opponents_remaining": 1,
                    "hero_pos": "BTN",
                },
            }
        )
    return rows


# Task 1 TDD tests — orchestration  # rot-allow


class TestSolveAndPersistOnMiss:
    def test_solve_and_persist_calls_solver_on_miss(self) -> None:
        """On cache miss, solve_and_persist_hand_ranges calls solver.solve_harvest_raw."""
        from src.study.hand_ranges import solve_and_persist_hand_ranges

        res = _make_harvest_result()
        solver = _make_mock_solver([res])

        conn = _mock_conn()

        with (
            patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[]),
            patch("src.study.hand_ranges.get_replay_by_hand", return_value=_make_hand_rows()),
            patch("src.study.hand_ranges.persist_harvest_range") as mock_persist,
        ):
            result = solve_and_persist_hand_ranges("hand1", conn=conn, solver=solver)

        solver.solve_harvest_raw.assert_called_once()
        assert result["status"] == "solved"
        assert mock_persist.call_count == 2  # hero seat + villain seat

    def test_per_block_invariant_violation_raises(self) -> None:
        """A result with mismatched strategy length raises before persisting."""
        from src.study.hand_ranges import solve_and_persist_hand_ranges

        bad_result = _make_harvest_result()
        # Break the invariant: strategy has wrong length
        bad_result["hero_grid"]["strategy"] = [0.5]  # should be n_actions * n_combos

        solver = _make_mock_solver([bad_result])
        conn = _mock_conn()

        with (
            patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[]),
            patch("src.study.hand_ranges.get_replay_by_hand", return_value=_make_hand_rows()),
            patch("src.study.hand_ranges.persist_harvest_range") as mock_persist,
        ):
            with pytest.raises((ValueError, AssertionError)):
                solve_and_persist_hand_ranges("hand1", conn=conn, solver=solver)

        mock_persist.assert_not_called()

    def test_hero_action_top_level_source(self) -> None:
        """hero_action is derived from top-level action_taken on the hand row."""
        from src.study.hand_ranges import solve_and_persist_hand_ranges

        res = _make_harvest_result(actions=["CHECK", "BET 50"])
        solver = _make_mock_solver([res])
        conn = _mock_conn()
        hand_rows = _make_hand_rows(action_taken="CHECK")

        persisted_payloads: list[dict] = []

        def capture_persist(row: Any, **kwargs: Any) -> None:
            persisted_payloads.append(row.payload)

        with (
            patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[]),
            patch("src.study.hand_ranges.get_replay_by_hand", return_value=hand_rows),
            patch("src.study.hand_ranges.persist_harvest_range", side_effect=capture_persist),
        ):
            solve_and_persist_hand_ranges("hand1", conn=conn, solver=solver)

        # Find the hero-seat payload (seat 1 = IP = hero_seat from root_player=1)
        hero_payloads = [p for p in persisted_payloads if "strategy" in p]
        assert len(hero_payloads) >= 1
        hero_payload = hero_payloads[0]
        assert hero_payload["hero_action"] == "CHECK"

    def test_hero_action_unmappable_yields_none(self) -> None:
        """An unmappable action_taken yields hero_action == None."""
        from src.study.hand_ranges import solve_and_persist_hand_ranges

        res = _make_harvest_result(actions=["CHECK", "BET 50"])
        solver = _make_mock_solver([res])
        conn = _mock_conn()
        hand_rows = _make_hand_rows(action_taken="WEIRD_UNKNOWN_TOKEN")

        persisted_payloads: list[dict] = []

        def capture_persist(row: Any, **kwargs: Any) -> None:
            persisted_payloads.append(row.payload)

        with (
            patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[]),
            patch("src.study.hand_ranges.get_replay_by_hand", return_value=hand_rows),
            patch("src.study.hand_ranges.persist_harvest_range", side_effect=capture_persist),
        ):
            solve_and_persist_hand_ranges("hand1", conn=conn, solver=solver)

        hero_payloads = [p for p in persisted_payloads if "strategy" in p]
        assert len(hero_payloads) >= 1
        assert hero_payloads[0]["hero_action"] is None

    def test_cache_hit_skips_solver(self) -> None:
        """When rows already exist, solve is skipped (cache hit)."""
        from src.study.hand_ranges import solve_and_persist_hand_ranges
        from src.study.harvest_ranges import HarvestRangeRow

        existing_row = HarvestRangeRow(
            hand_id="hand1",
            decision_id="hand1_dp0",
            seat=0,
            payload={"narrowing": "ok"},
        )
        solver = _make_mock_solver([])
        conn = _mock_conn()

        with patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[existing_row]):
            result = solve_and_persist_hand_ranges("hand1", conn=conn, solver=solver)

        solver.solve_harvest_raw.assert_not_called()
        assert result["status"] == "cached"

    def test_cache_hit_force_reruns(self) -> None:
        """force=True bypasses the cache hit and re-solves."""
        from src.study.hand_ranges import solve_and_persist_hand_ranges
        from src.study.harvest_ranges import HarvestRangeRow

        existing_row = HarvestRangeRow(
            hand_id="hand1",
            decision_id="hand1_dp0",
            seat=0,
            payload={"narrowing": "ok"},
        )
        res = _make_harvest_result()
        solver = _make_mock_solver([res])
        conn = _mock_conn()

        with (
            patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[existing_row]),
            patch("src.study.hand_ranges.get_replay_by_hand", return_value=_make_hand_rows()),
            patch("src.study.hand_ranges.persist_harvest_range"),
        ):
            result = solve_and_persist_hand_ranges("hand1", conn=conn, solver=solver, force=True)

        solver.solve_harvest_raw.assert_called_once()
        assert result["status"] == "solved"


class TestLoadHandRangesContract:
    def _make_harvest_row(
        self,
        hand_id: str,
        decision_id: str,
        seat: int,
        payload: dict[str, Any],
    ) -> Any:
        from src.study.harvest_ranges import HarvestRangeRow

        return HarvestRangeRow(
            hand_id=hand_id,
            decision_id=decision_id,
            seat=seat,
            payload=payload,
        )

    def test_contract_assembly_shape(self) -> None:
        """load_hand_ranges_contract returns frozen shape with hero-seat fields."""
        from src.study.hand_ranges import load_hand_ranges_contract

        combos_hero = ["AhKh", "AsKs", "AdKd"]
        combos_villain = ["QdQc", "QhQc"]
        n_a = 2
        n_h = len(combos_hero)
        n_v = len(combos_villain)

        hero_payload = {
            "narrowing": "ok",
            "street": "flop",
            "hero_seat": 1,
            "combos": combos_hero,
            "weights": [0.8] * n_h,
            "equity": [0.55] * n_h,
            "strategy": [0.6] * (n_a * n_h),
            "ev_detail": [5.0] * (n_a * n_h),
            "actions": ["CHECK", "BET 50"],
            "hero_action": "CHECK",
        }
        villain_payload = {
            "narrowing": "ok",
            "street": "flop",
            "hero_seat": 1,
            "combos": combos_villain,
            "weights": [0.7] * n_v,
            "equity": [0.45] * n_v,
        }

        rows = [
            self._make_harvest_row("hand1", "hand1_dp0", 1, hero_payload),
            self._make_harvest_row("hand1", "hand1_dp0", 0, villain_payload),
        ]

        conn = _mock_conn()
        with patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=rows):
            contract = load_hand_ranges_contract("hand1", conn=conn)

        assert len(contract) == 1
        entry = contract[0]

        assert entry["decision_id"] == "hand1_dp0"
        assert entry["street"] == "flop"
        assert entry["hero_seat"] == 1
        assert entry["narrowing"] == "ok"
        assert entry["multiway"] is False

        # IP = hero seat (seat 1) → carries strategy/ev_detail/actions/hero_action
        ip = entry["ip"]
        assert ip["combos"] == combos_hero
        assert len(ip["weights"]) == n_h
        assert len(ip["equity"]) == n_h
        assert len(ip["strategy"]) == n_a * n_h
        assert len(ip["ev_detail"]) == n_a * n_h
        assert ip["actions"] == ["CHECK", "BET 50"]
        assert ip["hero_action"] == "CHECK"

        # OOP = villain seat (seat 0) → only combos/weights/equity
        oop = entry["oop"]
        assert oop["combos"] == combos_villain
        assert len(oop["weights"]) == n_v
        assert len(oop["equity"]) == n_v
        assert "strategy" not in oop
        assert "ev_detail" not in oop

    def test_multiway_derived_not_stored(self) -> None:
        """multiway is DERIVED from narrowing == 'multiway_hu_unsupported'; no oop/ip."""
        from src.study.hand_ranges import load_hand_ranges_contract

        mw_payload = {
            "narrowing": "multiway_hu_unsupported",
            "street": "turn",
            "hero_seat": 0,
        }
        rows = [
            self._make_harvest_row("hand1", "hand1_dp1", 0, mw_payload),
            self._make_harvest_row("hand1", "hand1_dp1", 1, mw_payload),
        ]

        conn = _mock_conn()
        with patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=rows):
            contract = load_hand_ranges_contract("hand1", conn=conn)

        assert len(contract) == 1
        entry = contract[0]
        assert entry["narrowing"] == "multiway_hu_unsupported"
        assert entry["multiway"] is True
        assert "oop" not in entry or entry.get("oop") is None
        assert "ip" not in entry or entry.get("ip") is None

    def test_nav_failed_no_range_arrays(self) -> None:
        """nav_failed narrowing → no oop/ip in the contract entry."""
        from src.study.hand_ranges import load_hand_ranges_contract

        failed_payload = {
            "narrowing": "nav_failed",
            "street": "river",
            "hero_seat": 1,
        }
        rows = [
            self._make_harvest_row("hand1", "hand1_dp2", 0, failed_payload),
            self._make_harvest_row("hand1", "hand1_dp2", 1, failed_payload),
        ]

        conn = _mock_conn()
        with patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=rows):
            contract = load_hand_ranges_contract("hand1", conn=conn)

        assert len(contract) == 1
        entry = contract[0]
        assert entry["narrowing"] == "nav_failed"
        assert entry["multiway"] is False
        assert "oop" not in entry or entry.get("oop") is None

    def test_empty_hand_returns_empty_list(self) -> None:
        """No rows → empty contract list."""
        from src.study.hand_ranges import load_hand_ranges_contract

        conn = _mock_conn()
        with patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[]):
            contract = load_hand_ranges_contract("hand1", conn=conn)

        assert contract == []


# Task 2 endpoint tests — FastAPI router  # rot-allow


def _make_app_client():
    """Build a TestClient for the FastAPI app with tsdb override."""
    from fastapi.testclient import TestClient

    from src.api.deps import get_tsdb
    from src.api.main import app

    conn = _mock_conn()
    app.dependency_overrides[get_tsdb] = lambda: conn
    return TestClient(app, raise_server_exceptions=True), conn


class TestHandRangesRouter:
    def setup_method(self) -> None:
        from src.api.deps import get_tsdb
        from src.api.main import app

        self._original_overrides = dict(app.dependency_overrides)
        self._conn = _mock_conn()
        app.dependency_overrides[get_tsdb] = lambda: self._conn

    def teardown_method(self) -> None:
        from src.api.main import app

        app.dependency_overrides.clear()
        app.dependency_overrides.update(self._original_overrides)

    def test_get_ranges_unsolved_returns_empty(self) -> None:
        """GET /api/hands/by-hand/{hand_id}/ranges returns [] when nothing solved."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[]):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/hands/by-hand/hand1/ranges")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_ranges_returns_frozen_shape(self) -> None:
        """After seeding rows, GET ranges returns the frozen dual-seat contract."""
        from fastapi.testclient import TestClient

        from src.api.main import app
        from src.study.harvest_ranges import HarvestRangeRow

        combos_h = ["AhKh", "AsKs"]
        combos_v = ["QdQc"]
        hero_payload = {
            "narrowing": "ok",
            "street": "flop",
            "hero_seat": 1,
            "combos": combos_h,
            "weights": [0.8, 0.7],
            "equity": [0.55, 0.52],
            "strategy": [0.6, 0.4, 0.3, 0.7],
            "ev_detail": [5.0, 4.0, 3.5, 3.0],
            "actions": ["CHECK", "BET 50"],
            "hero_action": "CHECK",
        }
        villain_payload = {
            "narrowing": "ok",
            "street": "flop",
            "hero_seat": 1,
            "combos": combos_v,
            "weights": [0.9],
            "equity": [0.45],
        }
        rows = [
            HarvestRangeRow(hand_id="hand1", decision_id="hand1_dp0", seat=1, payload=hero_payload),
            HarvestRangeRow(hand_id="hand1", decision_id="hand1_dp0", seat=0, payload=villain_payload),
        ]

        with patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=rows):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/hands/by-hand/hand1/ranges")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["decision_id"] == "hand1_dp0"
        assert entry["narrowing"] == "ok"
        assert entry["multiway"] is False
        assert entry["ip"]["combos"] == combos_h
        assert "strategy" in entry["ip"]
        assert entry["oop"]["combos"] == combos_v
        assert "strategy" not in entry["oop"]

    def test_multiway_sentinel_in_contract(self) -> None:
        """Multiway sentinel row surfaces narrowing='multiway_hu_unsupported' AND multiway=True."""
        from fastapi.testclient import TestClient

        from src.api.main import app
        from src.study.harvest_ranges import HarvestRangeRow

        mw_payload = {"narrowing": "multiway_hu_unsupported", "street": "turn", "hero_seat": 0}
        rows = [
            HarvestRangeRow(hand_id="h1", decision_id="h1_dp0", seat=0, payload=mw_payload),
            HarvestRangeRow(hand_id="h1", decision_id="h1_dp0", seat=1, payload=mw_payload),
        ]

        with patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=rows):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/hands/by-hand/h1/ranges")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["narrowing"] == "multiway_hu_unsupported"
        assert data[0]["multiway"] is True
        assert data[0].get("oop") is None
        assert data[0].get("ip") is None

    def test_bad_hand_id_returns_400(self) -> None:
        """hand_id with invalid characters returns 400."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/hands/by-hand/hand%20id%20with%20spaces/ranges")
        assert resp.status_code == 400

    def test_post_solve_ranges_returns_job_id(self) -> None:
        """POST solve-ranges returns job_id + status immediately."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with (
            patch("src.study.hand_ranges.load_harvest_ranges_by_hand", return_value=[]),
            patch("src.api.hand_ranges._run_solve"),
        ):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/hands/by-hand/hand1/solve-ranges")

        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert "status" in body
