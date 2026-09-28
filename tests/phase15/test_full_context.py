"""Tests for D-16 full solve context — all required keys, resolve-before-inject."""  # rot-allow

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

D16_REQUIRED_KEYS = {
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


def _make_obs_row(with_action_line: bool = True) -> dict:
    felt = {
        "action_sequence": ["BTN:raise_half_pot", "BB:call", "BTN:bet_half_pot"] if with_action_line else [],
        "board_cards": ["Ah", "7c", "2d"],
        "effective_stack_bb": 97.5,
        "hero_bet_size_bb": 0.0,
        "hero_facing_bet_bb": 0.0,
        "hero_hole_cards": ["As", "Kd"],
        "hero_position": "BTN",
        "opponents_remaining": 1,
        "pot_size_bb": 10.0,
        "prior_street_aggressor": "BTN",
        "street": "flop",
        "pot_type": "3bet",
    }
    return {
        "obs_id": "obs-test-001",
        "cluster_key": "street=flop|pot_type=3bet|hero_position=BTN|n_players_active=2",
        "max_neighbor_distance": 0.85,
        "felt_snapshot": felt,
        "embedding": [0.1] * 80,
        "cluster_freq": 42,
        "ev_loss": 0.8,
        "priority_score": 42 * 0.8 * 0.85,
    }


def _make_cfg():
    from src._config import SolverQueueConfig

    return SolverQueueConfig(
        n_workers=11,
        target_exploitability_pct=0.5,
        max_iterations=10,
        timeout_s=30.0,
        bet_sizes=("33%", "66%", "e", "a"),
    )


def test_required_keys():
    from src.solver.queue_driver import _build_spot

    obs_row = _make_obs_row()
    cfg = _make_cfg()
    _, full_context = _build_spot(obs_row, {}, cfg)
    missing = D16_REQUIRED_KEYS - set(full_context.keys())
    assert not missing, f"full_context missing D-16 required keys: {missing}"


def test_persist_writes_context():
    """persist_solve is called with non-None spot_features dict."""
    from src.solver.queue_driver import _inject

    obs_row = _make_obs_row()

    mock_solver_result = MagicMock()
    mock_solver_result.action_dist = {"check": 0.5, "bet_33": 0.5}
    mock_solver_result.exploitability_pct = 0.45

    captured_entries = []

    def mock_persist(entry, *, _tsdb_conn=None):
        captured_entries.append(entry)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = ("node-uuid-1", [0.1] * 128)
    mock_conn.cursor.return_value = mock_cursor

    from src.solver.postflop_cli import SolverSpot

    spot = SolverSpot(
        pot=1000,
        effective_stack=9750,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
        max_iterations=10,
        target_exploitability_pct=0.5,
    )

    with patch("src.solver.queue_driver.persist_solve", side_effect=mock_persist):
        _inject(
            obs_row,
            spot,
            mock_solver_result,
            conn=mock_conn,
            milvus=MagicMock(),
            palette_lookup={},
        )

    assert len(captured_entries) == 1
    entry = captured_entries[0]
    assert entry.spot_features is not None
    assert isinstance(entry.spot_features, dict)
    assert "action_line_full" in entry.spot_features


def test_two_writes():
    """persist_solve is called BEFORE milvus.upsert (D-16 write order, solver path)."""
    from src.solver.queue_driver import _inject

    obs_row = _make_obs_row()
    call_order: list[str] = []

    def mock_persist(entry, *, _tsdb_conn=None):
        call_order.append("persist_solve")

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    mock_milvus = MagicMock()
    mock_milvus.upsert.side_effect = lambda **kwargs: call_order.append("milvus.upsert")

    from src.solver.postflop_cli import SolverSpot

    spot = SolverSpot(
        pot=1000,
        effective_stack=9750,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"check": 1.0}
    mock_result.exploitability_pct = 0.3

    with patch("src.solver.queue_driver.persist_solve", side_effect=mock_persist):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
        )

    assert "persist_solve" in call_order, f"persist_solve not called. Order: {call_order}"
    assert "milvus.upsert" in call_order, f"milvus.upsert not called. Order: {call_order}"
    assert call_order.index("persist_solve") < call_order.index("milvus.upsert"), (
        f"Expected persist_solve BEFORE milvus.upsert, got: {call_order}"
    )


def test_worker_error_isolation():
    """Worker raising SolverParseError propagates from _solve_one_spot (non-placeholder spot)."""
    from src._errors import SolverParseError

    obs_row = {
        "obs_id": "obs-err-001",
        "decision_id": "dp-err-001",
        "cluster_key": "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        "max_neighbor_distance": 0.8,
        "felt_snapshot": {
            "action_sequence": ["BB:check"],
            "board_cards": ["Ah", "7c", "2d"],
            "effective_stack_bb": 95.0,
            "hero_bet_size_bb": 0.0,
            "hero_facing_bet_bb": 0.0,
            "hero_hole_cards": ["As", "Kd"],
            "hero_position": "BTN",
            "opponents_remaining": 1,
            "pot_size_bb": 10.0,
            "prior_street_aggressor": "BTN",
            "street": "flop",
        },
        "embedding": [0.1] * 80,
        "cluster_freq": 10,
        "ev_loss": 0.5,
        "priority_score": 4.0,
    }
    cfg = _make_cfg()

    mock_solver = MagicMock()

    real_range = "AA,KK,QQ"
    with patch("src.solver.queue_driver._build_spot") as mock_build:
        mock_spot = MagicMock()
        mock_spot.range_ip = real_range
        mock_spot.range_oop = real_range
        mock_build.return_value = (
            mock_spot,
            {"range_ip": real_range, "range_oop": real_range, "action_line_full": ["BB:check"]},
        )
        mock_solver.solve.side_effect = SolverParseError("test solver crash")

        with pytest.raises(SolverParseError):
            from src.solver.queue_driver import _solve_one_spot

            _solve_one_spot(obs_row, mock_solver, cfg)


def test_ranges_non_placeholder_in_context():
    """range_ip/range_oop are palette-resolved for a 3bet spot using the real felt_snapshot shape.

    Real shape: BB 3bets BTN; hero=BTN (IP postflop). range_ip=BTN's defend range, range_oop=BB's 3bet range.
    Palette keys: bettor (BB) -> 'BB/3bet_vs_BTN'; opener (BTN) -> 'BTN/defend_3bet_vs_BB'.
    """
    from src.solver.queue_driver import _build_spot

    obs_row = {
        "obs_id": "obs-3bet-range-test",
        "cluster_key": "hero_pos_rel=IP|n_players_active=2|pot_type=3bet|street_class=postflop",
        "max_neighbor_distance": 0.85,
        "felt_snapshot": {
            "action_sequence": [
                "UTG:fold",
                "MP:fold",
                "CO:fold",
                "BTN:open_2.5",
                "SB:fold",
                "BB:3bet_9",
                "BTN:call",
                "BB:check",
                "BTN:bet_half_pot",
            ],
            "board_cards": ["Ah", "7c", "2d"],
            "effective_stack_bb": 91.0,
            "hero_bet_size_bb": 0.0,
            "hero_facing_bet_bb": 0.0,
            "hero_hole_cards": ["As", "Kd"],
            "hero_position": "BTN",
            "opponents_remaining": 1,
            "pot_size_bb": 18.0,
            "prior_street_aggressor": "BB",
            "street": "flop",
        },
        "cluster_freq": 42,
        "ev_loss": 0.8,
        "priority_score": 42 * 0.8 * 0.85,
    }
    cfg = _make_cfg()

    palette_lookup = {
        "BB/3bet_vs_BTN": {"AA": 100.0, "KK": 90.0, "QQ": 80.0},
        "BTN/defend_3bet_vs_BB": {"AKs": 75.0, "AQs": 60.0, "KQs": 50.0},
    }

    _, full_context = _build_spot(obs_row, palette_lookup, cfg)
    range_ip = full_context["range_ip"]
    range_oop = full_context["range_oop"]
    assert range_ip != "AA-22,AKs-A2s", (
        f"range_ip is still the placeholder for a 3bet spot with palette available: {range_ip!r}"
    )
    assert range_oop != "AA-22,AKs-A2s", f"range_oop is still the placeholder: {range_oop!r}"
    assert "AKs" in range_ip, f"range_ip (BTN defending = IP player) must contain AKs: {range_ip!r}"
    assert "AA" in range_oop, f"range_oop (BB 3betting = OOP player) must contain AA: {range_oop!r}"


def test_fail_loud_on_empty_action_line():
    """_inject raises ValueError if action_line_full is empty (D-16 / T-15-07 fail-loud guard)."""
    from src.solver.queue_driver import _inject

    obs_row = _make_obs_row(with_action_line=False)

    from src.solver.postflop_cli import SolverSpot

    spot = SolverSpot(
        pot=1000,
        effective_stack=9750,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"check": 1.0}
    mock_result.exploitability_pct = 0.3

    mock_conn = MagicMock()

    with pytest.raises(ValueError, match="action_line_full"):
        with patch("src.solver.queue_driver.persist_solve"):
            _inject(
                obs_row,
                spot,
                mock_result,
                conn=mock_conn,
                milvus=None,
                palette_lookup={},
            )
