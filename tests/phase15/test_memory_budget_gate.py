"""Adaptive memory-budget gate — Python wiring.

SolverResult parse + back-compat, payload forwarding, per-spot budget split, and
the _inject provenance stash + capped-stack SPR relabeling. (Rust ladder: cargo tests.)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.solver.postflop_cli import (
    PostflopCliBackend,
    SolverResult,
    SolverSpot,
    _spot_to_payload,
)

# --- SolverResult: provenance fields + back-compat defaults ---


def test_solver_result_gate_defaults() -> None:
    """Omitting gate fields → faithful defaults (none/0)."""
    r = SolverResult(action_dist={"check": 1.0}, exploitability_pct=0.2, solve_time_ms=10)
    assert r.distortion == "none"
    assert r.applied_prune == 0.0
    assert r.applied_n_sizes == 0
    assert r.final_effective_stack == 0
    assert r.mem_estimate_mb == 0


@pytest.fixture
def fake_spot() -> SolverSpot:
    return SolverSpot(
        pot=100,
        effective_stack=9700,
        board=["Ah", "7c", "2d"],
        range_ip="AKo,AKs",
        range_oop="22+",
    )


def _completed(stdout: dict) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["postflop-cli"], returncode=0, stdout=json.dumps(stdout), stderr=""
    )


def test_solve_parses_gate_fields(fake_spot: SolverSpot) -> None:
    """Binary emits gate provenance → SolverResult carries it."""
    out = {
        "exploitability_pct": 1.0,
        "time_ms": 500,
        "actions_root": ["CHECK"],
        "aggregate_freq_root": [1.0],
        "actions_ip_after_check": [],
        "aggregate_freq_ip_after_check": [],
        "distortion": "heavy",
        "applied_prune": 0.15,
        "applied_n_sizes": 2,
        "final_effective_stack": 2000,
        "mem_estimate_mb": 8123,
    }
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=_completed(out)):
        r = backend.solve(fake_spot)
    assert r.distortion == "heavy"
    assert r.applied_prune == pytest.approx(0.15)
    assert r.applied_n_sizes == 2
    assert r.final_effective_stack == 2000
    assert r.mem_estimate_mb == 8123


def test_solve_backcompat_no_gate_fields(fake_spot: SolverSpot) -> None:
    """Pre-rebuild binary omits gate fields → SolverResult defaults, no crash."""
    out = {
        "exploitability_pct": 0.5,
        "time_ms": 100,
        "actions_root": ["CHECK"],
        "aggregate_freq_root": [1.0],
        "actions_ip_after_check": [],
        "aggregate_freq_ip_after_check": [],
    }
    backend = PostflopCliBackend(binary_path=Path("/fake/postflop-cli"))
    with patch("src.solver.postflop_cli.subprocess.run", return_value=_completed(out)):
        r = backend.solve(fake_spot)
    assert r.distortion == "none"
    assert r.final_effective_stack == 0


# --- Payload forwarding ---


def test_payload_includes_budget_when_set() -> None:
    spot = SolverSpot(
        pot=100,
        effective_stack=9700,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
        memory_budget_mb=4500,
    )
    assert _spot_to_payload(spot)["memory_budget_mb"] == 4500


def test_payload_omits_budget_when_none() -> None:
    spot = SolverSpot(
        pot=100,
        effective_stack=9700,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    assert "memory_budget_mb" not in _spot_to_payload(spot)


# --- _build_spot: per-spot budget = total pool // n_workers ---


def _make_obs_row() -> dict:
    return {
        "obs_id": "obs-budget-001",
        "cluster_key": "street=flop|pot_type=srp|hero_position=BTN|n_players_active=2",
        "felt_snapshot": {
            "action_sequence": ["BTN:raise_half_pot", "BB:call", "BB:check"],
            "board_cards": ["Ah", "7c", "2d"],
            "effective_stack_bb": 97.5,
            "hero_facing_bet_bb": 0.0,
            "hero_hole_cards": ["As", "Kd"],
            "hero_position": "BTN",
            "opponents_remaining": 1,
            "pot_size_bb": 6.0,
            "street": "flop",
            "pot_type": "srp",
        },
        "embedding": [0.1] * 80,
    }


def _cfg(*, memory_budget_mb: int, n_workers: int):
    from src._config import SolverQueueConfig

    return SolverQueueConfig(
        n_workers=n_workers,
        memory_budget_mb=memory_budget_mb,
        max_iterations=10,
        timeout_s=30.0,
    )


def test_build_spot_splits_budget_across_workers() -> None:
    from src.solver.queue_driver import _build_spot

    spot, _ = _build_spot(_make_obs_row(), {}, _cfg(memory_budget_mb=9000, n_workers=3))
    assert spot.memory_budget_mb == 3000


def test_build_spot_zero_budget_disables_gate() -> None:
    from src.solver.queue_driver import _build_spot

    spot, _ = _build_spot(_make_obs_row(), {}, _cfg(memory_budget_mb=0, n_workers=3))
    assert spot.memory_budget_mb is None


# --- _inject: provenance stash + capped-stack SPR relabeling ---


def _inject_with(result: SolverResult):
    from unittest.mock import MagicMock

    from src.solver.queue_driver import _inject

    obs_row = {
        "obs_id": "obs-inject-budget",
        "decision_id": "dp-budget-001",
        "cluster_key": "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        "embedding": [0.1] * 80,
        "felt_snapshot": {
            "action_sequence": ["BB:check"],
            "board_cards": ["Jh", "7c", "2d"],
            "effective_stack_bb": 95.0,
            "hero_facing_bet_bb": 0.0,
            "hero_hole_cards": ["Ac", "Kh"],
            "hero_position": "BTN",
            "opponents_remaining": 1,
            "pot_size_bb": 8.0,
            "street": "flop",
            "pot_type": "srp",
        },
    }
    spot = SolverSpot(pot=800, effective_stack=9500, board=["Jh", "7c", "2d"], range_ip="AA", range_oop="AA")
    mock_milvus = MagicMock()
    with patch("src.solver.queue_driver.persist_solve") as persist:
        _inject(obs_row, spot, result, conn=MagicMock(), milvus=mock_milvus, palette_lookup={})
    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    entry = persist.call_args.args[0]
    return row, entry


def _result(*, final_effective_stack: int, distortion: str) -> SolverResult:
    return SolverResult(
        action_dist={"check": 1.0},
        exploitability_pct=0.2,
        solve_time_ms=10,
        distortion=distortion,
        applied_prune=0.15,
        applied_n_sizes=2,
        final_effective_stack=final_effective_stack,
        mem_estimate_mb=8000,
    )


def test_inject_spr_uses_capped_stack() -> None:
    """A 20bb-capped solve (2000 chips) over an 800 pot → spr_x100 == 250, not 1187."""
    row, _ = _inject_with(_result(final_effective_stack=2000, distortion="heavy"))
    assert row["spr_x100"] == 250


def test_inject_spr_uncapped_uses_requested_stack() -> None:
    """No cap (final == requested 9500) → spr from requested stack."""
    row, _ = _inject_with(_result(final_effective_stack=9500, distortion="none"))
    assert row["spr_x100"] == round(9500 / 800 * 100)


def test_inject_records_distortion_provenance() -> None:
    """Gate provenance lands in spot_features.solver_settings (persisted via JSONB)."""
    _, entry = _inject_with(_result(final_effective_stack=2000, distortion="heavy"))
    settings = entry.spot_features["solver_settings"]
    assert settings["distortion"] == "heavy"
    assert settings["applied_prune"] == pytest.approx(0.15)
    assert settings["applied_n_sizes"] == 2
    assert settings["final_effective_stack"] == 2000
    assert settings["mem_estimate_mb"] == 8000
