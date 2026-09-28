"""Harness drives the sizing-aware executor and surfaces divergence (M3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.canonicalizer import EncodeResult
from src.protocols.game_state import GameState
from src.sim.harness import run_record_session

_GS = GameState(
    street="preflop",
    hero_position="BTN",
    hero_hole_cards=("Ah", "Kd"),
    board_cards=(),
    pot_size_bb=3.0,
    effective_stack_bb=100.0,
    hero_facing_bet_bb=2.5,
    hero_bet_size_bb=0.0,
    action_sequence=(),
    opponents_remaining=2,
    prior_street_aggressor=None,
)

_ENC = EncodeResult(
    embedding=np.zeros(32),
    hard_filter={"street_class": "preflop", "pot_type": "srp"},
    schema_version=2,
)


def _mock_engine() -> MagicMock:
    engine = MagicMock()
    engine._canon.encode.return_value = _ENC
    engine.decide_with_encoding.return_value = ("open_3bb", False, None, _ENC)
    return engine


def test_harness_calls_execute_action_not_raw_step():
    """The harness must drive the env via adapter.execute_action (sizing-aware),
    not the old map_to_rlcard_action + raw env.step path."""
    adapter = MagicMock()
    adapter._env = MagicMock()
    adapter.has_more.side_effect = [True, False, True, False]
    adapter.next_game_state.return_value = _GS
    adapter.execute_action.return_value = {
        "intent_class": "aggressive",
        "diverged": False,
        "executed_action": None,
        "executed_chips": 6,
    }
    adapter.divergence_counts.return_value = {"aggressive": 0, "passive": 0}

    run_record_session(adapter, _mock_engine(), MagicMock(), session_seed=42, n_hands=2)

    assert adapter.execute_action.call_count == 2
    adapter.execute_action.assert_called_with("open_3bb")
    # Old path must not be used.
    adapter._env.step.assert_not_called()


def test_summary_exposes_divergence_counts():
    """Run summary must surface adapter divergence tallies (M3 observability)."""
    adapter = MagicMock()
    adapter._env = MagicMock()
    adapter.has_more.side_effect = [True, False]
    adapter.next_game_state.return_value = _GS
    adapter.execute_action.return_value = {
        "intent_class": "aggressive",
        "diverged": True,
        "executed_action": 4,
        "executed_chips": None,
    }
    adapter.divergence_counts.return_value = {"aggressive": 1, "passive": 0}

    summary = run_record_session(adapter, _mock_engine(), MagicMock(), session_seed=1, n_hands=1)

    assert "size_divergences" in summary
    assert summary["size_divergences"] == {"aggressive": 1, "passive": 0}
