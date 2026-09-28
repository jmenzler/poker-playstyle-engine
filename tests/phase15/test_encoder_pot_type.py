"""Encoder pot_type propagation tests.  long-ok

Verifies that GameState.pot_type is honored by the encoder:
    test_none_falls_back: encoder uses _infer_pot_type when gs.pot_type is None
    test_3bet_propagates: encoder uses gs.pot_type="3bet" → hard_filter/cluster_key carry it
    test_backward_compat_all_existing_kwargs: construction with 11 kwargs still works
    test_embedding_unchanged: same spot, pot_type None vs set → identical embedding (D-10)
"""

from __future__ import annotations

from src.canonicalizer import Canonicalizer
from src.protocols.game_state import GameState


def test_none_falls_back(make_game_state):
    """When gs.pot_type is None, encoder falls back to _infer_pot_type(action_sequence).

    action_sequence=('BTN:raise_2.5','BB:call') → 1 raise → infer 'srp'.
    hard_filter['pot_type'] must equal the inferred value (not None).
    """
    gs = make_game_state(
        action_sequence=("BTN:raise_2.5", "BB:call"),
        pot_type=None,
    )
    canon = Canonicalizer.default()
    result = canon.encode(gs)
    inferred = result.hard_filter.get("pot_type")
    assert inferred is not None
    assert inferred == "srp", f"expected inferred 'srp', got {inferred!r}"


def test_3bet_propagates(make_game_state):
    """gs.pot_type='3bet' propagates into hard_filter and cluster_key (D-09).

    action_sequence=('BB:check',) would infer 'srp' without the fix.
    With the fix, gs.pot_type takes precedence.
    """
    gs = make_game_state(
        street="flop",
        action_sequence=("BB:check",),
        pot_type="3bet",
    )
    canon = Canonicalizer.default()
    result = canon.encode(gs)
    assert result.hard_filter.get("pot_type") == "3bet", (
        f"hard_filter pot_type expected '3bet', got {result.hard_filter.get('pot_type')!r}"
    )
    # pot_type flows into observations.cluster_key via hard_filter['pot_type']
    assert "pot_type" in result.hard_filter, "pot_type must be present in hard_filter for cluster routing"


def test_backward_compat_all_existing_kwargs():
    """All 11 existing keyword args construct without pot_type; no positional shift."""
    gs = GameState(
        street="preflop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=(),
        pot_size_bb=3.0,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=2.5,
        hero_bet_size_bb=0.0,
        action_sequence=("UTG:fold", "CO:fold", "BTN:raise_2.5", "SB:fold", "BB:call"),
        opponents_remaining=1,
        prior_street_aggressor=None,
    )
    assert gs.pot_type is None
    assert gs.street == "preflop"


def test_embedding_unchanged(make_game_state):
    """Same GameState with pot_type=None vs pot_type='3bet' → identical embedding (D-10).

    pot_type is a filter/cluster_key field only. The embedding vector must be byte-identical
    regardless of whether gs.pot_type is set, proving zero re-embed is required.
    """
    gs_none = make_game_state(pot_type=None)
    gs_3bet = make_game_state(pot_type="3bet")
    canon = Canonicalizer.default()
    result_none = canon.encode(gs_none)
    result_3bet = canon.encode(gs_3bet)
    import numpy as np

    assert np.array_equal(result_none.embedding, result_3bet.embedding), (
        "Embedding changed when pot_type was set — this would force a full re-embed. "
        "pot_type must only affect hard_filter/cluster_key, not the embedding vector."
    )
