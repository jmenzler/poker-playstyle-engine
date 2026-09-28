"""Export boundaries and credential-free canonicalizer smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.canonicalizer import Canonicalizer
from src.protocols.game_state import GameState

# ---------------------------------------------------------------------------
# Research doc lock / completion verification
# ---------------------------------------------------------------------------

EXCLUDED_INTERNAL_DOCS = [
    "docs/research/SOLVER.md",
    "docs/research/DECISIONS-SCHEMA.md",
    "docs/research/FEATURES-v2.md",
    "docs/research/EMBEDDING-LITERATURE-REVIEW.md",
]


@pytest.mark.parametrize("rel_path", EXCLUDED_INTERNAL_DOCS)
def test_export_excludes_internal_research_docs(rel_path: str) -> None:
    assert not Path(rel_path).exists()
    assert Path("src/solver/postflop_cli.py").is_file()
    assert Path("src/vendor/treys/LICENSE").is_file()


# ---------------------------------------------------------------------------
# Canonicalizer smoke — CANON-01, CANON-02 (preflop + postflop)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _canonicalizer() -> Canonicalizer:
    return Canonicalizer.default()


def _preflop_gs() -> GameState:
    return GameState(
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


def _flop_gs() -> GameState:
    return GameState(
        street="flop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=("Qc", "7h", "2s"),
        pot_size_bb=10.0,
        effective_stack_bb=90.0,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("BTN:open_2.5", "BB:call"),
        opponents_remaining=1,
        prior_street_aggressor="BTN",
    )


def test_canonicalizer_preflop_smoke(_canonicalizer: Canonicalizer) -> None:
    """CANON-01: preflop encode() returns 34-dim float32 embedding with correct hard_filter."""
    result = _canonicalizer.encode(_preflop_gs())

    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.dtype == np.float32
    assert result.embedding.shape == (34,), f"Expected (34,), got {result.embedding.shape}"
    assert result.schema_version == 3

    required_keys = {"street_class", "pot_type", "hero_pos_rel", "n_players_active"}
    assert required_keys <= set(result.hard_filter.keys()), (
        f"hard_filter missing keys: {required_keys - set(result.hard_filter.keys())}"
    )
    assert result.hard_filter["street_class"] == "preflop"


def test_canonicalizer_postflop_smoke(_canonicalizer: Canonicalizer) -> None:
    """CANON-01: postflop encode() returns 80-dim float32 embedding with correct hard_filter."""
    result = _canonicalizer.encode(_flop_gs())

    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.dtype == np.float32
    assert result.embedding.shape == (80,), f"Expected (80,), got {result.embedding.shape}"
    assert result.schema_version == 3

    required_keys = {"street_class", "pot_type", "hero_pos_rel", "n_players_active"}
    assert required_keys <= set(result.hard_filter.keys()), (
        f"hard_filter missing keys: {required_keys - set(result.hard_filter.keys())}"
    )
    assert result.hard_filter["street_class"] == "postflop"


def test_canonicalizer_suit_iso_smoke(_canonicalizer: Canonicalizer) -> None:
    """CANON-02: suit-isomorphic GameState pairs produce identical embeddings."""
    # Original: Ah Kd on Qc 7h 2s
    gs1 = _flop_gs()
    # Suit-permuted: swap h↔s, d↔c → As Kc on Qh 7s 2h (same canonical form)
    gs2 = GameState(
        street="flop",
        hero_position="BTN",
        hero_hole_cards=("As", "Kc"),
        board_cards=("Qh", "7s", "2h"),
        pot_size_bb=10.0,
        effective_stack_bb=90.0,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("BTN:open_2.5", "BB:call"),
        opponents_remaining=1,
        prior_street_aggressor="BTN",
    )
    r1 = _canonicalizer.encode(gs1)
    r2 = _canonicalizer.encode(gs2)
    assert np.allclose(r1.embedding, r2.embedding, atol=1e-6), (
        f"Suit-iso GameState pairs produced different embeddings.\n"
        f"  max_diff: {float(np.max(np.abs(r1.embedding - r2.embedding))):.6f}"
    )
    assert r1.hard_filter == r2.hard_filter, f"hard_filter differs: {r1.hard_filter} vs {r2.hard_filter}"


def test_canonicalizer_deterministic(_canonicalizer: Canonicalizer) -> None:
    """CANON-01: encode() is deterministic — same input → byte-identical output."""
    gs = _flop_gs()
    r1 = _canonicalizer.encode(gs)
    r2 = _canonicalizer.encode(gs)
    assert np.array_equal(r1.embedding, r2.embedding), "encode() is not deterministic"
    assert r1.hard_filter == r2.hard_filter
