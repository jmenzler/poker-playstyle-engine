"""Unit tests for src/canonicalizer/encoder.py — Plan 01-20R §Test Strategy.

Tests:
    CANON-U1  test_encoder_preflop_returns_32_dim
    CANON-U2  test_encoder_postflop_returns_80_dim
    CANON-U3  test_encode_no_network_io  (uses no_network fixture)
    CANON-U4  test_encode_deterministic  (same GameState → identical embedding twice)
    CANON-U5  test_encode_result_schema_version
"""

from __future__ import annotations

import numpy as np
import pytest

from src.canonicalizer import Canonicalizer, EncodeResult
from src.protocols.game_state import GameState

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canonicalizer() -> Canonicalizer:
    """Single shared Canonicalizer instance for all unit tests (loads Parquet once)."""
    return Canonicalizer.default()


def _preflop_gs() -> GameState:
    """Minimal valid preflop GameState."""
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
    """Minimal valid flop GameState."""
    return GameState(
        street="flop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=("Jc", "7s", "2h"),
        pot_size_bb=6.0,
        effective_stack_bb=97.5,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("BB:check",),
        opponents_remaining=1,
        prior_street_aggressor="BTN",
    )


# ---------------------------------------------------------------------------
# CANON-U1: Preflop → 32-dim embedding
# ---------------------------------------------------------------------------


def test_encoder_preflop_returns_34_dim(canonicalizer: Canonicalizer) -> None:
    """encode() on a preflop GameState must return a float32 vector of shape (34,)."""
    result = canonicalizer.encode(_preflop_gs())

    assert isinstance(result, EncodeResult)
    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.dtype == np.float32
    assert result.embedding.shape == (34,), f"Expected (34,), got {result.embedding.shape}"


# ---------------------------------------------------------------------------
# CANON-U2: Postflop → 80-dim embedding
# ---------------------------------------------------------------------------


def test_encoder_postflop_returns_80_dim(canonicalizer: Canonicalizer) -> None:
    """encode() on a flop GameState must return a float32 vector of shape (80,)."""
    result = canonicalizer.encode(_flop_gs())

    assert isinstance(result, EncodeResult)
    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.dtype == np.float32
    assert result.embedding.shape == (80,), f"Expected (80,), got {result.embedding.shape}"


# ---------------------------------------------------------------------------
# CANON-U3: No network I/O during encode()
# ---------------------------------------------------------------------------


def test_encode_no_network_io(canonicalizer: Canonicalizer, no_network: None) -> None:
    """encode() must not open any network connections — CANON-01 invariant."""
    # Both preflop and postflop paths tested under network block
    result_pre = canonicalizer.encode(_preflop_gs())
    result_post = canonicalizer.encode(_flop_gs())

    assert result_pre.embedding.shape == (34,)
    assert result_post.embedding.shape == (80,)


# ---------------------------------------------------------------------------
# CANON-U4: Determinism — same GameState twice → byte-identical embedding
# ---------------------------------------------------------------------------


def test_encode_deterministic(canonicalizer: Canonicalizer) -> None:
    """encode() must be deterministic: same input → same output on repeated calls."""
    gs_pre = _preflop_gs()
    gs_post = _flop_gs()

    r1_pre = canonicalizer.encode(gs_pre)
    r2_pre = canonicalizer.encode(gs_pre)
    assert np.array_equal(r1_pre.embedding, r2_pre.embedding), "Preflop embedding is not deterministic"
    assert r1_pre.hard_filter == r2_pre.hard_filter

    r1_post = canonicalizer.encode(gs_post)
    r2_post = canonicalizer.encode(gs_post)
    assert np.array_equal(r1_post.embedding, r2_post.embedding), "Postflop embedding is not deterministic"
    assert r1_post.hard_filter == r2_post.hard_filter


# ---------------------------------------------------------------------------
# CANON-U5: EncodeResult schema_version and hard_filter structure
# ---------------------------------------------------------------------------


def test_encode_result_schema_version(canonicalizer: Canonicalizer) -> None:
    """EncodeResult.schema_version must be 3 and hard_filter must have required keys."""
    for gs in (_preflop_gs(), _flop_gs()):
        result = canonicalizer.encode(gs)

        assert result.schema_version == 3
        assert isinstance(result.hard_filter, dict)

        required_keys = {"street_class", "pot_type", "hero_pos_rel", "n_players_active"}
        missing = required_keys - set(result.hard_filter.keys())
        assert not missing, f"hard_filter missing keys: {missing}"


def test_default_singleton_is_shared_across_threads() -> None:
    """Concurrent default() calls must all return the one process-scoped instance."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: Canonicalizer.default(), range(16)))
    assert all(inst is instances[0] for inst in instances)
