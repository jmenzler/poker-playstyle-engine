"""End-to-end Phase 1 smoke: real Canonicalizer.encode → Milvus → retrieval.

Uses the CURRENT kNN architecture (Plan 01-20R):
  - EncodeResult.embedding is 34-dim (preflop) or 80-dim (postflop) float32 ndarray
  - EncodeResult.hard_filter is a dict with {street_class, pot_type, hero_pos_rel, n_players_active}
  - Collections: preflop_decisions (34-dim) + postflop_decisions (80-dim)

Validates Phase 1 success criterion #1:
  - observations hypertable (TSDB) writable and readable
  - Milvus dual-collection insert + scalar filter search works with real embedding
  - CANON-01: encode() is pure (no network I/O) — verified by no_network fixture (unit tests)
  - CANON-02: suit-isomorphic pairs produce identical embeddings (property test)

Note: TSDB round-trip is covered by test_smoke_round_trip.py using strategy_nodes table.
This file focuses on Milvus round-trip with real Canonicalizer.encode output.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.canonicalizer import Canonicalizer
from src.protocols.game_state import GameState

pytestmark = pytest.mark.integration


def _flop_gs() -> GameState:
    """Sample postflop GameState for real embedding tests."""
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


def _preflop_gs() -> GameState:
    """Sample preflop GameState for real embedding tests."""
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


@pytest.fixture(scope="module")
def canonicalizer() -> Canonicalizer:
    """Single Canonicalizer instance shared across module tests (parquet loaded once)."""
    return Canonicalizer.default()


@pytest.mark.parametrize(
    "gs_factory,collection_name,expected_dim",
    [
        (_preflop_gs, "preflop_decisions", 34),
        (_flop_gs, "postflop_decisions", 80),
    ],
)
def test_canonicalizer_milvus_round_trip(
    canonicalizer: Canonicalizer,
    milvus_uri: str,
    milvus_token: str | None,
    gs_factory,
    collection_name: str,
    expected_dim: int,
) -> None:
    """Real Canonicalizer.encode output inserts into Milvus and retrieves with filter."""
    from src.db.milvus import connect as mvconnect

    gs = gs_factory()
    result = canonicalizer.encode(gs)

    # Verify embedding shape matches expected collection dim
    assert result.embedding.shape == (expected_dim,), (
        f"Expected ({expected_dim},) for {collection_name}, got {result.embedding.shape}"
    )
    assert result.embedding.dtype == np.float32

    # Verify hard_filter has required keys for scalar field population
    required_keys = {"street_class", "pot_type", "hero_pos_rel", "n_players_active"}
    assert required_keys <= set(result.hard_filter.keys()), (
        f"hard_filter missing keys: {required_keys - set(result.hard_filter.keys())}"
    )

    mv = mvconnect(milvus_uri, token=milvus_token)
    decision_id = f"rt-{uuid.uuid4().hex[:12]}"

    doc = {
        "decision_id": decision_id,
        "embedding": result.embedding.tolist(),
        "street_class": result.hard_filter["street_class"],
        "pot_type": result.hard_filter["pot_type"],
        "hero_pos_rel": result.hard_filter["hero_pos_rel"],
        "n_players_active": int(result.hard_filter["n_players_active"]),
    }
    mv.insert(collection_name=collection_name, data=[doc])
    mv.flush(collection_name)

    try:
        # Scalar filter search: should return exactly our inserted vector
        n_players = int(result.hard_filter["n_players_active"])
        results = mv.search(
            collection_name=collection_name,
            data=[result.embedding.tolist()],
            limit=1,
            filter=f"n_players_active == {n_players}",
            output_fields=["decision_id"],
        )
        ids = {r["entity"]["decision_id"] for r in results[0]}
        assert decision_id in ids, (
            f"Inserted decision_id {decision_id!r} not found in filtered search results: {ids}"
        )
    finally:
        mv.delete(
            collection_name=collection_name,
            filter=f'decision_id == "{decision_id}"',
        )


def test_scalar_filter_excludes_other_player_counts(
    canonicalizer: Canonicalizer,
    milvus_uri: str,
    milvus_token: str | None,
) -> None:
    """Vectors inserted with n_players_active=2 must NOT appear in n_players_active==3 filter."""
    from src.db.milvus import connect as mvconnect

    gs = _flop_gs()
    result = canonicalizer.encode(gs)
    mv = mvconnect(milvus_uri, token=milvus_token)

    decision_id = f"filter-test-{uuid.uuid4().hex[:12]}"
    doc = {
        "decision_id": decision_id,
        "embedding": result.embedding.tolist(),
        "street_class": result.hard_filter["street_class"],
        "pot_type": result.hard_filter["pot_type"],
        "hero_pos_rel": result.hard_filter["hero_pos_rel"],
        "n_players_active": 2,  # HU
    }
    mv.insert(collection_name="postflop_decisions", data=[doc])
    mv.flush("postflop_decisions")

    try:
        # Search with n_players_active==3 — our vector (with n_players=2) must NOT appear
        results = mv.search(
            collection_name="postflop_decisions",
            data=[result.embedding.tolist()],
            limit=10,
            filter="n_players_active == 3",
            output_fields=["decision_id"],
        )
        ids = {r["entity"]["decision_id"] for r in results[0]}
        assert decision_id not in ids, (
            f"n_players_active=2 vector leaked into n_players_active==3 filter: {ids}"
        )
    finally:
        mv.delete(
            collection_name="postflop_decisions",
            filter=f'decision_id == "{decision_id}"',
        )
