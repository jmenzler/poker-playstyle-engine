"""Property test: suit-isomorphic GameState pairs produce identical embeddings.

CANON-02: For any GameState gs and any suit permutation perm,
    Canonicalizer.encode(gs).embedding == Canonicalizer.encode(apply_suit_permutation(gs, perm)).embedding

This verifies that joint_canonicalize correctly normalises the hole↔board
suit relationship before feature extraction.

Uses existing strategies from tests/property/conftest.py.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings

from src.canonicalizer import Canonicalizer
from tests.property.conftest import (
    apply_suit_permutation,
    game_state_strategy,
    suit_permutation_strategy,
)

# Instantiated once per hypothesis run — parquet loaded at first call and reused
_CANONICALIZER = None


def _get_canonicalizer() -> Canonicalizer:
    global _CANONICALIZER
    if _CANONICALIZER is None:
        _CANONICALIZER = Canonicalizer.default()
    return _CANONICALIZER


@given(gs=game_state_strategy(), perm=suit_permutation_strategy())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_suit_isomorphism_preserves_embedding(gs, perm) -> None:  # type: ignore[no-untyped-def]
    """CANON-02: suit permutation must not change the embedding or hard_filter."""
    c = _get_canonicalizer()
    gs_permuted = apply_suit_permutation(gs, perm)

    e1 = c.encode(gs)
    e2 = c.encode(gs_permuted)

    assert np.allclose(e1.embedding, e2.embedding, atol=1e-6), (
        f"Suit permutation changed embedding.\n"
        f"  gs:         street={gs.street} hole={gs.hero_hole_cards} board={gs.board_cards}\n"
        f"  permuted:   hole={gs_permuted.hero_hole_cards} board={gs_permuted.board_cards}\n"
        f"  perm:       {perm}\n"
        f"  max_diff:   {float(np.max(np.abs(e1.embedding - e2.embedding))):.6f}"
    )
    assert e1.hard_filter == e2.hard_filter, (
        f"Suit permutation changed hard_filter: {e1.hard_filter} vs {e2.hard_filter}"
    )
