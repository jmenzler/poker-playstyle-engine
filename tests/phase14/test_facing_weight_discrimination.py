"""Postflop retrieval must separate facing regimes (checked-to vs facing-bet).

A checked-to query should rank a facing-MATCHED neighbor above a facing-MISMATCHED
one, even when the mismatched neighbor is hand-identical and the matched neighbor is
hand-far. Facing context is a regime gate; hand similarity ranks WITHIN a regime.

These tests exercise the real production weighting (build_postflop_weight_vector +
normalize) on hand-built raw feature vectors. mean=0/std=1 isolates the group-weight
effect from z-scoring. With equity (Group A=1.5) swamping facing (Groups E/F=1.0) the
mismatched distractor wins -> RED until the weights are rebalanced.
"""

from __future__ import annotations

import numpy as np

from tools.feature_extractors.postflop import _GROUP_WEIGHTS
from tools.upsert_milvus import build_postflop_weight_vector, normalize

# Postflop dim layout (0-indexed), per tools/upsert_milvus._POSTFLOP_GROUP_DIM_COUNTS:
#   A equity 0-11 | B draws 12-19 | C made 20-25 | D pos 26-33 | E geom 34-40
#   F betting 41-56 | G prior 57-62 | H texture 63-69 | J villain 70-79
_EQUITY = slice(0, 12)
_DRAWS = slice(12, 20)
_FACING_BET_FRAC = 38  # Group E
_FACING_FLAG = 39  # Group E
_BUCKET_CHECKTO = 44  # Group F facing_size_bucket[0]
_BUCKET_POTBET = 50  # Group F facing_size_bucket[6] (~1x pot)

_NEUTRAL = 0.5  # shared context value for all non-perturbed dims


def _base() -> np.ndarray:
    """80-dim raw vector: neutral context, checked-to facing one-hot."""
    v = np.full(80, _NEUTRAL, dtype=np.float64)
    v[_EQUITY] = 0.5
    v[_DRAWS] = 0.4
    # checked-to facing: bucket0 hot, no bet faced
    v[34:41] = 0.0  # Group E neutralised; facing flags below
    v[41:57] = 0.0  # Group F neutralised; bucket below
    v[_BUCKET_CHECKTO] = 1.0
    return v


def _query() -> np.ndarray:
    return _base()


def _facing_matched_hand_far() -> np.ndarray:
    """Same facing regime (checked-to) but a very different hand."""
    v = _base()
    v[_EQUITY] = 0.0  # large equity gap vs query
    v[_DRAWS] = 0.0  # large draw gap vs query
    return v


def _facing_mismatched_hand_identical() -> np.ndarray:
    """Identical hand to the query, but facing a pot-sized bet (wrong regime)."""
    v = _base()
    v[_BUCKET_CHECKTO] = 0.0
    v[_BUCKET_POTBET] = 1.0
    v[_FACING_FLAG] = 1.0
    v[_FACING_BET_FRAC] = 0.2  # ~1x pot / 5.0
    return v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_facing_match_outranks_hand_similar_distractor() -> None:
    """Facing-matched (hand-far) neighbor must beat facing-mismatched (hand-identical)."""
    w = build_postflop_weight_vector()
    mean = np.zeros(80)
    std = np.ones(80)

    q = normalize(_query(), mean, std, w)
    matched = normalize(_facing_matched_hand_far(), mean, std, w)
    mismatched = normalize(_facing_mismatched_hand_identical(), mean, std, w)

    assert _cos(q, matched) > _cos(q, mismatched), (
        f"facing-mismatched distractor ranks higher: "
        f"cos(matched)={_cos(q, matched):.4f} <= cos(mismatched)={_cos(q, mismatched):.4f}"
    )


def test_facing_group_weight_at_least_equity() -> None:
    """Facing groups (E,F) must not be out-weighted by equity/draws (A,B)."""
    facing = max(_GROUP_WEIGHTS["E"], _GROUP_WEIGHTS["F"])
    hand = max(_GROUP_WEIGHTS["A"], _GROUP_WEIGHTS["B"])
    assert facing >= hand, f"facing weight {facing} < equity/draw weight {hand}"
