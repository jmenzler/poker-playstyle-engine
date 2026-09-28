"""tests/phase5/test_strategy_override.py — Unit tests for StrategyOverride.

Requirements:
- Target cluster_key: samples from candidate_dist; base_engine.decide_with_encoding NOT called.
- Non-target cluster: delegates to base_engine.decide_with_encoding.
- Zero Milvus calls when intercepting target cluster.
- No DB connection opened by StrategyOverride instantiation.
- Returns 4-tuple matching KNNDecisionEngine.decide_with_encoding contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

TARGET_KEY = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
OTHER_KEY = "hero_pos_rel=UTG|n_players_active=3|pot_type=3bet|street_class=preflop"

CANDIDATE_DIST = {"call": 1.0}  # deterministic: always "call"


def _make_enc_mock(hard_filter: dict) -> MagicMock:
    """Return an EncodeResult-shaped MagicMock with the given hard_filter."""
    enc = MagicMock()
    enc.hard_filter = hard_filter
    return enc


def _hard_filter_from_key(cluster_key: str) -> dict:
    """Inverse of _cluster_key_from_filter — split 'k=v' pairs."""
    result: dict = {}
    for token in cluster_key.split("|"):
        k, v = token.split("=", 1)
        # Attempt numeric coercion for n_players_active
        try:
            result[k] = int(v)
        except ValueError:
            result[k] = v
    return result


def _make_base_engine(hard_filter_for_encode: dict) -> MagicMock:
    """Fabricate a KNNDecisionEngine-shaped MagicMock.

    _canon.encode(gs) returns an EncodeResult mock with the given hard_filter.
    """
    base = MagicMock()
    enc = _make_enc_mock(hard_filter_for_encode)
    base._canon.encode.return_value = enc
    base._client = MagicMock()  # Milvus client attribute
    return base


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_intercepts_target_cluster_samples_from_candidate_dist():
    """Target cluster: action sampled from candidate_dist; base engine NOT called."""
    from src.validation.override import StrategyOverride

    hard_filter = _hard_filter_from_key(TARGET_KEY)
    base = _make_base_engine(hard_filter)

    rng = np.random.default_rng(42)
    override = StrategyOverride(base, TARGET_KEY, CANDIDATE_DIST, rng)

    gs = MagicMock()
    action, _flagged_sparse, _max_neighbor_distance, enc = override.decide_with_encoding(gs)

    assert action == "call", f"Expected 'call' from candidate_dist, got {action!r}"
    base.decide_with_encoding.assert_not_called()
    # enc should be the encode result from base._canon.encode
    assert enc is base._canon.encode.return_value


def test_delegates_non_target_cluster_to_base_engine():
    """Non-target cluster: delegates to base_engine.decide_with_encoding; returns its tuple."""
    from src.validation.override import StrategyOverride

    # Encode returns OTHER_KEY's filter
    hard_filter = _hard_filter_from_key(OTHER_KEY)
    base = _make_base_engine(hard_filter)

    # base engine returns a valid 4-tuple for the non-target case
    other_enc = _make_enc_mock(hard_filter)
    base.decide_with_encoding.return_value = ("fold", False, None, other_enc)

    rng = np.random.default_rng(42)
    # Override is configured for TARGET_KEY; gs produces OTHER_KEY
    override = StrategyOverride(base, TARGET_KEY, CANDIDATE_DIST, rng)

    gs = MagicMock()
    result = override.decide_with_encoding(gs)

    base.decide_with_encoding.assert_called_once_with(gs)
    assert result == ("fold", False, None, other_enc)


def test_intercepts_target_makes_zero_milvus_calls():
    """Intercepting target cluster: base_engine._client.search is NEVER called."""
    from src.validation.override import StrategyOverride

    hard_filter = _hard_filter_from_key(TARGET_KEY)
    base = _make_base_engine(hard_filter)

    rng = np.random.default_rng(0)
    override = StrategyOverride(base, TARGET_KEY, CANDIDATE_DIST, rng)

    gs = MagicMock()
    override.decide_with_encoding(gs)

    base._client.search.assert_not_called()


def test_no_db_connection_held(monkeypatch: pytest.MonkeyPatch):
    """Instantiating StrategyOverride opens no DB connections."""
    from src.validation.override import StrategyOverride

    # Patch both db connect helpers to detect any calls
    timescale_connect_called = []
    milvus_connect_called = []

    monkeypatch.setattr(
        "src.db.timescale.connect",
        lambda *a, **kw: timescale_connect_called.append(1),
    )
    monkeypatch.setattr(
        "src.db.milvus.connect",
        lambda *a, **kw: milvus_connect_called.append(1),
    )

    hard_filter = _hard_filter_from_key(TARGET_KEY)
    base = _make_base_engine(hard_filter)

    StrategyOverride(base, TARGET_KEY, CANDIDATE_DIST, np.random.default_rng(0))

    assert not timescale_connect_called, "timescale.connect should not be called"
    assert not milvus_connect_called, "milvus.connect should not be called"


def test_returns_4_tuple_matching_engine_contract():
    """Both intercept and delegate paths return a 4-tuple (str, bool, float|None, EncodeResult)."""
    from src.validation.override import StrategyOverride

    # --- Intercept path ---
    target_filter = _hard_filter_from_key(TARGET_KEY)
    base_target = _make_base_engine(target_filter)
    rng1 = np.random.default_rng(7)
    override_target = StrategyOverride(base_target, TARGET_KEY, CANDIDATE_DIST, rng1)

    gs = MagicMock()
    result_intercept = override_target.decide_with_encoding(gs)
    assert len(result_intercept) == 4, "Intercept path must return 4-tuple"
    action_t, flagged_t, dist_t, _enc_t = result_intercept
    assert isinstance(action_t, str)
    assert isinstance(flagged_t, bool)
    assert dist_t is None  # intercept: max_neighbor_distance is None

    # --- Delegate path ---
    other_filter = _hard_filter_from_key(OTHER_KEY)
    base_other = _make_base_engine(other_filter)
    other_enc = _make_enc_mock(other_filter)
    base_other.decide_with_encoding.return_value = ("raise_min", True, 0.42, other_enc)
    rng2 = np.random.default_rng(7)
    override_other = StrategyOverride(base_other, TARGET_KEY, CANDIDATE_DIST, rng2)

    result_delegate = override_other.decide_with_encoding(gs)
    assert len(result_delegate) == 4, "Delegate path must return 4-tuple"
    action_d, flagged_d, dist_d, _enc_d = result_delegate
    assert isinstance(action_d, str)
    assert isinstance(flagged_d, bool)
    assert dist_d == 0.42
