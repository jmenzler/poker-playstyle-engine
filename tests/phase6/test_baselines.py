"""Tests for the 7 baseline opponent strategies (D-NEW-30).

Coverage:
- Protocol conformance (isinstance check against runtime-checkable Strategy)
- REGISTRY completeness — 7 entries
- Per-baseline behaviour (RandomStrategy determinism, AlwaysCall/Raise priorities, etc.)
- Legality invariant: 100 random states, action returned MUST be legal
- PrevEngineStrategy is a v2 placeholder (NotImplementedError citing OQ-4)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.baselines import (
    REGISTRY,
    AlwaysCallStrategy,
    AlwaysRaiseStrategy,
    LAGStrategy,
    PrevEngineStrategy,
    RandomStrategy,
    TAGStrategy,
    TightPassiveStrategy,
)
from src.eval.strategy import Strategy


def _state(
    legal: tuple[int, ...] = (0, 1, 2),
    hand: tuple[str, str] = ("As", "Kd"),
    public: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build a minimal RLCard-shaped state dict for unit tests."""
    return {
        "legal_actions": list(legal),
        "hand": list(hand),
        "public_cards": list(public),
    }


def _rlcard_state(
    legal: tuple[int, ...] = (0, 1, 2),
    hand: tuple[str, str] = ("As", "Kd"),
    public: tuple[str, ...] = (),
) -> dict[str, object]:
    """Production-shaped RLCard state: hand/public_cards live under raw_obs.

    legal_actions stays top-level, mirroring env.get_state() output.
    """
    return {
        "legal_actions": list(legal),
        "raw_obs": {
            "hand": list(hand),
            "public_cards": list(public),
        },
    }


# ─── Protocol conformance ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cls",
    [
        RandomStrategy,
        AlwaysCallStrategy,
        AlwaysRaiseStrategy,
        TightPassiveStrategy,
        LAGStrategy,
        TAGStrategy,
    ],
)
def test_protocol_conformance(cls: type) -> None:
    """All 6 concrete baselines satisfy the runtime-checkable Strategy protocol."""
    inst = cls(seed=0)
    assert isinstance(inst, Strategy)


def test_registry_complete() -> None:
    """REGISTRY exposes exactly the 7 documented baseline names."""
    assert set(REGISTRY.keys()) == {
        "random",
        "always-call",
        "always-raise",
        "tight-passive",
        "LAG-profile",
        "TAG-profile",
        "prev-engine",
    }


# ─── RandomStrategy ───────────────────────────────────────────────────────────


def test_random_returns_legal_action() -> None:
    s = RandomStrategy(seed=0)
    state = _state(legal=(0, 1, 2))
    for _ in range(50):
        a = s.decide(state)
        assert a in (0, 1, 2)


def test_random_deterministic() -> None:
    """Same seed → identical sequence (T-06-11 mitigation)."""
    s1 = RandomStrategy(seed=42)
    s2 = RandomStrategy(seed=42)
    state = _state(legal=(0, 1, 2, 3, 4))
    seq1 = [s1.decide(state) for _ in range(20)]
    seq2 = [s2.decide(state) for _ in range(20)]
    assert seq1 == seq2


# ─── AlwaysCallStrategy ───────────────────────────────────────────────────────


def test_always_call_prefers_check_call() -> None:
    s = AlwaysCallStrategy()
    assert s.decide(_state(legal=(0, 1, 2))) == 1  # CHECK_CALL preferred
    assert s.decide(_state(legal=(0, 2))) == 0  # CHECK_CALL not legal → FOLD
    assert s.decide(_state(legal=(1,))) == 1  # only CHECK_CALL


# ─── AlwaysRaiseStrategy ──────────────────────────────────────────────────────


def test_always_raise_prefers_half_pot() -> None:
    s = AlwaysRaiseStrategy()
    assert s.decide(_state(legal=(0, 1, 2, 3))) == 2  # HALF_POT preferred
    assert s.decide(_state(legal=(0, 1, 3))) == 3  # HALF_POT not legal → POT
    assert s.decide(_state(legal=(0, 1))) == 1  # No raise legal → CHECK_CALL


# ─── TightPassiveStrategy ─────────────────────────────────────────────────────


def test_tight_passive_weak_preflop_is_legal() -> None:
    """72o preflop returns a legal action (FOLD or CHECK_CALL)."""
    s = TightPassiveStrategy(seed=0)
    state = _state(legal=(0, 1), hand=("7c", "2d"), public=())
    a = s.decide(state)
    assert a in (0, 1)


def test_tight_passive_legal_action_always() -> None:
    s = TightPassiveStrategy(seed=0)
    for legal in [(0, 1), (0, 1, 2), (1, 2, 3, 4), (0,)]:
        a = s.decide(_state(legal=legal))
        assert a in legal


def test_tight_passive_reads_hand_from_raw_obs_and_opens_premium() -> None:
    """With production-shaped state, AKo premium opens preflop (HALF_POT)."""
    s = TightPassiveStrategy(seed=0)
    state = _rlcard_state(legal=(0, 1, 2, 3), hand=("As", "Kh"), public=())
    assert s.decide(state) == 2  # HALF_POT — only reachable if hand read from raw_obs


# ─── LAG / TAG ────────────────────────────────────────────────────────────────


def _random_legal_set(rng: np.random.Generator) -> tuple[int, ...]:
    """Generate a non-empty random subset of {0,1,2,3,4} (legal actions)."""
    size = int(rng.integers(1, 6))  # 1..5 inclusive
    return tuple(sorted(set(int(x) for x in rng.choice([0, 1, 2, 3, 4], size=size))))


def test_lag_returns_legal_over_100_states() -> None:
    s = LAGStrategy(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(100):
        legal = _random_legal_set(rng)
        a = s.decide(_state(legal=legal))
        assert a in legal


def test_tag_returns_legal_over_100_states() -> None:
    s = TAGStrategy(seed=0)
    rng = np.random.default_rng(2)
    for _ in range(100):
        legal = _random_legal_set(rng)
        a = s.decide(_state(legal=legal))
        assert a in legal


def test_tag_reads_hand_from_raw_obs_and_opens_premium() -> None:
    """With production-shaped state, premium broadways raise more than they fold.

    Without raw_obs unwrap, hand is [] → is_premium False → mostly fold/limp.
    """
    s = TAGStrategy(seed=0)
    state = _rlcard_state(legal=(0, 1, 2, 3), hand=("As", "Kh"), public=())
    raises = sum(1 for _ in range(200) if s.decide(state) in (2, 3))
    assert raises > 100  # 80% premium-open rate dominates


def test_tag_reaches_postflop_branch_from_raw_obs() -> None:
    """A flop under raw_obs reaches the postflop branch (can bet HALF_POT).

    Preflop with a non-premium hand never bets; only the postflop branch does.
    """
    s = TAGStrategy(seed=0)
    state = _rlcard_state(legal=(0, 1, 2, 3), hand=("7c", "2d"), public=("Ah", "Td", "5c"))
    bets = sum(1 for _ in range(200) if s.decide(state) == 2)
    assert bets > 0  # postflop 30% bet branch is reachable


def test_lag_is_more_aggressive_than_tight_passive() -> None:
    """LAG should pick a raise action more often than TightPassive on equal states.

    Sampled over many invocations with both seeds fixed for determinism.
    """
    lag = LAGStrategy(seed=123)
    tp = TightPassiveStrategy(seed=123)
    state = _state(legal=(0, 1, 2, 3), hand=("7c", "2d"), public=("Ah", "Td", "5c"))
    lag_raises = sum(1 for _ in range(200) if lag.decide(state) in (2, 3, 4))
    tp_raises = sum(1 for _ in range(200) if tp.decide(state) in (2, 3, 4))
    assert lag_raises > tp_raises


# ─── PrevEngineStrategy (v2 placeholder) ──────────────────────────────────────


def test_prev_engine_raises_with_v2_message() -> None:
    """Constructor must surface OQ-4 deferral (T-06-13 mitigation)."""
    with pytest.raises(NotImplementedError, match="OQ-4"):
        PrevEngineStrategy(version="v0.2.0")
