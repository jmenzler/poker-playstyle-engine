"""Head-to-head corpus-version A/B.

Plays the v_new engine against the v_old engine at one table (paired seed), each
pinned via ``engine_from_env``. ``bb_per_100`` is v_new's edge over v_old;
factory + runner are injectable."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src._log import get_logger

log = get_logger("study.compare")


def _default_engine_factory(version: int) -> Any:
    """Build a pinned engine for ``version`` via the env factory."""
    from src.decision_engine.engine import engine_from_env

    return engine_from_env(as_of_version=version)


def _default_head_to_head(engine_new: Any, engine_old: Any, *, hands: int, seed: int) -> Any:
    """Play engine_new (hero) against engine_old (villain) at one HU table, paired seed."""
    from src.eval.run_match import run_match

    return run_match(
        opponent_engine=engine_old,
        opponent_label="engine_old",
        hands=hands,
        seed=seed,
        persist=False,
        _engine=engine_new,
    )


def compare(
    v_old: int,
    v_new: int,
    *,
    hands: int = 2000,
    seed: int = 42,
    _engine_factory: Callable[[int], Any] = _default_engine_factory,
    _head_to_head: Callable[..., Any] = _default_head_to_head,
) -> dict:
    """Head-to-head A/B: play the v_new engine vs the v_old engine (paired seed).

    Returns {v_old, v_new, bb_per_100, ci, n_hands, mode}; bb_per_100 is v_new's
    edge over v_old (positive = v_new wins).
    """
    log.info("study.compare.started", v_old=v_old, v_new=v_new, hands=hands, seed=seed)

    engine_old = _engine_factory(v_old)
    engine_new = _engine_factory(v_new)

    result = _head_to_head(engine_new, engine_old, hands=hands, seed=seed)

    out = {
        "v_old": v_old,
        "v_new": v_new,
        "bb_per_100": float(result.bb_per_100),
        "ci": (float(result.ci_low), float(result.ci_high)),
        "n_hands": hands,
        "mode": "head_to_head",
    }
    log.info("study.compare.complete", v_old=v_old, v_new=v_new, bb_per_100=out["bb_per_100"])
    return out
