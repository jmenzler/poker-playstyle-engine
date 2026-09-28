"""Eval baselines registry (D-NEW-30).

Importing this package exposes all 8 baseline classes plus a name → class REGISTRY.
The REGISTRY is consumed by ``src/eval/run_match.py`` (Plan 04) and the FastAPI
``/api/eval`` router (Plan 09) to instantiate opponents by their stable name.

Baselines:
    random         — RandomStrategy         — uniform random over legal actions
    always-call    — AlwaysCallStrategy     — passive (CHECK_CALL > FOLD)
    always-raise   — AlwaysRaiseStrategy    — aggressive (HALF_POT > POT > CHECK_CALL)
    tight-passive  — TightPassiveStrategy   — top-12% RFI; fold to postflop aggression
    LAG-profile    — LAGStrategy            — loose-aggressive heuristic
    TAG-profile    — TAGStrategy            — tight-aggressive heuristic
    prev-engine    — PrevEngineStrategy     — v2 placeholder (OQ-4 deferred)
    chen-bot       — ChenBot                — Chen formula preflop + treys postflop
"""

from __future__ import annotations

from src.eval.baselines.always_call import AlwaysCallStrategy
from src.eval.baselines.always_raise import AlwaysRaiseStrategy
from src.eval.baselines.chen_bot import ChenBot
from src.eval.baselines.lag import LAGStrategy
from src.eval.baselines.prev_engine import PrevEngineStrategy
from src.eval.baselines.random import RandomStrategy
from src.eval.baselines.tag import TAGStrategy
from src.eval.baselines.tight_passive import TightPassiveStrategy

REGISTRY: dict[str, type] = {
    "random": RandomStrategy,
    "always-call": AlwaysCallStrategy,
    "always-raise": AlwaysRaiseStrategy,
    "tight-passive": TightPassiveStrategy,
    "LAG-profile": LAGStrategy,
    "TAG-profile": TAGStrategy,
    "prev-engine": PrevEngineStrategy,
    "chen-bot": ChenBot,
}

__all__ = [
    "REGISTRY",
    "AlwaysCallStrategy",
    "AlwaysRaiseStrategy",
    "ChenBot",
    "LAGStrategy",
    "PrevEngineStrategy",
    "RandomStrategy",
    "TAGStrategy",
    "TightPassiveStrategy",
]
