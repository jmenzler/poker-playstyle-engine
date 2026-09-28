"""LRU result cache + hot-node warmup helpers (ENGN-05).

Caches per-(hard_filter, embedding-bytes-hash) Milvus search results to bypass
the 5-15 ms HNSW path on repeated queries. Plan 03-04's PC bench uses this to
hit p99 < 5 ms on the sim hot path.

Layers:
    1. Hot-node warmup: engine_from_env pre-populates the cache with the most
       common preflop spots via WARMUP_PREFLOP_STATES (best-effort).
    2. Per-decision LRU: subsequent decide() calls on identical (hard_filter,
       embedding) reuse the cached neighbor list — no Milvus search RPC.

Not thread-safe. Phase 3 sim is single-threaded; if reused for a multi-threaded
inference server, wrap with threading.Lock at the call site.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from src._log import get_logger
from src.protocols.game_state import GameState

log = get_logger("decision_engine.cache")


# WARMUP_PREFLOP_STATES: deterministic small set covering the most-common
# preflop spots. engine_from_env calls decide() on each to pre-populate the
# LRU. Choice rationale: the ~169 hole classes x ~5 contexts space is too
# large to warm exhaustively, but the top dozen cover ~40% of HH-frequency
# mass per the Phase 2 frequency analysis.
WARMUP_PREFLOP_STATES: list[GameState] = [
    GameState(
        street="preflop",
        hero_position=pos,
        hero_hole_cards=hc,
        board_cards=(),
        pot_size_bb=pot,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=facing,
        hero_bet_size_bb=0.0,
        action_sequence=(),
        opponents_remaining=2,
        prior_street_aggressor=None,
    )
    for pos, hc, pot, facing in (
        ("BTN", ("Ah", "Kd"), 3.0, 2.5),
        ("BTN", ("Qs", "Qd"), 3.0, 2.5),
        ("BTN", ("7c", "7d"), 3.0, 2.5),
        ("CO", ("Ah", "Kd"), 3.0, 2.5),
        ("CO", ("Ts", "Jh"), 3.0, 2.5),
        ("MP", ("Ah", "Ad"), 3.0, 2.5),
        ("MP", ("Kh", "Qd"), 3.0, 2.5),
        ("BB", ("Ah", "Kd"), 7.5, 5.0),  # facing 3bet
        ("BB", ("7c", "2d"), 3.0, 2.5),
        ("SB", ("Ah", "Kh"), 3.0, 2.5),
        ("UTG", ("As", "Ad"), 1.5, 1.0),
        ("UTG", ("Ts", "Th"), 1.5, 1.0),
    )
]


def make_cache_key(hard_filter: dict, embedding: np.ndarray) -> tuple:
    """Deterministic cache key: (sorted hard_filter tuple, embedding SHA1 digest).

    Why a tuple-of-sorted-items: Python dicts preserve insertion order; sorting
    canonicalizes the key irrespective of how the encoder built the dict.

    Why SHA1 over the embedding bytes: a 20-byte digest is small + fast; the
    Birthday-paradox collision probability for 4096 cached entries with a
    160-bit hash is ~7e-44. Adequate for cache identity.

    Note: embedding.tobytes() incorporates dtype + shape + content. Embeddings
    of different dtypes (float32 vs float64) hash differently, which is the
    correct behavior — they are not interchangeable inputs to Milvus.
    """
    hf_tuple = tuple(sorted(hard_filter.items()))
    emb_hash = hashlib.sha1(embedding.tobytes()).digest()
    return (hf_tuple, emb_hash)


class LRUResultCache:
    """Simple bounded LRU cache keyed by (hard_filter tuple, embedding hash).

    Implementation: OrderedDict with move_to_end on read and popitem(last=False)
    on overflow (evicts the least-recently-used entry).

    Not thread-safe. For multi-threaded use, wrap with threading.Lock.

    Args:
        maxsize: Maximum number of cached entries. Default 4096 per ENGN-05
            (entry size ~ k=10 neighbor dicts ~ 200 bytes each => ~8 MB at full
            cache). Set to 0 to disable caching entirely (always miss).
    """

    def __init__(self, maxsize: int = 4096) -> None:
        self._cache: OrderedDict[tuple, Any] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, key: tuple) -> Any | None:
        """Return cached value (and refresh LRU position), else None."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: tuple, value: Any) -> None:
        """Insert or refresh entry. Evicts oldest if at maxsize and key is new."""
        if self._maxsize <= 0:
            return  # disabled cache
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # evict oldest

    @property
    def hits(self) -> int:
        """Number of get() calls that returned a cached value."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of get() calls that returned None."""
        return self._misses

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)
