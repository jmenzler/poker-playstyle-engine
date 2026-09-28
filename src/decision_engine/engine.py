"""KNNDecisionEngine: kNN retrieval + blending + sampling (Phase 3).

Pipeline (synchronous, read-only):
    GameState
      -> Canonicalizer.encode()  # embedding + hard_filter (single encode call)
      -> LRU cache lookup on (hard_filter, embedding_hash)
         hit  -> reuse cached Milvus search_results
         miss -> normalize() with per-street z-score manifest + group weights
              -> Milvus client.search() filtered by `active == True` AND hard_filter
              -> store result in LRU
      -> blend_distributions()  # weight = similarity * confidence * gto_score
      -> sample_action()
      -> action_str

Requirements satisfied:
    ENGN-01: decide() returns one of 15 canonical actions; no DB writes
    ENGN-02: kNN retrieval with hard-filter expression INCLUDING `active == True`
    ENGN-03: blend formula weight = similarity * confidence * gto_score (see blending.py)
    ENGN-04: p99 < 50ms on Mac (mock Milvus benchmark)
    ENGN-05: p99 < 5ms in sim path via LRU cache + hot-node warmup + ef=64
    ERR-03:  NoStrategyError on empty Milvus results
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from src._errors import NoStrategyError
from src._log import get_logger
from src.canonicalizer import Canonicalizer, EncodeResult
from src.decision_engine.blending import blend_distributions, legalize_actions, sample_action
from src.decision_engine.cache import (
    WARMUP_PREFLOP_STATES,
    LRUResultCache,
    make_cache_key,
)
from src.decision_engine.preflop_chart import PreflopChartStrategy
from src.decision_engine.sparseness import compute_flagged_sparse, compute_max_neighbor_distance
from src.protocols.game_state import GameState
from src.study import corpus_versions
from tools.upsert_milvus import (
    build_postflop_weight_vector,
    build_preflop_weight_vector,
    normalize,
)
from tools.zscore_fit import load_manifest

if TYPE_CHECKING:
    from pymilvus import MilvusClient

log = get_logger("decision_engine.engine")

_DEFAULT_K = 10
_DEFAULT_EF = 64
_DEFAULT_LRU_SIZE = 4096

# Default chart tree, swappable via env PREFLOP_CHART_SET.
_DEFAULT_PREFLOP_CHART_DIR = "charts/preflop/inferred_6max"

# Sentinel results_row marking a preflop chart hit, so decide_with_encoding skips
# the kNN sparseness helpers (no neighbor list to route) and reports a clean spot.
CHART_HIT: object = object()

# ENGN-02: kNN retrieval MUST be filtered to active==True (soft-delete safety).
# pymilvus BOOL filter uses Python-style True/False; capital T per convention.
_ACTIVE_FILTER_FRAGMENT = "active == True"

# Milvus output_fields. The 3 size fields drive the postflop snap but are absent
# on pre-snap indexes, so they're filtered to the live schema before each search.
_OUTPUT_FIELDS = [
    "decision_id",
    "hero_action_type",
    "confidence",
    "gto_score",
    "hero_action_size_pot_frac",
    "raise_ratio",
    "hero_action_allin",
    "action_dist",
]

# action_dist rides back only from migrated collections, so it gets the same
# schema-gated treatment as the snap fields (kept iff present in the live schema).
_SNAP_FIELDS = frozenset({"hero_action_size_pot_frac", "raise_ratio", "hero_action_allin", "action_dist"})
_ENGINE_SCHEMA_FIELDS: dict[str, set[str]] = {}


def _engine_output_fields(client, collection: str) -> list[str]:
    """_OUTPUT_FIELDS filtered to fields the collection actually has (cached).

    Fails loud on a describe_collection RPC error: silently dropping the snap
    fields collapses all postflop bet/raise sizing to defaults with no signal,
    so a transient Milvus fault must surface, not degrade. A collection that
    genuinely lacks a snap field is handled by the membership check below.
    """
    names = _ENGINE_SCHEMA_FIELDS.get(collection)
    if names is None:
        names = {f["name"] for f in client.describe_collection(collection)["fields"]}
        _ENGINE_SCHEMA_FIELDS[collection] = names
    return [f for f in _OUTPUT_FIELDS if f in names or f not in _SNAP_FIELDS]


def _cosine_sim_to_distance(hits: list[dict]) -> list[dict]:
    """Convert pymilvus COSINE scores (similarity, 1.0=identical) to cosine
    distance (0.0=identical) in-place, so blend + sparseness — which both assume
    a distance — consume the right scale."""
    for h in hits:
        h["distance"] = 1.0 - float(h.get("distance", 0.0))
    return hits


def _build_filter(
    hard_filter: dict,
    *,
    include_active: bool = True,
    spot_features: dict | None = None,
    as_of_ts: int | None = None,
) -> str:
    """Build a pymilvus filter expression from a canonicalizer hard_filter dict.

    Always prefixes `active == True` (ENGN-02) unless explicitly disabled.
    String scalars are double-quoted; bool/int scalars are unquoted with
    capitalised Python literals (True/False/123).

    Args:
        hard_filter: dict of canonical scalar values from EncodeResult.hard_filter.
            Expected keys: street_class (str), pot_type (str), hero_pos_rel (str),
            n_players_active (int).
        include_active: if False, omit the `active == True` prefix (test-only
            escape hatch — production callers MUST keep it on).
        spot_features: optional decide-time scalar payload (D-07-11b). May carry
            `street` (str) and `spr_x100` (int). These are filter-only — they
            are NEVER added to hard_filter and never enter cluster_key, so the
            ~282k existing PC rows remain queryable. `spr_x100 == -1` is the
            preflop sentinel and skips the spr predicate entirely.
        as_of_ts: corpus-version pin (INT epoch). When set, omit the active
            prefix and select then-state via the added_at/removed_at window.

    Returns:
        pymilvus filter expression as a single string, parts joined by ` and `.
    """
    parts: list[str] = []
    if as_of_ts is not None:
        # As-of replays then-state: active is current state, so the window
        # (added_at/removed_at) replaces the active prefix entirely.
        parts.append(f"added_at <= {as_of_ts}")
        parts.append(f"(removed_at == 0 or removed_at > {as_of_ts})")
    elif include_active:
        parts.append(_ACTIVE_FILTER_FRAGMENT)
    for key, val in hard_filter.items():
        if isinstance(val, bool):
            parts.append(f"{key} == {val}")
        elif isinstance(val, str):
            parts.append(f'{key} == "{val}"')
        else:
            parts.append(f"{key} == {val}")

    # D-07-11b: street + spr_x100 are SEARCH-time scalars. They sit outside
    # hard_filter so cluster_key format is preserved.
    if spot_features:
        street_val = spot_features.get("street")
        if street_val:
            parts.append(f'street == "{street_val}"')
        spr_val = spot_features.get("spr_x100")
        if spr_val is not None and spr_val != -1:
            spr = int(spr_val)
            lo = max(0, int(spr * 0.5))
            hi = int(spr * 1.5)
            parts.append(f"spr_x100 >= {lo} and spr_x100 <= {hi}")
    return " and ".join(parts)


class KNNDecisionEngine:
    """kNN-based DecisionEngine. Implements src.protocols.decision_engine.DecisionEngine.

    Construction is heavy (loads two z-score manifests + builds weight vectors
    + instantiates Canonicalizer). decide()/decide_with_encoding() are pure +
    read-only: only Milvus search() is invoked; no insert/upsert/delete.

    Args:
        milvus_client: pymilvus 3.0 MilvusClient instance.
        preflop_manifest_path: Path to tools/zscore_preflop.json.
        postflop_manifest_path: Path to tools/zscore_postflop.json.
        k: kNN limit (default 10).
        ef: HNSW search ef parameter (default 64 — production-tuned).
        rng_seed: numpy RNG seed for sample_action (None = nondeterministic).
        lru_size: LRU result-cache size (default 4096, set 0 to disable).
        as_of_ts: corpus-version pin (INT epoch). None = live; set = then-state.
    """

    def __init__(
        self,
        milvus_client: MilvusClient,
        preflop_manifest_path: Path,
        postflop_manifest_path: Path,
        *,
        k: int = _DEFAULT_K,
        ef: int = _DEFAULT_EF,
        rng_seed: int | None = None,
        lru_size: int = _DEFAULT_LRU_SIZE,
        use_preflop_charts: bool = True,
        preflop_chart_dir: str | None = None,
        as_of_ts: int | None = None,
    ) -> None:
        self._client = milvus_client
        self._as_of_ts = as_of_ts
        self._canon = Canonicalizer.default()
        self._preflop_mean, self._preflop_std = load_manifest(preflop_manifest_path)
        self._postflop_mean, self._postflop_std = load_manifest(postflop_manifest_path)
        self._preflop_weights = build_preflop_weight_vector()
        self._postflop_weights = build_postflop_weight_vector()
        self._k = k
        self._ef = ef
        self._rng = np.random.default_rng(rng_seed)
        self._cache = LRUResultCache(maxsize=lru_size)
        self._use_preflop_charts = use_preflop_charts
        self._preflop_chart: PreflopChartStrategy | None = None
        if use_preflop_charts:
            chart_dir = preflop_chart_dir or os.environ.get("PREFLOP_CHART_SET") or _DEFAULT_PREFLOP_CHART_DIR
            self._preflop_chart = PreflopChartStrategy.from_dir(chart_dir)

    def decide(self, gs: GameState) -> str:
        """ENGN-01 entry point: returns one of 15 canonical action strings."""
        action, _flag, _max_dist, _enc = self.decide_with_encoding(gs)
        return action

    def decide_with_encoding(self, gs: GameState) -> tuple[str, bool, float | None, EncodeResult]:
        """Return (action, flagged_sparse, max_neighbor_distance, EncodeResult).

        Used by SimHarness to populate observation rows with the same
        embedding / cluster_key / hard_filter that drove the decision —
        avoids double-encoding. Both sparseness signals are computed from the
        same Milvus result_row that produced the action.

        Returns:
            action: one of 15 canonical action strings.
            flagged_sparse: True when the kNN neighborhood was sparse at
                decide-time ((max_distance > SPARSE_DIST_TAU) OR
                (n_neighbors < SPARSE_N_MIN)).
            max_neighbor_distance: maximum COSINE distance among retrieved
                neighbors, or None on the NoStrategy fallback path (raised
                before this function returns — callers catch NoStrategyError
                and set both signals to their fallback values directly).
            enc: EncodeResult carrying embedding + hard_filter used by the
                harness for cluster_key construction and observation writes.
        """
        enc = self._canon.encode(gs)
        action, results_row = self._decide_from_enc(enc, gs)
        if results_row is CHART_HIT:
            # Covered preflop spot: clean, not sparse. Skip the sparseness helpers
            # (no neighbor list) so autoloop isn't told a covered spot is sparse.
            return action, False, 0.0, enc
        neighbors = cast("list[dict[str, Any]]", results_row)
        flagged_sparse = compute_flagged_sparse(neighbors)
        max_neighbor_distance = compute_max_neighbor_distance(neighbors)
        return action, flagged_sparse, max_neighbor_distance, enc

    def _decide_from_enc(self, enc: EncodeResult, gs: GameState) -> tuple[str, list[dict] | object]:
        """Run the chart lookup (preflop) or the kNN pipeline on an EncodeResult.
        Returns (action, results_row): the Milvus hit list for the kNN path, or
        the CHART_HIT sentinel when a preflop chart covered the spot. ``gs`` also
        feeds the SEARCH-time scalar filters (street + spr_x100).
        """
        is_preflop = str(enc.hard_filter.get("street_class", "")).lower() == "preflop"

        # Covered preflop spots short-circuit the kNN blend. Sample by frequency
        # from the same seeded RNG the kNN path uses (determinism intact).
        if is_preflop and self._use_preflop_charts and self._preflop_chart is not None:
            dist = self._preflop_chart.lookup(gs, enc.hard_filter)
            if dist is not None:
                # Chart dists exclude fold (it's the implicit remainder). Inject it back
                # so legalize_actions doesn't renormalize the fold mass into the played
                # verbs — without this the engine never folds the bottom of a range.
                fold_mass = max(0.0, 1.0 - sum(dist.values()))
                dist = {**dist, "fold": dist.get("fold", 0.0) + fold_mass}
                facing = float(getattr(gs, "hero_facing_bet_bb", 0) or 0)
                return sample_action(legalize_actions(dist, facing), self._rng), CHART_HIT

        if is_preflop:
            collection = "preflop_decisions"
            mean = self._preflop_mean
            std = self._preflop_std
            weights = self._preflop_weights
        else:
            collection = "postflop_decisions"
            mean = self._postflop_mean
            std = self._postflop_std
            weights = self._postflop_weights

        vec = normalize(np.asarray(enc.embedding, dtype=np.float64), mean, std, weights)

        # LRU cache lookup — keyed by (sorted hard_filter, sha1(vec))
        cache_key = make_cache_key(enc.hard_filter, vec)
        cached = self._cache.get(cache_key)
        if cached is not None:
            results_row = cached
        else:
            # D-07-11b: build decide-time scalar payload from GameState. street +
            # spr_x100 enter the Milvus filter expr only; hard_filter / cluster_key
            # untouched so existing PC rows remain queryable.
            sf: dict = {}
            gs_street = getattr(gs, "street", None)
            if gs_street:
                sf["street"] = str(gs_street).lower()
            # SPR banding is postflop-only: a preflop band strips ~98% of 4bets.
            pot_bb = float(getattr(gs, "pot_size_bb", 0) or 0)
            eff_bb = float(getattr(gs, "effective_stack_bb", 0) or 0)
            if not is_preflop and pot_bb > 0:
                sf["spr_x100"] = round((eff_bb / pot_bb) * 100)
            else:
                sf["spr_x100"] = -1
            filter_expr = _build_filter(
                enc.hard_filter, include_active=True, spot_features=sf, as_of_ts=self._as_of_ts
            )
            results = self._client.search(
                collection_name=collection,
                data=[vec.tolist()],
                limit=self._k,
                filter=filter_expr,
                search_params={"metric_type": "COSINE", "params": {"ef": self._ef}},
                output_fields=_engine_output_fields(self._client, collection),
            )
            if not results or not results[0]:
                raise NoStrategyError(
                    f"No strategy found: collection={collection} hard_filter={enc.hard_filter} k={self._k}"
                )
            results_row = _cosine_sim_to_distance(results[0])
            self._cache.put(cache_key, results_row)

        # Preflop raises snap to sized labels (open/3bet/4bet) from THIS spot's
        # pot_type + position — the neighbor only votes the stem.
        preflop_ctx = None
        if is_preflop:
            preflop_ctx = {
                "pot_type": enc.hard_filter.get("pot_type"),
                "hero_pos_rel": enc.hard_filter.get("hero_pos_rel"),
                "hero_pos": str(getattr(gs, "hero_position", "") or "") or None,
            }
        blended = blend_distributions(results_row, collection=collection, preflop_ctx=preflop_ctx)
        facing = float(getattr(gs, "hero_facing_bet_bb", 0) or 0)
        return sample_action(legalize_actions(blended, facing), self._rng), results_row

    @property
    def cache(self) -> LRUResultCache:
        """Expose the LRU cache for instrumentation / warmup verification."""
        return self._cache


def _milvus_client_from_env():
    """Build MilvusClient from env vars via src.db.milvus.connect (ERR-01 fail-loud).

    Reads MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN; constructs a `root:<token>`
    auth string when the token is not already in `user:pass` form.
    """
    from src.db.milvus import connect

    host = os.environ["MILVUS_HOST"]
    port = os.environ["MILVUS_PORT"]
    raw = os.environ.get("MILVUS_TOKEN")
    token = raw if (raw and ":" in raw) else (f"root:{raw}" if raw else None)
    return connect(f"http://{host}:{port}", token=token)


def engine_from_env(
    *,
    preflop_manifest: Path | None = None,
    postflop_manifest: Path | None = None,
    k: int = _DEFAULT_K,
    rng_seed: int | None = None,
    warmup: bool = True,
    as_of_version: int | None = None,
) -> KNNDecisionEngine:
    """Factory: build engine from env vars + default manifest paths.

    Steps:
        1. Construct MilvusClient via fail-loud connect().
        2. load_collection() for both DP collections (warms Milvus-side caches).
        3. Build engine.
        4. If warmup=True: call decide() for each WARMUP_PREFLOP_STATES entry
           to pre-populate the LRU cache. Warmup errors are logged + swallowed
           (best-effort — never block boot on a single bad warmup state).

    Args:
        preflop_manifest: path override; defaults to env PREFLOP_MANIFEST or
            "tools/zscore_preflop.json".
        postflop_manifest: same for postflop.
        k: kNN limit.
        rng_seed: numpy RNG seed for sample_action.
        warmup: if True, run WARMUP_PREFLOP_STATES through decide() once each.
        as_of_version: corpus version to pin; resolved once to a cutoff_ts.
    """
    if preflop_manifest is None:
        preflop_manifest = Path(os.environ.get("PREFLOP_MANIFEST", "tools/zscore_preflop.json"))
    if postflop_manifest is None:
        postflop_manifest = Path(os.environ.get("POSTFLOP_MANIFEST", "tools/zscore_postflop.json"))
    as_of_ts = corpus_versions.resolve_cutoff_ts(as_of_version) if as_of_version is not None else None
    client = _milvus_client_from_env()
    client.load_collection("preflop_decisions")
    client.load_collection("postflop_decisions")
    engine = KNNDecisionEngine(
        client,
        preflop_manifest,
        postflop_manifest,
        k=k,
        rng_seed=rng_seed,
        as_of_ts=as_of_ts,
    )

    if warmup:
        warmed = 0
        for gs in WARMUP_PREFLOP_STATES:
            try:
                engine.decide(gs)
                warmed += 1
            except Exception as e:
                log.warning(
                    "engine.warmup_failed",
                    error=str(e),
                    state_pos=gs.hero_position,
                )
        log.info(
            "engine.warmup_complete",
            warmed=warmed,
            cache_size=engine.cache.size,
        )
    return engine
