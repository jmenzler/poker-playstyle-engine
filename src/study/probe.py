"""Probe panel backend (D-NEW-24).

Two entry points:
    * ``probe_by_cluster_key(cluster_key)``  — Mode A
    * ``probe_by_spot(spot_dict)``           — Mode B
                                                (encode -> dispatch to Mode A)

Mode B encoding MUST reuse the frozen z-score manifests
(``tools/zscore_preflop.json`` / ``tools/zscore_postflop.json``).  This module
lazy-imports the encoder so unit tests don't need to load it.

Returns sections per UI-SPEC v2.2 §Probe response:
    * ``engine_response`` — source, n_obs, action_dist, ev_loss, ci_low,
                            ci_high, flagged_sparse, max_neighbor_dist
    * ``knn_neighbors``   — list of ≤k neighbors with cluster_key, distance,
                            action_dist
    * ``solver_truth``    — None when no cached solver row, else
                            {action_dist, kl_actual_vs_solver}
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from src._errors import NoStrategyError
from src._log import get_logger
from src.db import milvus as milvus_db
from src.db import timescale
from src.decision_engine.blending import blend_distributions
from src.decision_engine.preflop_chart import PreflopChartStrategy
from src.decision_engine.sparseness import compute_flagged_sparse, compute_max_neighbor_distance

log = get_logger("study.probe")

_PREFLOP_CHART: PreflopChartStrategy | None = None
_PREFLOP_CHART_LOADED = False


def _preflop_chart() -> PreflopChartStrategy | None:
    """Lazily load the chart tree once (env PREFLOP_CHART_SET, default tree)."""
    global _PREFLOP_CHART, _PREFLOP_CHART_LOADED
    if not _PREFLOP_CHART_LOADED:
        _PREFLOP_CHART_LOADED = True
        chart_dir = os.environ.get("PREFLOP_CHART_SET", "charts/preflop/inferred_6max")
        try:
            _PREFLOP_CHART = PreflopChartStrategy.from_dir(chart_dir)
        except Exception as e:
            log.warning("study.probe.chart_load_failed", error=str(e))
            _PREFLOP_CHART = None
    return _PREFLOP_CHART


def _preflop_chart_response(spot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Engine_response from the chart lookup (source=preflop_chart), or None on miss.

    Mirrors KNNDecisionEngine's preflop chart short-circuit so the probe panel
    matches the live engine. Reconstructs the GameState + hard_filter from the spot.
    """
    if not spot or spot.get("street") != "preflop":
        return None
    chart = _preflop_chart()
    if chart is None:
        return None
    from src.canonicalizer import Canonicalizer
    from tools.build_embedding import _gamestate_from_spot

    gs = _gamestate_from_spot(spot)
    enc = Canonicalizer.default().encode(gs)
    dist = chart.lookup(gs, enc.hard_filter)
    if dist is None:
        return None
    action_dist = {a: round(p, 4) for a, p in dist.items() if p > 0}
    return {
        "source": "preflop_chart",
        "action_dist": action_dist,
        "n_obs": 0,
        "ev_loss": None,
        "ci_low": None,
        "ci_high": None,
        "flagged_sparse": False,
        "max_neighbor_dist": 0.0,
    }


def _load_zscore_manifest(collection: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mean/std/weights for the given collection, mirroring the build path.

    mean/std come from the frozen manifest's per-dimension ``dim_stats`` list; weights
    come from the same build_*_weight_vector() helpers used at upsert time, NOT from the
    manifest (the manifest stores no weights). Build and serve must share both transforms
    or COSINE distance is computed across mismatched spaces.
    """
    from tools.upsert_milvus import build_postflop_weight_vector, build_preflop_weight_vector

    is_preflop = "preflop" in collection
    stem = "zscore_preflop" if is_preflop else "zscore_postflop"
    path = Path(__file__).resolve().parent.parent.parent / "tools" / f"{stem}.json"
    data = json.loads(path.read_text())
    dim_stats = data["dim_stats"]
    mean = np.array([d["mean"] for d in dim_stats], dtype=np.float64)
    std = np.array([d["std"] for d in dim_stats], dtype=np.float64)
    weights = build_preflop_weight_vector() if is_preflop else build_postflop_weight_vector()
    return mean, std, weights


def _encode_embedding_for_spot(spot: dict) -> np.ndarray:
    """Return the raw float64 embedding vector for spot via the cached Canonicalizer."""
    from tools.build_embedding import encode_spot_to_embedding

    return encode_spot_to_embedding(spot)


def probe_by_cluster_key(
    cluster_key: str,
    *,
    k: int = 5,
    spot: dict | None = None,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
) -> dict:
    """Mode A: query engine for a known cluster_key.

    Returns three sections per UI-SPEC v2.2 §Probe response.  When the
    cluster_key has no active row in strategy_nodes, ``engine_response``
    returns the empty-sentinel shape (all None / 0) and ``knn_neighbors``
    is an empty list.

    ``spot`` is the optional spot dict — when provided, it enables the
    no-node kNN encode fallback in ``_knn_neighbors`` (Mode B forwards it).
    """
    log.info("study.probe.cluster_key.started", cluster_key=cluster_key)
    own_tsdb = _tsdb_conn is None
    own_milvus = _milvus is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    client = _milvus if _milvus is not None else milvus_db.connect_from_env()
    try:
        engine_response = _engine_response(conn, cluster_key)
        knn_neighbors = _knn_neighbors(conn, client, cluster_key, spot=spot, k=k)
        # Lockstep with the live engine: a covered preflop spot is served by the
        # chart, not the kNN blend. Try the chart before falling back to the blend.
        chart_resp = (
            _preflop_chart_response(spot)
            if engine_response.get("source") is None and "street_class=preflop" in cluster_key
            else None
        )
        if chart_resp is not None:
            engine_response = chart_resp
        elif engine_response.get("source") is None and knn_neighbors:
            is_preflop = "street_class=preflop" in cluster_key
            collection = "preflop_decisions" if is_preflop else "postflop_decisions"
            preflop_ctx = None
            if is_preflop:
                hf = _hard_filter_from_cluster_key(cluster_key)
                preflop_ctx = {
                    "pot_type": hf.get("pot_type"),
                    "hero_pos_rel": hf.get("hero_pos_rel"),
                    "hero_pos": (spot or {}).get("hero_position"),
                }
            blended = _engine_response_from_knn(knn_neighbors, collection, preflop_ctx=preflop_ctx)
            if blended is not None:
                # Drop actions the facing state makes illegal (e.g. fold when checked-to) —
                # the coarse cluster_key partition doesn't encode facing-action, so the raw
                # blend can surface impossible actions. Postflop only; preflop legality differs.
                if spot is not None and spot.get("street") in _POSTFLOP_STREETS:
                    blended["action_dist"] = _filter_illegal_actions(
                        blended["action_dist"], _facing_bet_from_spot(spot)
                    )
                engine_response = blended
        solver_truth = _solver_truth(conn, cluster_key, engine_response.get("action_dist"))
        return {
            "engine_response": engine_response,
            "knn_neighbors": knn_neighbors,
            "solver_truth": solver_truth,
        }
    finally:
        if own_tsdb:
            with contextlib.suppress(Exception):
                conn.close()
        if own_milvus and hasattr(client, "close"):
            with contextlib.suppress(Exception):
                client.close()


def probe_by_spot(
    spot: dict,
    *,
    k: int = 5,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
) -> dict:
    """Mode B: encode a spot dict via existing pipeline -> dispatch to Mode A.

    Thin delegator to ``probe_by_cluster_key`` — does NOT eagerly acquire
    connections (CR-03 fix). When callers patch ``probe_by_cluster_key`` in
    tests, no DSN env vars are read and no real DB/Milvus handles open.

    Mode B encoding reuses ``tools/build_embedding.encode_spot_to_cluster_key``
    (or equivalent) — do NOT refit z-score manifests.  Imports are lazy so
    unit tests can mock the encoder.
    """
    log.info("study.probe.spot.started", street=spot.get("street"))
    from tools.build_embedding import encode_spot_to_cluster_key

    cluster_key = encode_spot_to_cluster_key(spot)
    log.info("study.probe.spot.encoded", cluster_key=cluster_key)
    return probe_by_cluster_key(
        cluster_key,
        k=k,
        spot=spot,
        _tsdb_conn=_tsdb_conn,
        _milvus=_milvus,
    )


# -----------------------------------------------------------------------------
# Section builders
# -----------------------------------------------------------------------------


_EMPTY_ENGINE_RESPONSE = {
    "source": None,
    "action_dist": None,
    "n_obs": 0,
    "ev_loss": None,
    "ci_low": None,
    "ci_high": None,
    "flagged_sparse": None,
    "max_neighbor_dist": None,
}


def _engine_response(conn: Any, cluster_key: str) -> dict:
    """Build the ``engine_response`` section.  Returns the empty sentinel
    when no active strategy_nodes row exists for the cluster_key."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source, action_dist FROM strategy_nodes WHERE cluster_key = %s AND active = TRUE LIMIT 1",
            (cluster_key,),
        )
        row = cur.fetchone()
    if row is None:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM observations WHERE cluster_key = %s", (cluster_key,))
            n = int(cur.fetchone()[0])
        reason = "no_observations" if n == 0 else "no_strategy_node"
        return {**_EMPTY_ENGINE_RESPONSE, "empty_reason": reason}
    source, action_dist = row
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM observations WHERE cluster_key = %s", (cluster_key,))
        n_obs = int(cur.fetchone()[0])
        cur.execute(
            "SELECT bool_or(flagged_sparse), MAX(max_neighbor_distance) "
            "FROM observations WHERE cluster_key = %s",
            (cluster_key,),
        )
        flagged_row = cur.fetchone()
        flagged, max_dist = flagged_row if flagged_row is not None else (None, None)
        cur.execute(
            "SELECT AVG(value) FROM metrics WHERE cluster_key = %s AND metric_name = 'ev_loss'",
            (cluster_key,),
        )
        # AVG over zero rows still returns one row whose value is NULL; the
        # is-not-None guard belongs on the column, not the row.
        (ev,) = cur.fetchone()
    return {
        "source": source,
        "action_dist": action_dist,
        "n_obs": n_obs,
        "ev_loss": float(ev) if ev is not None else None,
        "ci_low": None,
        "ci_high": None,  # CI populated by Stage A EB if caller wants
        "flagged_sparse": bool(flagged) if flagged is not None else None,
        "max_neighbor_dist": float(max_dist) if max_dist is not None else None,
    }


_KNN_OUTPUT_FIELDS = [
    "decision_id",
    "cluster_key",
    "street_class",
    "pot_type",
    "hero_pos_rel",
    "n_players_active",
    "hero_action_type",
    "gto_score",
    "confidence",
    "hero_action_size_pot_frac",
    "raise_ratio",
    "hero_action_allin",
    "action_dist",
]

_SCHEMA_FIELD_CACHE: dict[str, set[str]] = {}


def _collection_field_names(client: Any, collection: str) -> set[str]:
    """Field names in a Milvus collection, cached. Empty set on describe failure."""
    cached = _SCHEMA_FIELD_CACHE.get(collection)
    if cached is not None:
        return cached
    try:
        desc = client.describe_collection(collection)
        names = {f["name"] for f in desc["fields"]}
    except Exception as e:
        # Don't cache failures — a transient describe error must not stick.
        log.warning("study.probe.knn.describe_failed", collection=collection, error=str(e))
        return set()
    _SCHEMA_FIELD_CACHE[collection] = names
    return names


def _hard_filter_from_cluster_key(cluster_key: str) -> dict[str, Any]:
    """Parse a cluster_key (``key=val|...``) into the hard_filter dict.

    n_players_active casts to int so _build_filter emits it unquoted (Milvus Int).
    """
    hf: dict[str, Any] = {}
    for part in cluster_key.split("|"):
        key, sep, val = part.partition("=")
        if not sep:
            continue
        hf[key] = int(val) if (key == "n_players_active" and val.isdigit()) else val
    return hf


def _knn_neighbors(
    conn: Any, client: Any, cluster_key: str, *, spot: dict | None = None, k: int = 5
) -> list[dict]:
    """Top-k Milvus neighbors with action_dist preview."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM strategy_nodes WHERE cluster_key = %s AND active = TRUE LIMIT 1",
            (cluster_key,),
        )
        row = cur.fetchone()
    if row is None:
        if spot is None:
            return []
        raw = _encode_embedding_for_spot(spot)
    else:
        raw = np.asarray(row[0], dtype=np.float64)
    collection = "preflop_decisions" if "street_class=preflop" in cluster_key else "postflop_decisions"
    mean, std, weights = _load_zscore_manifest(collection)
    std_safe = np.where(std == 0, 1.0, std)
    normalized = ((raw - mean) / std_safe) * weights
    embedding_to_search = normalized.tolist()
    # Milvus rejects the whole search if any output_field is absent (code=65535), and
    # the corpus has no cluster_key field — request only live-schema fields, reconstruct.
    available = _collection_field_names(client, collection)
    if available:
        output_fields = [f for f in _KNN_OUTPUT_FIELDS if f in available]
    else:
        output_fields = [f for f in _KNN_OUTPUT_FIELDS if f != "cluster_key"]
    # Hard-partition on the cluster_key components like the live engine, not a global
    # COSINE search (which surfaced limp neighbors for a 3bet spot). spr band omitted.
    from src.decision_engine.engine import _build_filter

    hard_filter = _hard_filter_from_cluster_key(cluster_key)
    spot_features = {"street": str(spot["street"]).lower()} if spot and spot.get("street") else None
    filter_expr = _build_filter(hard_filter, include_active=True, spot_features=spot_features)
    try:
        results = client.search(
            collection_name=collection,
            data=[embedding_to_search],
            limit=k,
            anns_field="embedding",
            filter=filter_expr,
            search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            output_fields=output_fields,
        )
    except Exception as e:
        log.warning("study.probe.knn.search_failed", collection=collection, error=str(e))
        return []
    out: list[dict] = []
    for hit in results[0] if results else []:
        e = hit.entity
        nbr_cluster_key = _reconstruct_cluster_key_from_entity(e)
        sn_row = _lookup_strategy_node(conn, nbr_cluster_key)
        if sn_row is not None:
            _hero_action_type, action_dist = sn_row
        else:
            action = e.get("hero_action_type")
            action_dist = {action: 1.0} if action else None
        # decision_id ("{hand_id}_dp{idx}") carries the hit's hand_id → replayable via
        # HM3 by-hand. observations is empty on the HM path, so don't source it there.
        decision_id = e.get("decision_id")
        corpus_hand_id = decision_id.rsplit("_dp", 1)[0] if decision_id else None
        obs_row = _lookup_observation(conn, nbr_cluster_key)
        if obs_row is not None:
            obs_id, hand_id = obs_row
        else:
            obs_id, hand_id = None, corpus_hand_id
        n_obs = _count_observations(conn, nbr_cluster_key)
        out.append(
            {
                "cluster_key": nbr_cluster_key,
                "decision_id": decision_id,
                # pymilvus returns the COSINE score as similarity (1.0=identical);
                # convert to distance (0.0=identical) so smaller=closer, like the engine.
                "distance": 1.0 - float(hit.distance),
                "action_dist": action_dist,
                "obs_id": str(obs_id) if obs_id else None,
                "hand_id": hand_id,
                "n_obs": n_obs,
                # raw fields the engine-blend fallback consumes (frontend ignores them)
                "hero_action_type": e.get("hero_action_type"),
                "gto_score": e.get("gto_score"),
                "confidence": e.get("confidence"),
                "hero_action_size_pot_frac": e.get("hero_action_size_pot_frac"),
                "raise_ratio": e.get("raise_ratio"),
                "hero_action_allin": e.get("hero_action_allin"),
            }
        )
    return out


def _engine_response_from_knn(
    neighbors: list[dict], collection: str, *, preflop_ctx: dict | None = None
) -> dict | None:
    """The engine's decision for a spot with no strategy_nodes override: the kNN
    blend over the retrieved neighbors — the exact mechanism KNNDecisionEngine uses.

    Returns the engine_response shape (source='knn_blend'), or None if the blend
    has zero weight (NoStrategyError) so the caller keeps the empty sentinel.
    """
    hits = [
        {
            "distance": n["distance"],
            "entity": {
                "hero_action_type": n.get("hero_action_type"),
                "confidence": n.get("confidence") if n.get("confidence") is not None else 1.0,
                "gto_score": n.get("gto_score") if n.get("gto_score") is not None else 1.0,
                "hero_action_size_pot_frac": n.get("hero_action_size_pot_frac"),
                "raise_ratio": n.get("raise_ratio"),
                "hero_action_allin": n.get("hero_action_allin"),
                "action_dist": n.get("action_dist"),
            },
        }
        for n in neighbors
    ]
    try:
        blended = blend_distributions(hits, collection=collection, preflop_ctx=preflop_ctx)
    except NoStrategyError:
        return None
    action_dist = {a: round(p, 4) for a, p in blended.items() if p > 0}
    return {
        "source": "knn_blend",
        "action_dist": action_dist,
        "n_obs": len(neighbors),
        "ev_loss": None,
        "ci_low": None,
        "ci_high": None,
        "flagged_sparse": compute_flagged_sparse(hits),
        "max_neighbor_dist": compute_max_neighbor_distance(hits),
    }


_POSTFLOP_STREETS = {"flop", "turn", "river"}


def _facing_bet_from_spot(spot: dict) -> bool:
    """Whether hero faces a live wager at this decision (vs first-to-act / checked-to).

    Read from the last action_sequence token — the action hero responds to. A trailing
    bet/raise/allin means a wager is pending; check/call (street closed → hero first on
    the next street) or an empty sequence means none.
    """
    seq = spot.get("action_sequence") or []
    if not seq:
        return False
    last = seq[-1]
    verb = (last.split(":", 1)[1] if ":" in last else last).lower()
    return verb == "allin" or verb.startswith("bet") or verb.startswith("raise")


def _filter_illegal_actions(action_dist: dict[str, float], facing_bet: bool) -> dict[str, float]:
    """Drop actions illegal for the facing state, then renormalize.

    Facing a wager: legal = fold / call / raise* / allin.  First-to-act or checked-to:
    legal = check / bet* / allin.  If the blend put all weight on illegal actions
    (degenerate retrieval for this node), fall back to the passive legal action
    rather than surfacing an impossible one.
    """

    def _legal(action: str) -> bool:
        a = action.lower()
        if a == "allin":
            return True
        if facing_bet:
            return a in ("fold", "call") or a.startswith("raise")
        return a == "check" or a.startswith("bet")

    kept = {a: p for a, p in action_dist.items() if _legal(a)}
    total = sum(kept.values())
    if total <= 0:
        return {"fold": 1.0} if facing_bet else {"check": 1.0}
    return {a: round(p / total, 4) for a, p in kept.items()}


def _reconstruct_cluster_key_from_entity(entity: Any) -> str:
    """Build a cluster_key string from a Milvus search-hit entity.

    Preferred path: entity exposes a single 'cluster_key' field (the canonical
    contract per CR-01). Fallback: reconstruct from the four hard_filter scalars
    (street_class/pot_type/hero_pos_rel/n_players_active) for collections that
    have not been re-upserted with the cluster_key field yet. Output matches
    src.sim.harness._cluster_key_from_filter byte-for-byte.
    """
    ck = entity.get("cluster_key")
    if ck:
        return str(ck)
    hard_filter = {
        "street_class": entity.get("street_class"),
        "pot_type": entity.get("pot_type"),
        "hero_pos_rel": entity.get("hero_pos_rel"),
        "n_players_active": entity.get("n_players_active"),
    }
    return "|".join(f"{k}={v}" for k, v in sorted(hard_filter.items()))


def _lookup_strategy_node(conn: Any, cluster_key: str) -> tuple[str, dict] | None:
    """Read-only lookup: (hero_action_type, action_dist) for an active strategy_node."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source, action_dist FROM strategy_nodes WHERE cluster_key = %s AND active = TRUE LIMIT 1",
            (cluster_key,),
        )
        return cur.fetchone()


def _lookup_observation(conn: Any, cluster_key: str) -> tuple[Any, str] | None:
    """Read-only lookup: one (obs_id, session_id) row for a cluster_key, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT obs_id, session_id FROM observations WHERE cluster_key = %s LIMIT 1",
            (cluster_key,),
        )
        return cur.fetchone()


def _count_observations(conn: Any, cluster_key: str) -> int:
    """Read-only count of observations rows for a cluster_key."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM observations WHERE cluster_key = %s",
            (cluster_key,),
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _solver_truth(conn: Any, cluster_key: str, actual_dist: dict | None) -> dict | None:
    """Cached solver_truth row + KL(actual || solver).  ``None`` if no cache."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT action_dist FROM strategy_nodes "
            "WHERE cluster_key = %s AND source IN ('solver','solver_verify') AND active = TRUE LIMIT 1",
            (cluster_key,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    solver_dist = row[0]
    kl = _kl_divergence(actual_dist, solver_dist) if actual_dist else None
    return {"action_dist": solver_dist, "kl_actual_vs_solver": kl}


def _kl_divergence(p: dict | None, q: dict | None, eps: float = 1e-6) -> float | None:
    """KL(P || Q) with epsilon smoothing per Phase 4 metrics convention."""
    if p is None or q is None:
        return None
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        pi = max(float(p.get(k, 0.0)), eps)
        qi = max(float(q.get(k, 0.0)), eps)
        s += pi * math.log(pi / qi)
    return s


def probe_range_for_actor(
    spot: dict,
    actor_position: str,
    *,
    _tsdb_conn: Any = None,
    _milvus: Any = None,
) -> dict:
    """Encode all 169 starting hands for actor_position at its decision node."""
    from tools.build_embedding import encode_spot_to_cluster_key
    from tools.build_preflop_equity_table import ALL_HAND_CLASSES, _villain_combos_for_class

    own_tsdb = _tsdb_conn is None
    own_milvus = _milvus is None
    conn = _tsdb_conn if _tsdb_conn is not None else timescale.connect(_tsdb_dsn_from_env())
    client = _milvus if _milvus is not None else milvus_db.connect_from_env()
    try:
        actor_spot = dict(spot)
        actor_spot["hero_position"] = actor_position

        # Blockers: board cards + the actor's own hole cards (if the caller pinned them).
        # Empty for preflop spots with no hero_hole pin; nontrivial for turn/river.
        board_cards = set(spot.get("board", []) or [])
        hero_hole_cards = set(spot.get("hero_hole", []) or [])
        blocked = board_cards | hero_hole_cards

        hands: list[dict] = []
        legend: dict[str, float] = {}
        nonfold_combos = 0.0
        total_combos = 0  # combo-denominator drops below 1326 when blockers are present
        uncovered_combos = 0  # classes the engine has no strategy for (distinct from fold)

        for cls in ALL_HAND_CLASSES:
            combos = _villain_combos_for_class(cls, blocked)
            combo_count = len(combos) if combos else 0
            total_combos += combo_count
            rep = combos[0] if combos else None

            if rep is None:
                hands.append({"hand_class": cls, "action_dist": None, "combos": combo_count})
                continue

            spot_i = dict(actor_spot)
            spot_i["hero_hole"] = rep
            cluster_key = encode_spot_to_cluster_key(spot_i)

            resp = probe_by_cluster_key(
                cluster_key,
                k=5,
                spot=spot_i,
                _tsdb_conn=conn,
                _milvus=client,
            )
            action_dist = resp["engine_response"].get("action_dist")
            hands.append({"hand_class": cls, "action_dist": action_dist, "combos": combo_count})

            if action_dist is not None:
                fold_p = float(action_dist.get("fold", 0.0))
                nonfold_combos += combo_count * (1.0 - fold_p)
                for action, prob in action_dist.items():
                    legend[action] = legend.get(action, 0.0) + combo_count * float(prob)
            else:
                uncovered_combos += combo_count

        legend_int = {a: round(v) for a, v in legend.items()}
        aggregate_range_pct = nonfold_combos / total_combos if total_combos > 0 else 0.0

        return {
            "hands": hands,
            "aggregate_range_pct": aggregate_range_pct,
            "uncovered_combos": uncovered_combos,
            "legend": legend_int,
        }
    finally:
        if own_tsdb:
            with contextlib.suppress(Exception):
                conn.close()
        if own_milvus and hasattr(client, "close"):
            with contextlib.suppress(Exception):
                client.close()


def _tsdb_dsn_from_env() -> str:
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"
