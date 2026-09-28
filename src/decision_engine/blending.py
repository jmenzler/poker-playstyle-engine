"""Blend + sample for the kNN DecisionEngine (ENGN-03).  long-ok

A patch/solver row carries a full action_dist (JSON) and the blend splits that
row's weight proportionally across its actions; a base-corpus row has no
action_dist and contributes its single hero_action_type as a one-hot. Per-row:

    weight = similarity * confidence * gto_score

where similarity = max(0, 1 - cosine_distance). Action labels not in
CANONICAL_ACTIONS are skipped so output mass stays over the canonical vocab. If
every weight is zero, raise NoStrategyError — the caller (KNNDecisionEngine)
handles fallback. Logs a warning only when an action_dist field fails to parse.
"""

from __future__ import annotations

import json

import numpy as np

from src._errors import NoStrategyError
from src._log import get_logger
from tools.action_buckets import preflop_action_label, snap_postflop_action

log = get_logger("decision_engine.blending")

# The 15-action vocab. Order is locked by PRD §Action set and used by SimAdapter
# (Plan 03) when mapping these strings into RLCard's discrete action space.
CANONICAL_ACTIONS: tuple[str, ...] = (
    "check",
    "fold",
    "call",
    # preflop sized raises (snap-at-serve from stem + query spot)
    "open_2_2bb",
    "open_3bb",
    "3bet_3x",
    "3bet_4x",
    "4bet_2_5x",
    # postflop sized bets/raises (still alias-collapsed until the postflop re-ingest)
    "bet_25",
    "bet_33",
    "bet_50",
    "bet_75",
    "bet_100",
    "bet_150",
    "bet_overbet",
    "raise_min",
    "raise_2_5x",
    "raise_3x",
    "raise_pot",
    "allin",
)


# Postflop stems stay collapsed until their re-ingest adds raw sizes. Preflop is
# NOT aliased here — it snaps via action_buckets.preflop_action_label (query spot).
_GENERIC_LABEL_ALIAS: dict[str, str] = {"raise": "raise_2_5x", "bet": "bet_50"}


def _snap(action: str, entity: dict, preflop_ctx: dict | None) -> str:
    """Map a neighbor action label to a canonical bucket (preflop vs postflop path)."""
    if preflop_ctx is not None:
        return preflop_action_label(action, **preflop_ctx)
    return snap_postflop_action(
        action,
        entity.get("hero_action_size_pot_frac"),
        entity.get("raise_ratio"),
        bool(entity.get("hero_action_allin", False)),
    )


def _parse_action_dist(raw: object) -> dict[str, float] | None:
    """Decode a Milvus action_dist VARCHAR (JSON) into a dict, or None if absent/bad."""
    if not raw:
        return None
    try:
        loaded = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(loaded, dict) and loaded:
            return {str(k): float(v) for k, v in loaded.items()}
    except (ValueError, TypeError) as exc:
        log.warning("blending.parse_action_dist_failed", raw_type=type(raw).__name__, error=str(exc))
        return None
    return None


def blend_distributions(
    search_results: list[dict],
    *,
    collection: str = "<unknown>",
    preflop_ctx: dict | None = None,
) -> dict[str, float]:
    """Weighted blend of kNN neighbor hero_action_type labels.

    ENGN-03 formula: weight = similarity * confidence * gto_score, where
    similarity = max(0, 1 - cosine_distance). Confidence and gto_score
    default to 1.0 if absent on the neighbor entity (defensive — Plan 01
    schema additions guarantee DP rows carry these, but pre-Phase-3
    chunks may legitimately default).

    Args:
        search_results: list of pymilvus search-result hit dicts. Each hit
            has 'distance' (float) and 'entity' (dict with at least
            'hero_action_type' plus optional 'confidence', 'gto_score').
        collection: Name of the Milvus collection that produced these hits
            (for diagnostic inclusion in NoStrategyError messages).

    Returns:
        dict[action_str -> probability]. Always contains all 15 canonical
        action keys; values sum to 1.0 +/- 1e-9.

    Raises:
        NoStrategyError: if total weight across all neighbors is zero
            (all neighbors had non-canonical labels, or all similarities
            were zero, or all confidence/gto_score were zero).
    """
    acc: dict[str, float] = dict.fromkeys(CANONICAL_ACTIONS, 0.0)
    total_weight = 0.0
    for hit in search_results:
        distance = float(hit.get("distance", 1.0))
        similarity = max(0.0, 1.0 - distance)
        entity = hit.get("entity", {})
        confidence = float(entity.get("confidence", 1.0))
        gto_score = float(entity.get("gto_score", 1.0))
        weight = similarity * confidence * gto_score
        if weight <= 0.0:
            continue

        # Split the row's weight across action_dist (patch/solver rows) so the row
        # contributes `weight` once; base rows lack it and fall back to the label.
        parsed = _parse_action_dist(entity.get("action_dist"))
        contributed = False
        if parsed is not None:
            for raw_action, prob in parsed.items():
                if prob <= 0.0:
                    continue
                snapped = _snap(raw_action, entity, preflop_ctx)
                if snapped not in acc:
                    continue
                acc[snapped] += weight * prob
                contributed = True
        else:
            snapped = _snap(entity.get("hero_action_type", ""), entity, preflop_ctx)
            if snapped in acc:
                acc[snapped] += weight
                contributed = True

        if contributed:
            total_weight += weight

    if total_weight == 0.0:
        raise NoStrategyError(
            f"All retrieved neighbors had zero blend weight "
            f"(collection={collection}, n_neighbors={len(search_results)})"
        )

    return {action: w / total_weight for action, w in acc.items()}


# Remap a voted verb to the node's legal family: a raise-of-zero clamps to a degenerate
# 1bb min-bet, so facing no bet a raise becomes a bet of matching size (and vice versa).
_REMAP_UNFACED: dict[str, str] = {
    "fold": "check",
    "call": "check",
    "raise_min": "bet_25",
    "raise_2_5x": "bet_50",
    "raise_3x": "bet_75",
    "raise_pot": "bet_100",
}
_REMAP_FACED: dict[str, str] = {
    "check": "call",
    "bet_25": "raise_min",
    "bet_33": "raise_min",
    "bet_50": "raise_2_5x",
    "bet_75": "raise_3x",
    "bet_100": "raise_pot",
    "bet_150": "raise_pot",
    "bet_overbet": "raise_pot",
}


def legalize_actions(dist: dict[str, float], facing_bet_bb: float) -> dict[str, float]:
    """Remap each voted verb to the node's legal vocab, then renormalize, before sampling.

    Facing no bet: fold/call -> check, raise_X -> bet of the matching size. Facing a bet:
    check -> call, bet_X -> raise of the matching size. Verbs already legal pass through.
    """
    remap = _REMAP_UNFACED if facing_bet_bb <= 0 else _REMAP_FACED
    out: dict[str, float] = {}
    for action, prob in dist.items():
        key = remap.get(action, action)
        out[key] = out.get(key, 0.0) + prob
    total = sum(out.values())
    if total <= 0:
        return {"check" if facing_bet_bb <= 0 else "call": 1.0}
    return {action: prob / total for action, prob in out.items()}


def sample_action(dist: dict[str, float], rng: np.random.Generator) -> str:
    """Sample one action from a probability distribution dict.

    Renormalizes probs defensively (float-safety) so callers passing
    distributions with rounding drift still get a valid sample.

    Args:
        dist: dict[action_str -> probability]. Probabilities should be
            non-negative; zero entries are allowed.
        rng: numpy random Generator (caller controls seed for determinism).

    Returns:
        Action string sampled from `dist` according to its probability mass.

    Raises:
        ValueError: if the total probability mass is <= 0.
    """
    actions = list(dist.keys())
    probs = np.array([dist[a] for a in actions], dtype=np.float64)
    total = probs.sum()
    if total <= 0.0:
        raise ValueError("sample_action: zero-mass distribution")
    probs /= total
    idx = rng.choice(len(actions), p=probs)
    return actions[idx]
