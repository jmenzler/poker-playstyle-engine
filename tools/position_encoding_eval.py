"""Offline LOO harness comparing postflop position encodings (no Milvus/prod/re-ingest).
Swaps only the position block per variant; scores leave-one-out cosine-kNN action
prediction within the production hard-filter partition. Full design + results in
docs/plans/2026-06-04-position-encoding-comparison-design.md.
"""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np

from tools.equity_lookup import EquityLookup
from tools.feature_extractors.postflop import (
    _POS_IDX,
    _group_a_postflop,
    _group_b,
    _group_c,
    _group_d_postflop,
    _group_e_postflop,
    _group_f_postflop,
    _group_g,
    _group_h,
    _group_j_postflop,
)

# ---- Position-encoding constants ---------------------------------------------

# Hand-specified open-order axis (tightest->widest opener). Blinds sit OFF the
# axis (postflop they are forced OOP) -> value 0.0, distinguished by blind flags.
OPEN_ORDER: dict[str, float] = {"UTG": 0.0, "MP": 0.2, "CO": 0.4, "BTN": 0.6, "SB": 0.0, "BB": 0.0}
BLINDS = frozenset({"SB", "BB"})

# Position-based opponent-range proxies (mirror _group_j_postflop's inline maps).
_TIGHTNESS = {"UTG": 1.0, "MP": 0.8, "CO": 0.6, "BTN": 0.4, "SB": 0.5, "BB": 0.5}
_BREADTH = {"UTG": 0.3, "MP": 0.4, "CO": 0.6, "BTN": 0.8, "SB": 0.5, "BB": 0.7}

COARSE_ACTIONS = ("fold", "check", "call", "bet", "raise")
_ACTION_IDX = {a: i for i, a in enumerate(COARSE_ACTIONS)}

# Non-position group weights (production values; A B C E F G H), position omitted.
_NON_POS_WEIGHTS = np.array(
    [1.2] * 12 + [1.2] * 8 + [1.2] * 6 + [1.8] * 7 + [1.8] * 16 + [0.8] * 6 + [0.7] * 7,
    dtype=np.float64,
)  # 62 dims


def _villain_pos(dp: dict) -> str | None:
    v = dp.get("facing_pos") or dp.get("preflop_aggressor")
    return str(v) if v else None


def _villain_was_pf_agg(dp: dict, vill: str | None) -> float:
    hero = dp.get("hero_pos", "")
    pf = dp.get("preflop_aggressor", "")
    return 1.0 if (pf and pf != hero and pf == vill) else 0.0


def _rel_pos_norm(hero: str, vill: str | None) -> float:
    hero_seat = _POS_IDX.get(hero, 0)
    vill_seat = _POS_IDX.get(vill, 0) if vill else 0
    return float(np.clip(((vill_seat - hero_seat) / 6.0 + 1.0) / 2.0, 0.0, 1.0))


def _n_players_norm(dp: dict) -> float:
    return float(np.clip(dp.get("n_players_at_street", 2) / 6.0, 0.0, 1.0))


# ---- Position blocks (one per variant) ---------------------------------------


def pos_block_baseline(dp: dict) -> np.ndarray:
    """18 dims: current one-hot encoding (Group D ++ Group J), reused verbatim."""
    return np.concatenate([_group_d_postflop(dp), _group_j_postflop(dp)]).astype(np.float32)


def pos_block_ordinal(dp: dict, seat_value: dict[str, float]) -> np.ndarray:
    """10 dims: hero/villain ordinal scalar (from seat_value) + blind flags + kept relatives.
    Open-order passes OPEN_ORDER; data-derived an empirically ranked map. Blinds carry
    separate flags so a seat_value of 0.0 for a blind is not confused with UTG.
    """
    hero = dp.get("hero_pos", "")
    vill = _villain_pos(dp)
    return np.array(
        [
            seat_value.get(hero, 0.0),
            seat_value.get(vill, 0.0) if vill else 0.0,
            1.0 if hero in BLINDS else 0.0,
            1.0 if vill in BLINDS else 0.0,
            1.0 if dp.get("hero_pos_rel") == "IP" else 0.0,
            _rel_pos_norm(hero, vill),
            _TIGHTNESS.get(str(vill), 0.5),
            _BREADTH.get(str(vill), 0.5),
            _n_players_norm(dp),
            _villain_was_pf_agg(dp, vill),
        ],
        dtype=np.float32,
    )


def pos_block_relationship(dp: dict) -> np.ndarray:
    """8 dims: pair dynamic with absolute seat dropped."""
    hero = dp.get("hero_pos", "")
    vill = _villain_pos(dp)
    hero_seat = _POS_IDX.get(hero, 0)
    vill_seat = _POS_IDX.get(vill, 0) if vill else 0
    seats_between = abs(hero_seat - vill_seat) / 5.0 if vill else 0.0
    range_adv = (_TIGHTNESS.get(str(vill), 0.5) - _TIGHTNESS.get(hero, 0.5) + 1.0) / 2.0
    return np.array(
        [
            1.0 if dp.get("hero_pos_rel") == "IP" else 0.0,
            float(np.clip(seats_between, 0.0, 1.0)),
            1.0 if hero in BLINDS else 0.0,
            1.0 if vill in BLINDS else 0.0,
            1.0 if (hero in BLINDS and vill in BLINDS) else 0.0,
            float(np.clip(range_adv, 0.0, 1.0)),
            _n_players_norm(dp),
            _villain_was_pf_agg(dp, vill),
        ],
        dtype=np.float32,
    )


# ---- Non-position block (identical across variants) --------------------------


def non_position_block(dp: dict, eq: EquityLookup, pop_mean: dict[str, float]) -> np.ndarray:
    """62 dims: groups A B C E F G H (everything except position D/J)."""
    street = str(dp.get("street", "flop"))
    return np.concatenate(
        [
            _group_a_postflop(dp, eq, pop_mean, street),
            _group_b(dp, street),
            _group_c(dp),
            _group_e_postflop(dp),
            _group_f_postflop(dp, street),
            _group_g(dp, street),
            _group_h(dp),
        ],
        dtype=np.float32,
    )


# ---- Data prep ---------------------------------------------------------------


def reconstruct_scenario_key(dp: dict) -> str:
    """Build a scenario_key from raw row fields so equity lookup hits the table.

    Format mirrors the equity table: pot_type|hero_rel|nN|villain:detail. Only the
    first three parts must match (prefix fallback covers the villain detail).
    """
    vill = _villain_pos(dp) or "NA"
    return (
        f"{dp.get('pot_type', '')}|{dp.get('hero_pos_rel', '')}|n{dp.get('n_players_at_street', 2)}|{vill}:x"
    )


def coarse_action(dp: dict) -> str | None:
    a = dp.get("hero_action_type", "")
    return a if a in _ACTION_IDX else None


def facing_bet(dp: dict) -> bool:
    return dp.get("facing") in ("bet_to", "raise_to")


def partition_key(dp: dict) -> tuple:
    return (
        "postflop",
        dp.get("pot_type", ""),
        dp.get("hero_pos_rel", ""),
        int(dp.get("n_players_at_street", 2)),
    )


def _load_all(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            dp = json.loads(line)
            if dp.get("street") in ("flop", "turn", "river") and coarse_action(dp) is not None:
                rows.append(dp)
    return rows


def load_sample(path: str, sample_n: int, seed: int) -> list[dict]:
    """Load postflop rows with a valid coarse action, then seeded-sample sample_n."""
    rows = _load_all(path)
    rng = random.Random(seed)
    if sample_n < len(rows):
        rows = rng.sample(rows, sample_n)
    return rows


def load_sample_and_holdout(path: str, sample_n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded eval sample + its disjoint complement (for leak-safe descriptor fitting)."""
    rows = _load_all(path)
    rng = random.Random(seed)
    if sample_n >= len(rows):
        return rows, rows  # tiny corpora: fall back to in-sample fit
    idx = set(rng.sample(range(len(rows)), sample_n))
    sample = [rows[i] for i in range(len(rows)) if i in idx]
    holdout = [rows[i] for i in range(len(rows)) if i not in idx]
    return sample, holdout


# ---- Data-driven position character (leak-safe: fit on holdout) ---------------

# A position's postflop range character (agg/fold/passive rates), measured per
# pot_type so blinds are handled right. Replaces hand-picked _TIGHTNESS/_BREADTH.

_CHAR_MIN_COUNT = 30  # below this, fall back to per-position then global


def fit_position_character(holdout: list[dict]) -> tuple:
    """(by_pot_pos, by_pos, global) each pos -> [agg_rate, fold_rate, passive_rate]."""

    def _acc(rows: list[dict], keyfn) -> dict:
        counts: dict = {}
        for dp in rows:
            a = coarse_action(dp)
            if a is None:
                continue
            c = counts.setdefault(keyfn(dp), [0, 0, 0, 0])  # agg, fold, passive, total
            if a in ("bet", "raise"):
                c[0] += 1
            elif a == "fold":
                c[1] += 1
            else:
                c[2] += 1
            c[3] += 1
        return counts

    def _rates(counts: dict) -> dict:
        return {
            k: np.array([c[0] / c[3], c[1] / c[3], c[2] / c[3]], dtype=np.float64)
            for k, c in counts.items()
            if c[3] > 0
        }

    by_pt_pos_raw = _acc(holdout, lambda dp: (dp.get("pot_type", ""), dp.get("hero_pos", "")))
    by_pos_raw = _acc(holdout, lambda dp: dp.get("hero_pos", ""))
    glob_c = [sum(x) for x in zip(*by_pos_raw.values(), strict=True)] if by_pos_raw else [1, 1, 1, 3]
    glob = np.array([glob_c[0] / glob_c[3], glob_c[1] / glob_c[3], glob_c[2] / glob_c[3]])
    return (
        {k: v for k, v in _rates(by_pt_pos_raw).items() if by_pt_pos_raw[k][3] >= _CHAR_MIN_COUNT},
        _rates(by_pos_raw),
        glob,
    )


def char_of(char: tuple, pot_type: str, pos: str | None) -> np.ndarray:
    """Range character for (pot_type, pos) with fallback chain -> per-pos -> global."""
    by_pt_pos, by_pos, glob = char
    if pos is None:
        return glob
    return by_pt_pos.get((pot_type, pos)) if (pot_type, pos) in by_pt_pos else by_pos.get(pos, glob)


def _char_strength(c: np.ndarray) -> float:
    """Net range strength scalar: agg_rate - fold_rate, in [-1, 1]."""
    return float(c[0] - c[1])


def pos_block_range_character(dp: dict, char: tuple) -> np.ndarray:
    """9 dims: measured hero+villain range character (agg/fold) + geometry + blind flags."""
    hero = dp.get("hero_pos", "")
    vill = _villain_pos(dp)
    pt = dp.get("pot_type", "")
    hc = char_of(char, pt, hero)
    vc = char_of(char, pt, vill)
    hero_seat = _POS_IDX.get(hero, 0)
    vill_seat = _POS_IDX.get(vill, 0) if vill else 0
    seats_between = abs(hero_seat - vill_seat) / 5.0 if vill else 0.0
    return np.array(
        [
            1.0 if dp.get("hero_pos_rel") == "IP" else 0.0,
            hc[0],
            hc[1],  # hero agg, fold
            vc[0],
            vc[1],  # villain agg, fold
            float(np.clip(seats_between, 0.0, 1.0)),
            1.0 if hero in BLINDS else 0.0,
            1.0 if vill in BLINDS else 0.0,
            _villain_was_pf_agg(dp, vill),
        ],
        dtype=np.float32,
    )


def pos_block_relationship_v2(dp: dict, char: tuple) -> np.ndarray:
    """8 dims: relationship structure + DATA-driven range advantage (replaces guessed tightness)."""
    hero = dp.get("hero_pos", "")
    vill = _villain_pos(dp)
    pt = dp.get("pot_type", "")
    hero_seat = _POS_IDX.get(hero, 0)
    vill_seat = _POS_IDX.get(vill, 0) if vill else 0
    seats_between = abs(hero_seat - vill_seat) / 5.0 if vill else 0.0
    range_adv = (
        _char_strength(char_of(char, pt, vill)) - _char_strength(char_of(char, pt, hero)) + 2.0
    ) / 4.0
    return np.array(
        [
            1.0 if dp.get("hero_pos_rel") == "IP" else 0.0,
            float(np.clip(seats_between, 0.0, 1.0)),
            1.0 if hero in BLINDS else 0.0,
            1.0 if vill in BLINDS else 0.0,
            1.0 if (hero in BLINDS and vill in BLINDS) else 0.0,
            float(np.clip(range_adv, 0.0, 1.0)),
            _n_players_norm(dp),
            _villain_was_pf_agg(dp, vill),
        ],
        dtype=np.float32,
    )


# ---- Normalization (mirror production order: z-score THEN weight) -------------


def fit_zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim mean/std over the sample; std<1e-6 -> 1.0 (matches tools/zscore_fit)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return ((x - mean) / std) * weights


def _unit(sub: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return sub / norms


def topk_cosine(unit: np.ndarray, k: int, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Memory-safe per-row top-k neighbors (self excluded) over a partition.

    Computes cosine in row-chunks instead of one m*m matrix so 25k-row partitions
    fit in RAM. Returns (idx[m,kk], sim[m,kk]) — neighbor indices and their cosines.
    """
    m = unit.shape[0]
    kk = min(k, m - 1)
    idx = np.empty((m, kk), dtype=np.int64)
    val = np.empty((m, kk), dtype=np.float64)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        sims = unit[s:e] @ unit.T  # (chunk, m)
        rows_idx = np.arange(e - s)
        sims[rows_idx, s + rows_idx] = -np.inf  # exclude self
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        idx[s:e] = part
        val[s:e] = np.take_along_axis(sims, part, axis=1)
    return idx, val


# ---- Metrics -----------------------------------------------------------------


def _legal_mask(facing: bool) -> np.ndarray:
    """Boolean mask over COARSE_ACTIONS for the facing state (allin folded into raise/bet)."""
    legal = {"fold", "call", "raise"} if facing else {"check", "bet"}
    return np.array([a in legal for a in COARSE_ACTIONS], dtype=bool)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2 -> range [0, 1]."""
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def build_pair_similarity(rows: list[dict]) -> dict:
    """Data-derived (hero_pos, villain_pos) -> action-dist; plus sim = 1 - JS lookup."""
    counts: dict[tuple, np.ndarray] = {}
    for dp in rows:
        pair = (dp.get("hero_pos", ""), _villain_pos(dp) or "NA")
        a = coarse_action(dp)
        if a is None:
            continue
        counts.setdefault(pair, np.zeros(len(COARSE_ACTIONS)))[_ACTION_IDX[a]] += 1.0
    dists = {pair: c / c.sum() for pair, c in counts.items() if c.sum() > 0}
    return dists


def pair_sim(dists: dict, pair_a: tuple, pair_b: tuple) -> float:
    da, db = dists.get(pair_a), dists.get(pair_b)
    if da is None or db is None:
        return 0.0
    return 1.0 - _js_divergence(da, db)


# ---- LOO over one variant ----------------------------------------------------


def eval_variant(
    rows: list[dict],
    full_vecs: np.ndarray,
    k: int,
    pair_dists: dict,
    modal_action: dict[tuple, str],
) -> dict:
    """Leave-one-out cosine kNN within each hard-filter partition.

    Returns overall loss, log-loss, pos-sensitive-subset loss, position fidelity,
    and coverage (fraction of rows scored).
    """
    n = len(rows)
    parts: dict[tuple, list[int]] = {}
    for i, dp in enumerate(rows):
        parts.setdefault(partition_key(dp), []).append(i)

    pairs = [(dp.get("hero_pos", ""), _villain_pos(dp) or "NA") for dp in rows]
    true_idx = np.array([_ACTION_IDX[coarse_action(dp)] for dp in rows])

    losses, loglosses, fidelities = [], [], []
    pos_sensitive_losses = []
    scored = 0

    for part, idxs in parts.items():
        if len(idxs) < k + 1:
            continue
        idxs_arr = np.array(idxs)
        unit = _unit(full_vecs[idxs_arr])
        nbr, nbr_sim = topk_cosine(unit, k)
        m = len(idxs)

        for local_i in range(m):
            gi = idxs_arr[local_i]
            nbr_local = nbr[local_i]
            w = np.maximum(nbr_sim[local_i], 0.0)  # similarity weights (clamped)
            # Blend coarse actions
            dist = np.zeros(len(COARSE_ACTIONS))
            for j_local, wj in zip(nbr_local, w, strict=True):
                if wj <= 0:
                    continue
                dist[true_idx[idxs_arr[j_local]]] += wj
            # Restrict to legal actions for this query's facing state, renormalize
            mask = _legal_mask(facing_bet(rows[gi]))
            dist = dist * mask
            tot = dist.sum()
            if tot <= 0:
                # Degenerate: passive legal fallback (fold if facing, else check)
                dist = np.zeros(len(COARSE_ACTIONS))
                dist[_ACTION_IDX["fold" if facing_bet(rows[gi]) else "check"]] = 1.0
            else:
                dist = dist / tot

            p_true = float(dist[true_idx[gi]])
            losses.append(1.0 - p_true)
            loglosses.append(-math.log(max(p_true, 1e-12)))

            if coarse_action(rows[gi]) != modal_action.get(part):
                pos_sensitive_losses.append(1.0 - p_true)

            # Position fidelity: mean pair-sim over neighbors (skipped when not needed)
            if pair_dists is not None:
                q_pair = pairs[gi]
                fid = np.mean([pair_sim(pair_dists, q_pair, pairs[idxs_arr[j]]) for j in nbr_local])
                fidelities.append(float(fid))
            scored += 1

    return {
        "loo_loss": float(np.mean(losses)) if losses else float("nan"),
        "loo_logloss": float(np.mean(loglosses)) if loglosses else float("nan"),
        "loo_loss_possensitive": float(np.mean(pos_sensitive_losses))
        if pos_sensitive_losses
        else float("nan"),
        "position_fidelity": float(np.mean(fidelities)) if fidelities else float("nan"),
        "coverage": scored / n if n else 0.0,
        "n_scored": scored,
        "n_pos_sensitive": len(pos_sensitive_losses),
    }


# ---- Driver ------------------------------------------------------------------


def compute_seat_aggression(rows: list[dict]) -> dict[str, float]:
    """Data-derived seat ordering: rank seats by P(bet|raise as hero), map to [0,1]."""
    agg: dict[str, list[int]] = {}
    for dp in rows:
        a = coarse_action(dp)
        if a is None:
            continue
        agg.setdefault(dp.get("hero_pos", ""), []).append(1 if a in ("bet", "raise") else 0)
    rate = {pos: (sum(v) / len(v) if v else 0.0) for pos, v in agg.items()}
    order = sorted(rate, key=lambda p: rate[p])  # least -> most aggressive
    n = max(len(order) - 1, 1)
    return {pos: i / n for i, pos in enumerate(order)}


def _assign_folds(n: int, n_folds: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    folds = np.empty(n, dtype=np.int64)
    for rank, i in enumerate(order):
        folds[i] = rank % n_folds
    return folds


def _variant_pos_blocks(rows: list[dict], holdout: list[dict] | None, kfold: int) -> dict[str, np.ndarray]:
    """Per-variant raw position blocks. Data-driven descriptors fit leak-free: on
    `holdout` (disjoint) when given, else k-fold (each row's char from other folds)
    for full-corpus runs; in-sample only as last resort.
    """
    n = len(rows)
    if kfold and kfold > 1:
        folds = _assign_folds(n, kfold, seed=0)
        char_by_fold = [
            fit_position_character([rows[i] for i in range(n) if folds[i] != f]) for f in range(kfold)
        ]
        seat_by_fold = [
            compute_seat_aggression([rows[i] for i in range(n) if folds[i] != f]) for f in range(kfold)
        ]
        char_per_row = [char_by_fold[folds[i]] for i in range(n)]
        seat_per_row = [seat_by_fold[folds[i]] for i in range(n)]
    else:
        fit_rows = holdout if holdout is not None else rows
        cg, sg = fit_position_character(fit_rows), compute_seat_aggression(fit_rows)
        char_per_row, seat_per_row = [cg] * n, [sg] * n

    return {
        "baseline": np.stack([pos_block_baseline(rows[i]) for i in range(n)]),
        "ordinal-openorder": np.stack([pos_block_ordinal(rows[i], OPEN_ORDER) for i in range(n)]),
        "ordinal-dataderived": np.stack([pos_block_ordinal(rows[i], seat_per_row[i]) for i in range(n)]),
        "relationship": np.stack([pos_block_relationship(rows[i]) for i in range(n)]),
        "relationship-v2": np.stack([pos_block_relationship_v2(rows[i], char_per_row[i]) for i in range(n)]),
        "range-char": np.stack([pos_block_range_character(rows[i], char_per_row[i]) for i in range(n)]),
    }


def _build_matrices(
    rows: list[dict],
    pos_weight: float,
    eq: EquityLookup,
    pop_mean: dict[str, float],
    holdout: list[dict] | None = None,
    kfold: int = 0,
) -> dict[str, np.ndarray]:
    """Per-variant normalized full vectors (non-position block shared, position swapped)."""
    non_pos = np.stack([non_position_block(dp, eq, pop_mean) for dp in rows])
    pos_blocks = _variant_pos_blocks(rows, holdout, kfold)
    matrices = {}
    for name, pblock in pos_blocks.items():
        raw = np.concatenate([non_pos, pblock], axis=1).astype(np.float64)
        mean, std = fit_zscore(raw)
        weights = np.concatenate([_NON_POS_WEIGHTS, np.full(pblock.shape[1], pos_weight)])
        matrices[name] = normalize(raw, mean, std, weights)
    return matrices


def _load_for_eval(
    decisions: str, sample_n: int, seed: int, full: bool
) -> tuple[list[dict], list[dict] | None, int]:
    """Returns (rows, holdout, kfold). full -> all postflop spots with k-fold char."""
    if full:
        rows: list[dict] = _load_all(decisions)
        holdout, kfold = None, 5
    else:
        rows, holdout = load_sample_and_holdout(decisions, sample_n, seed)
        kfold = 0
    for dp in rows:
        dp["scenario_key"] = reconstruct_scenario_key(dp)
    return rows, holdout, kfold


def run(decisions: str, sample_n: int, k: int, seed: int, pos_weight: float, full: bool = False) -> dict:
    rows, holdout, kfold = _load_for_eval(decisions, sample_n, seed, full)

    eq = EquityLookup("tools/equity_table.parquet")
    pop_mean = eq.population_mean_equity()
    matrices = _build_matrices(rows, pos_weight, eq, pop_mean, holdout, kfold)

    pair_dists = build_pair_similarity(rows)
    parts: dict[tuple, list[str]] = {}
    for dp in rows:
        parts.setdefault(partition_key(dp), []).append(coarse_action(dp))
    modal_action = {part: max(set(acts), key=acts.count) for part, acts in parts.items() if acts}

    results = {}
    for name, full in matrices.items():
        results[name] = eval_variant(rows, full, k, pair_dists, modal_action)
        results[name]["pos_dims"] = int(full.shape[1] - len(_NON_POS_WEIGHTS))

    return {"n_sample": len(rows), "k": k, "seed": seed, "pos_weight": pos_weight, "variants": results}


# ---- Archetype analysis: same-action@k across many spots, sliced ------------


def _is_blind_battle(dp: dict) -> bool:
    return dp.get("hero_pos") in BLINDS and (_villain_pos(dp) or "") in BLINDS


def _topk_per_row(sub: np.ndarray, k: int) -> np.ndarray:
    """Top-k neighbor column indices per row within a partition (self excluded)."""
    return topk_cosine(_unit(sub), k)[0]


def analyze_archetypes(
    rows: list[dict], matrices: dict[str, np.ndarray], k: int, eq: EquityLookup, pop_mean: dict[str, float]
) -> tuple[dict, dict]:
    """Per-variant same-action@k (mean over a query's k neighbors of [nbr action ==
    query's true action]) sliced by archetype, plus pooling stats. Rawer than LOO
    loss; mirrors the eyeballed 'same-action X/8' column.
    """
    parts: dict[tuple, list[int]] = {}
    for i, dp in enumerate(rows):
        parts.setdefault(partition_key(dp), []).append(i)
    modal = {}
    for part, idxs in parts.items():
        acts = [coarse_action(rows[i]) for i in idxs]
        modal[part] = max(set(acts), key=acts.count)

    true_act = [coarse_action(dp) for dp in rows]
    eq_mean = [_eq_mean(dp, eq, pop_mean) for dp in rows]

    def _tags(i: int) -> list[str]:
        dp = rows[i]
        m = eq_mean[i]
        eqb = "eq:strong>.7" if m > 0.7 else ("eq:marginal.4-.7" if m >= 0.4 else "eq:weak<.4")
        t = ["overall", eqb, "facing-bet" if facing_bet(dp) else "no-bet", f"pot:{dp.get('pot_type', '')}"]
        hero = dp.get("hero_pos", "")
        vill = _villain_pos(dp)
        hb, vb = hero in BLINDS, (vill in BLINDS if vill else False)
        if hb and vb:
            t.append("blind-battle")
        elif hb:
            t.append(f"{hero}-vs-opener")
        elif vb:
            t.append("vs-blind")
        if true_act[i] != modal[partition_key(dp)]:
            t.append("pos-sensitive")
        return t

    cells: dict[str, dict[str, list[float]]] = {name: {} for name in matrices}
    pooling: dict[str, dict[str, list[float]]] = {
        name: {"exact_pair": [], "blindbattle_purity": []} for name in matrices
    }
    for name, full in matrices.items():
        for idxs in parts.values():
            if len(idxs) < k + 1:
                continue
            idxs_arr = np.array(idxs)
            nbr = _topk_per_row(full[idxs_arr], k)
            for li in range(len(idxs)):
                gi = idxs_arr[li]
                nb = idxs_arr[nbr[li]]
                frac = float(np.mean([true_act[j] == true_act[gi] for j in nb]))
                for tag in _tags(gi):
                    cells[name].setdefault(tag, []).append(frac)
                q_pair = (rows[gi].get("hero_pos", ""), _villain_pos(rows[gi]) or "-")
                pooling[name]["exact_pair"].append(
                    float(
                        np.mean(
                            [
                                (rows[j].get("hero_pos", ""), _villain_pos(rows[j]) or "-") == q_pair
                                for j in nb
                            ]
                        )
                    )
                )
                if _is_blind_battle(rows[gi]):
                    pooling[name]["blindbattle_purity"].append(
                        float(np.mean([_is_blind_battle(rows[j]) for j in nb]))
                    )
    return cells, pooling


def _print_archetypes(cells: dict, pooling: dict, k: int) -> None:
    variants = list(cells.keys())
    # archetype display order
    order = [
        "overall",
        "eq:weak<.4",
        "eq:marginal.4-.7",
        "eq:strong>.7",
        "no-bet",
        "facing-bet",
        "pot:srp",
        "pot:3bet",
        "pot:4bet",
        "pot:limp",
        "blind-battle",
        "BB-vs-opener",
        "SB-vs-opener",
        "vs-blind",
        "pos-sensitive",
    ]
    present = [a for a in order if a in cells["baseline"]]
    print(f"\nsame-action@{k} by archetype  (higher = neighbors predict the true action better)\n")
    hdr = f"{'archetype':<20} {'n':>6} " + " ".join(f"{v[:11]:>12}" for v in variants)
    print(hdr)
    print("-" * len(hdr))
    for a in present:
        n = len(cells["baseline"][a])
        base = float(np.mean(cells["baseline"][a]))
        cellstrs = []
        for v in variants:
            val = float(np.mean(cells[v][a]))
            cellstrs.append(f"{val:.3f}" if v == "baseline" else f"{val:.3f}({val - base:+.3f})")
        print(f"{a:<20} {n:>6} " + " ".join(f"{c:>12}" for c in cellstrs))
    print("\npooling behavior:")
    print(f"  {'metric':<28} " + " ".join(f"{v[:11]:>12}" for v in variants))
    print(
        f"  {'exact-pair frac of nbrs':<28} "
        + " ".join(f"{np.mean(pooling[v]['exact_pair']):>12.3f}" for v in variants)
    )
    bb = [v for v in variants if pooling[v]["blindbattle_purity"]]
    if bb:
        print(
            f"  {'blind-battle nbr purity':<28} "
            + " ".join(f"{np.mean(pooling[v]['blindbattle_purity']):>12.3f}" for v in variants)
        )
        print("    (of blind-battle queries: fraction of neighbors that are also blind battles)")


# ---- Inspect: eyeball the actual neighbors each encoding retrieves -----------


def _eq_mean(dp: dict, eq: EquityLookup, pop_mean: dict[str, float]) -> float:
    """Mean equity for a spot (for display); pop_mean on miss."""
    from tools.joint_canonicalize import joint_canonicalize

    hole = dp.get("hero_hole") or []
    board = dp.get("board") or []
    if hole and board:
        h, b = joint_canonicalize(hole, board)
        r = eq.get(h, b, dp.get("scenario_key", ""))
        if r is not None:
            return float(r["mean"])
    return float(pop_mean["mean"])


def _spot_str(dp: dict, eq: EquityLookup, pop_mean: dict[str, float]) -> str:
    hero = dp.get("hero_pos", "?")
    vill = _villain_pos(dp) or "-"
    board = "".join(dp.get("board") or [])
    hole = "".join(dp.get("hero_hole") or [])
    return (
        f"{hero:>3} vs {vill:<3} | {dp.get('pot_type', ''):<5} {dp.get('street', ''):<5} "
        f"| {board:<10} {hole:<5} | {dp.get('facing', ''):<8} -> {str(coarse_action(dp)).upper():<5} "
        f"| eqμ={_eq_mean(dp, eq, pop_mean):.2f}"
    )


def _hole_matches(dp: dict, pat: str) -> bool:
    """Match hero hole to a pattern: 2 chars = ranks (e.g. KK, AK suited-agnostic), 4 = exact."""
    hole = dp.get("hero_hole") or []
    if len(hole) != 2:
        return False
    if len(pat) == 2:
        return sorted(c[0] for c in hole) == sorted(pat.upper())
    return "".join(hole) == pat


def inspect(
    decisions: str,
    sample_n: int,
    k: int,
    seed: int,
    pos_weight: float,
    n_queries: int,
    hero_filter: str | None,
    vill_filter: str | None,
    hole_filter: str | None,
    pot_filter: str | None,
    full: bool = False,
) -> None:
    """Print, per query spot, the top-k neighbors each variant retrieves (within partition)."""
    rows, holdout, kfold = _load_for_eval(decisions, sample_n, seed, full)
    eq = EquityLookup("tools/equity_table.parquet")
    pop_mean = eq.population_mean_equity()
    matrices = _build_matrices(rows, pos_weight, eq, pop_mean, holdout, kfold)

    parts: dict[tuple, list[int]] = {}
    for i, dp in enumerate(rows):
        parts.setdefault(partition_key(dp), []).append(i)

    def _ok(dp: dict) -> bool:
        if hero_filter and dp.get("hero_pos") != hero_filter:
            return False
        if vill_filter and (_villain_pos(dp) or "") != vill_filter:
            return False
        if pot_filter and dp.get("pot_type") != pot_filter:
            return False
        return not (hole_filter and not _hole_matches(dp, hole_filter))

    rng = random.Random(seed + 1)
    candidates = [i for i, dp in enumerate(rows) if _ok(dp) and len(parts[partition_key(dp)]) >= k + 1]
    if not candidates:
        print("no query spots match the filter with a populated partition")
        return
    queries = rng.sample(candidates, min(n_queries, len(candidates)))

    for n, qi in enumerate(queries, 1):
        part = partition_key(rows[qi])
        idxs = np.array(parts[part])
        q_pair = (rows[qi].get("hero_pos", ""), _villain_pos(rows[qi]) or "-")
        print(f"\n{'=' * 100}")
        print(f"QUERY #{n}  [partition {part[1]}/{part[2]}/n{part[3]}, size {len(idxs)}]")
        print(f"  {_spot_str(rows[qi], eq, pop_mean)}")
        for name, mat in matrices.items():
            unit = _unit(mat[idxs])
            qpos = int(np.where(idxs == qi)[0][0])
            sims = unit @ unit[qpos]
            sims[qpos] = -np.inf
            top = idxs[np.argsort(-sims)[:k]]
            top_sims = np.sort(sims)[::-1][:k]
            pair_match = sum(
                1 for t in top if (rows[t].get("hero_pos", ""), _villain_pos(rows[t]) or "-") == q_pair
            )
            same_action = sum(1 for t in top if coarse_action(rows[t]) == coarse_action(rows[qi]))
            print(f"\n  [{name}]  exact-pair {pair_match}/{k}   same-action {same_action}/{k}")
            for t, s in zip(top, top_sims, strict=True):
                print(f"      sim={s:.3f}  {_spot_str(rows[t], eq, pop_mean)}")


def _print_table(out: dict) -> None:
    print(
        f"\nPosition-encoding comparison  (n={out['n_sample']}, k={out['k']}, "
        f"seed={out['seed']}, pos_weight={out['pos_weight']})\n"
    )
    hdr = f"{'variant':<22} {'dims':>4} {'LOO loss':>9} {'logloss':>8} {'pos-sens':>9} {'fidelity':>9} {'cov':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in out["variants"].items():
        print(
            f"{name:<22} {r['pos_dims']:>4} {r['loo_loss']:>9.4f} {r['loo_logloss']:>8.4f} "
            f"{r['loo_loss_possensitive']:>9.4f} {r['position_fidelity']:>9.4f} {r['coverage']:>6.2f}"
        )
    print("\nLower LOO loss = better action prediction. Higher fidelity = pulls similar position pairs.")
    base = out["variants"]["baseline"]
    print(f"\nbaseline: loss={base['loo_loss']:.4f} fidelity={base['position_fidelity']:.4f}")
    for name, r in out["variants"].items():
        if name == "baseline":
            continue
        dl = r["loo_loss"] - base["loo_loss"]
        df = r["position_fidelity"] - base["position_fidelity"]
        verdict = "WIN" if (dl < 0 and df > 0) else ("loss↓" if dl < 0 else ("fid↑" if df > 0 else "no"))
        print(f"  {name:<22} Δloss={dl:+.4f}  Δfidelity={df:+.4f}  [{verdict}]")


def sweep(decisions: str, sample_n: int, k: int, seed: int, weights: list[float], full: bool) -> dict:
    """LOO loss for every (variant, pos_weight). Confounds check: is the encoding win
    robust to weighting, and what weight is optimal? z-score is fit once per variant
    (weight-independent); only the position dims are re-scaled per weight point."""
    rows, holdout, kfold = _load_for_eval(decisions, sample_n, seed, full)
    eq = EquityLookup("tools/equity_table.parquet")
    pop_mean = eq.population_mean_equity()
    non_pos = np.stack([non_position_block(dp, eq, pop_mean) for dp in rows])
    pos_blocks = _variant_pos_blocks(rows, holdout, kfold)

    parts: dict[tuple, list[str]] = {}
    for dp in rows:
        parts.setdefault(partition_key(dp), []).append(coarse_action(dp))
    modal = {p: max(set(a), key=a.count) for p, a in parts.items() if a}

    n62 = len(_NON_POS_WEIGHTS)
    results: dict[str, dict[float, float]] = {}
    for name, pblock in pos_blocks.items():
        raw = np.concatenate([non_pos, pblock], axis=1).astype(np.float64)
        mean, std = fit_zscore(raw)
        z = (raw - mean) / std
        zN = z[:, :n62] * _NON_POS_WEIGHTS  # constant across weights
        zP = z[:, n62:]
        results[name] = {}
        for w in weights:
            full_vec = np.concatenate([zN, zP * w], axis=1)
            results[name][w] = eval_variant(rows, full_vec, k, None, modal)["loo_loss"]
    return {"n": len(rows), "k": k, "weights": weights, "full": full, "results": results}


def _print_sweep(out: dict) -> None:
    weights = out["weights"]
    print(f"\nLOO loss vs pos_weight  (n={out['n']}, k={out['k']}, full={out['full']})")
    print("pos_weight=0 => position ignored (shared floor). Best weight per variant marked *.\n")
    hdr = f"{'variant':<22} " + " ".join(f"{f'w={w:g}':>9}" for w in weights) + f"  {'best':>14}"
    print(hdr)
    print("-" * len(hdr))
    for name, row in out["results"].items():
        best_w = min(row, key=lambda w: row[w])
        cells = " ".join(f"{('*' if w == best_w else ' ')}{row[w]:.4f}" for w in weights)
        print(f"{name:<22} {cells}  {f'{row[best_w]:.4f}@{best_w:g}':>14}")
    base_best = min(out["results"]["baseline"].values())
    print(f"\nbaseline best loss = {base_best:.4f}")
    for name, row in out["results"].items():
        if name == "baseline":
            continue
        b = min(row.values())
        print(f"  {name:<22} best={b:.4f}  Δ vs baseline-best={b - base_best:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline LOO comparison of postflop position encodings")
    ap.add_argument("--decisions", required=True, help="postflop decisions JSONL")
    ap.add_argument("--sample-n", type=int, default=8000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-weight", type=float, default=0.85, help="constant weight for position dims")
    ap.add_argument("--json-out", default=None, help="optional path to dump the result dict as JSON")
    ap.add_argument("--inspect", action="store_true", help="print actual kNN neighbors per variant")
    ap.add_argument("--inspect-n", type=int, default=5, help="number of query spots to inspect")
    ap.add_argument("--inspect-hero", default=None, help="filter query spots to this hero_pos")
    ap.add_argument("--inspect-vill", default=None, help="filter query spots to this villain pos")
    ap.add_argument("--inspect-hole", default=None, help="filter query hole: 'KK' (ranks) or 'KhKd' (exact)")
    ap.add_argument("--inspect-pot", default=None, help="filter query pot_type (srp/3bet/4bet/...)")
    ap.add_argument("--archetypes", action="store_true", help="aggregate same-action@k sliced by archetype")
    ap.add_argument("--full", action="store_true", help="eval ALL postflop spots (k-fold char), not a sample")
    ap.add_argument("--sweep", action="store_true", help="sweep pos_weight per variant (confound check)")
    ap.add_argument("--sweep-weights", default="0,0.4,0.85,1.5,3,6", help="comma-separated pos_weights")
    args = ap.parse_args()

    if args.sweep:
        weights = [float(w) for w in args.sweep_weights.split(",")]
        _print_sweep(sweep(args.decisions, args.sample_n, args.k, args.seed, weights, args.full))
        return

    if args.archetypes:
        rows, holdout, kfold = _load_for_eval(args.decisions, args.sample_n, args.seed, args.full)
        eq = EquityLookup("tools/equity_table.parquet")
        pop_mean = eq.population_mean_equity()
        matrices = _build_matrices(rows, args.pos_weight, eq, pop_mean, holdout, kfold)
        cells, pooling = analyze_archetypes(rows, matrices, args.k, eq, pop_mean)
        print(
            f"\n(n={len(rows)}, k={args.k}, seed={args.seed}, pos_weight={args.pos_weight}, full={args.full})"
        )
        _print_archetypes(cells, pooling, args.k)
        return

    if args.inspect:
        inspect(
            args.decisions,
            args.sample_n,
            args.k,
            args.seed,
            args.pos_weight,
            args.inspect_n,
            args.inspect_hero,
            args.inspect_vill,
            args.inspect_hole,
            args.inspect_pot,
            args.full,
        )
        return

    out = run(args.decisions, args.sample_n, args.k, args.seed, args.pos_weight, args.full)
    _print_table(out)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
