"""Build preflop equity decile table via Monte Carlo simulation.

For each (hero_hand_class, scenario) cell, runs N_TRIALS MC evals:
  - Samples a villain combo by palette-weighted class distribution
  - Deals random 5-card board from remaining deck
  - Evaluates hero vs villain 7-card best hand
  - Computes equity as win + 0.5*tie rate

Output parquet schema matches equity_table.parquet (postflop) exactly so
EquityLookup can union both files.

Run:
    uv run python tools/build_preflop_equity_table.py \\
        --decisions research/preflop-ranges/outputs/hm_decisions.jsonl \\
        --out tools/preflop_equity_table.parquet \\
        --n-trials 500 \\
        --threads 12 \\
        --log-file outputs/preflop_equity_build.log
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Repo root on path so tools/ imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.build_equity_table import _role_to_filename_candidates
from tools.joint_canonicalize import joint_canonicalize

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario key logic (inlined from enumerate_equity_tuples to avoid sys.path
# issues when that module uses bare `from joint_canonicalize import ...`)
# ---------------------------------------------------------------------------


def _classify_villain_role(
    pos: str,
    preflop_seq: list[dict],
    preflop_aggressor: str | None,
    pot_type: str | None = None,
) -> tuple[str, str | None]:
    """Classify a villain's preflop role + target."""
    actions = [a for a in preflop_seq if a["pos"] == pos]
    if not actions:
        # Limp pots only log folds/raises; an unlogged villain entered by limping.
        if pot_type == "limp":
            return ("limper", None)
        return ("unknown", None)

    raises_before_villain: list[str] = []
    role = "unknown"
    target = None

    for step in preflop_seq:
        p = step["pos"]
        act = step["action"]
        if p == pos:
            if act == "raise":
                if not raises_before_villain:
                    role = "pfr"
                    target = None
                elif len(raises_before_villain) == 1:
                    role = "3bettor"
                    target = raises_before_villain[-1]
                elif len(raises_before_villain) == 2:
                    role = "cold_4bettor" if pos != preflop_aggressor else "4bettor"
                    target = raises_before_villain[-1]
                else:
                    role = "5bettor"
                    target = raises_before_villain[-1]
            elif act == "call":
                if not raises_before_villain:
                    role = "limper"
                elif role in ("unknown", "limper"):
                    role = "caller" if len(raises_before_villain) == 1 else "cold_caller"
                    target = raises_before_villain[-1]
                else:
                    target = raises_before_villain[-1]
            elif act == "check" and not raises_before_villain:
                role = "checker"
        elif act == "raise":
            raises_before_villain.append(p)

    return (role, target)


def _compute_scenario_key(dp: dict) -> str:
    """Map a DP dict to a palette scenario key string."""
    pot_type = dp["pot_type"]
    hero_rel = dp["hero_pos_rel"]
    n = dp["n_players_at_street"]
    seq = dp.get("preflop_action_seq") or []
    pfr = dp.get("preflop_aggressor")

    villains = dp.get("villains_active") or []
    if not villains:
        return f"{pot_type}|{hero_rel}|n{n}|vNA"

    specs = []
    for v in villains:
        pos = v["pos"]
        role, tgt = _classify_villain_role(pos, seq, pfr, pot_type)
        if tgt:
            specs.append(f"{pos}:{role}:{tgt}")
        else:
            specs.append(f"{pos}:{role}")
    specs.sort()
    return f"{pot_type}|{hero_rel}|n{n}|" + "+".join(specs)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANKS = "23456789TJQKA"
SUITS = "cdhs"
_RANK_VAL = {r: i for i, r in enumerate(RANKS)}
PALETTE_DIR = Path("research/preflop-ranges/outputs")

# Full 52-card deck
_DECK: list[str] = [r + s for r in RANKS for s in SUITS]

# All 169 canonical hand classes
_PAIRS = [r + r for r in RANKS]  # 13 pairs
_SUITED = [r1 + r2 + "s" for i, r1 in enumerate(RANKS) for r2 in RANKS[:i]]  # 78 suited
_OFFSUIT = [r1 + r2 + "o" for i, r1 in enumerate(RANKS) for r2 in RANKS[:i]]  # 78 offsuit
ALL_HAND_CLASSES: list[str] = _PAIRS + _SUITED + _OFFSUIT  # 169 total

# Decile percentile points
_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]


# ---------------------------------------------------------------------------
# Hand evaluation (reused from postflop.py — pure Python)
# ---------------------------------------------------------------------------


def _evaluate_5(cards: list[str]) -> tuple[int, list[int]]:
    """Evaluate a 5-card hand. Returns (hand_rank_int, tiebreakers).

    hand_rank_int: 0=high_card, 1=one_pair, ..., 8=straight_flush
    """
    from collections import Counter

    ranks = sorted([_RANK_VAL[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    is_flush = len(set(suits)) == 1

    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[-1] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [12, 3, 2, 1, 0]:
            is_straight = True
            straight_high = 3  # five-high

    freq = Counter(ranks)
    counts = sorted(freq.values(), reverse=True)
    count_ranks = sorted(freq.keys(), key=lambda r: (freq[r], r), reverse=True)

    if is_straight and is_flush:
        return 8, [straight_high]
    if counts[0] == 4:
        return 7, [count_ranks[0], count_ranks[1]]
    if counts[0] == 3 and counts[1] == 2:
        return 6, [count_ranks[0], count_ranks[1]]
    if is_flush:
        return 5, ranks
    if is_straight:
        return 4, [straight_high]
    if counts[0] == 3:
        return 3, [count_ranks[0], *sorted([r for r in ranks if freq[r] == 1], reverse=True)]
    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in freq.items() if c == 2], reverse=True)
        kicker = max(r for r, c in freq.items() if c == 1)
        return 2, [*pairs, kicker]
    if counts[0] == 2:
        pair_rank = count_ranks[0]
        kickers = sorted([r for r in ranks if freq[r] == 1], reverse=True)
        return 1, [pair_rank, *kickers]
    return 0, ranks


def _best_5_from_7(hole: list[str], board: list[str]) -> tuple[int, list[int]]:
    """Find best 5-card hand from 7 cards (hole + board)."""
    all_cards = hole + board
    best_rank = -1
    best_tb: list[int] = []
    for combo in combinations(all_cards, 5):
        hr, tb = _evaluate_5(list(combo))
        if (hr, tb) > (best_rank, best_tb):
            best_rank = hr
            best_tb = tb
    return best_rank, best_tb


# ---------------------------------------------------------------------------
# Deterministic hero combo from hand class
# ---------------------------------------------------------------------------


def _hero_combo_for_class(hand_class: str) -> list[str]:
    """Return a deterministic 2-card combo for a hand class.

    Pair XX    -> Xc Xd
    Suited XYs -> Xc Yc
    Offsuit XYo -> Xc Yd
    """
    if len(hand_class) == 2:
        # Pair
        r = hand_class[0]
        return [r + "c", r + "d"]
    r1, r2, suitness = hand_class[0], hand_class[1], hand_class[2]
    # r1 is higher rank (class notation: stronger first by convention)
    # For RANKS-indexed pairs with r2 < r1 by index: verify and swap if needed
    if _RANK_VAL[r2] > _RANK_VAL[r1]:
        r1, r2 = r2, r1
    if suitness == "s":
        return [r1 + "c", r2 + "c"]
    else:  # offsuit
        return [r1 + "c", r2 + "d"]


# ---------------------------------------------------------------------------
# Villain combo generation from a hand class
# ---------------------------------------------------------------------------


def _villain_combos_for_class(hand_class: str, blocked: set[str]) -> list[list[str]] | None:
    """Generate all possible villain combos for a hand class, excluding blocked cards.

    Returns list of 2-card combos, or None if no valid combo available.
    """
    if len(hand_class) == 2:
        # Pair: all 6 combinations of 4 suits
        r = hand_class[0]
        cards = [r + s for s in SUITS]
        combos = [list(c) for c in combinations(cards, 2)]
    else:
        r1, r2, suitness = hand_class[0], hand_class[1], hand_class[2]
        if _RANK_VAL[r2] > _RANK_VAL[r1]:
            r1, r2 = r2, r1
        if suitness == "s":
            # All 4 suited combos
            combos = [[r1 + s, r2 + s] for s in SUITS]
        else:
            # All 12 offsuit combos
            combos = [[r1 + s1, r2 + s2] for s1 in SUITS for s2 in SUITS if s1 != s2]

    valid = [c for c in combos if not (set(c) & blocked)]
    return valid if valid else None


def _sample_villain_combo(hand_class: str, blocked: set[str], rng: random.Random) -> list[str] | None:
    """Sample a random villain combo from hand_class, avoiding blocked cards."""
    valid = _villain_combos_for_class(hand_class, blocked)
    if not valid:
        return None
    return rng.choice(valid)


# ---------------------------------------------------------------------------
# Palette loading: combo_counts dict per scenario
# ---------------------------------------------------------------------------


def _build_pooled_field_range(palette_dir: Path) -> dict[str, float]:
    """Union all palette combo_counts into one pooled "field" range.

    Pools ALL positions and actions so vNA ("unknown villain") gets an
    uninformative prior covering the full observed player population.
    Returns {hand_class: total_weight} or {} if no files found.
    """
    pooled: dict[str, float] = {}
    for path in palette_dir.glob("*/*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cc = _resolve_combo_counts(data)
        if not cc:
            continue
        for hc, w in cc.items():
            pooled[hc] = pooled.get(hc, 0.0) + float(w)
    return pooled


def _load_palette_combo_counts(
    scenario_key: str,
    pooled_field_range: dict[str, float] | None = None,
) -> list[dict[str, float]] | None:
    """Load per-villain combo_counts for a scenario.

    Returns list[dict] — one dict per villain — or None if scenario cannot be resolved.
    Single-villain scenarios return a list of length 1.
    vNA scenarios return [pooled_field_range] when provided, else None.
    """
    parts = scenario_key.split("|")
    if len(parts) != 4:
        return None
    _pot_type, _hero_rel, nstr, vstr = parts
    n = int(nstr[1:])
    villain_specs = vstr.split("+")

    if villain_specs == ["vNA"] or vstr == "vNA":
        if pooled_field_range:
            return [pooled_field_range]
        return None

    multiway = n >= 3
    per_villain: list[dict[str, float]] = []

    for spec in villain_specs:
        bits = spec.split(":")
        if len(bits) < 2:
            return None
        pos = bits[0]
        role = bits[1]
        target = bits[2] if len(bits) > 2 else None

        candidates = _role_to_filename_candidates(role, target, multiway)
        found = False
        for c in candidates:
            p = PALETTE_DIR / pos / c
            if p.exists():
                with p.open() as f:
                    data = json.load(f)
                cc = _resolve_combo_counts(data)
                if cc:
                    per_villain.append({hc: float(w) for hc, w in cc.items()})
                    found = True
                    break
        if not found:
            return None

    return per_villain if per_villain else None


def _resolve_combo_counts(data: dict) -> dict[str, float] | None:
    """Pick the villain distribution for a palette file.

    Thin samples (low_sample, or empty/absent empirical combo_counts) fall back
    to the bundled inferred GTO prior when present. Otherwise the empirical
    combo_counts win. Returns class->weight, or None if neither resolves.
    """
    empirical = data.get("combo_counts") or {}
    low_sample = bool(data.get("low_sample"))
    inferred = data.get("inferred_ranges") or data.get("inferred_rates")
    if (low_sample or not empirical) and inferred:
        return {hc: float(w) for hc, w in inferred.items() if w}
    return {hc: float(c) for hc, c in empirical.items() if c} or None


# ---------------------------------------------------------------------------
# MC equity computation for one (hero_class, scenario) cell
# ---------------------------------------------------------------------------


# Loaded once per process; populated by _load_matrix() in matrix mode.
_MATRIX: dict[str, dict[str, float]] | None = None


def _load_matrix(path: Path) -> dict[str, dict[str, float]]:
    """Load the exact 169x169 hand-vs-hand parquet into {hero: {villain: eq}}."""
    table = pq.read_table(path)
    hero = table.column("hero_class").to_pylist()
    vill = table.column("villain_class").to_pylist()
    eq = table.column("equity").to_pylist()
    m: dict[str, dict[str, float]] = {}
    for h, v, e in zip(hero, vill, eq, strict=True):
        m.setdefault(h, {})[v] = e
    return m


def _matrix_cell_equity(
    hero_class: str,
    combo_counts: dict[str, float],
    matrix: dict[str, dict[str, float]],
) -> dict[str, float] | None:
    """Exact equity of hero_class vs a weighted villain range, from the matrix.

    Equity per villain class is exact (board already enumerated). The deciles
    describe the SPREAD of hero's equity across the villain range, weighted by
    combo_counts — the intended Group-A feature. Collision-correct combo counts
    (hero blocks villain combos of overlapping classes) are folded into the
    weights so a hero pair vs the same villain pair is down-weighted accordingly.
    """
    row = matrix.get(hero_class)
    if row is None:
        return None

    pairs: list[tuple[float, float]] = []  # (equity, weight)
    for vclass, w in combo_counts.items():
        if w <= 0:
            continue
        e = row.get(vclass)
        if e is None:
            continue
        adj = w * _combo_collision_factor(hero_class, vclass)
        if adj > 0:
            pairs.append((e, adj))
    if not pairs:
        return None

    pairs.sort(key=lambda x: x[0])
    eqs = np.array([p[0] for p in pairs], dtype=np.float64)
    wts = np.array([p[1] for p in pairs], dtype=np.float64)
    total = wts.sum()
    if total <= 0:
        return None

    mean_eq = float((eqs * wts).sum() / total)
    cdf = np.cumsum(wts) / total
    deciles = [float(np.interp(p / 100.0, cdf, eqs)) for p in _PERCENTILES]
    return {
        "mean_equity": mean_eq,
        **{f"p{p}": d for p, d in zip(_PERCENTILES, deciles, strict=True)},
    }


def _matrix_cell_equity_multiway(
    hero_class: str,
    per_villain_combo_counts: list[dict[str, float]],
    matrix: dict[str, dict[str, float]],
) -> dict[str, float] | None:
    """Exact equity of hero_class vs multiple villains using Cartesian product.

    Each villain's range is represented as a per-class weight dict. The joint
    equity of beating ALL villains is computed as the product of per-villain equities
    over the Cartesian product of villain classes. For k>=3 villains, each villain is
    capped to a k-aware top-N classes by weight (N = 500000**(1/k)) to bound worst-case
    enumeration. 2-villain case is always fully enumerated.

    Bias note: inter-villain card collisions are ignored — slight overestimate of
    multiway equity (<0.5% mean). Hero-villain collision is handled per villain.
    """
    if len(per_villain_combo_counts) == 1:
        return _matrix_cell_equity(hero_class, per_villain_combo_counts[0], matrix)

    row = matrix.get(hero_class)
    if row is None:
        return None

    # k-aware per-villain class cap so the Cartesian product stays bounded
    # (~500k tuples worst case): k=2 unlimited, k=3→79, k=4→26, k=5→13, k>=6→9.
    k = len(per_villain_combo_counts)
    top_k = int(500_000 ** (1.0 / k)) if k >= 3 else None

    def _build_pairs(combo_counts: dict[str, float]) -> list[tuple[str, float, float]]:
        """(class, equity, weight) for each villain class, collision-adjusted."""
        out = []
        for vclass, w in combo_counts.items():
            if w <= 0:
                continue
            e = row.get(vclass)
            if e is None:
                continue
            adj = w * _combo_collision_factor(hero_class, vclass)
            if adj > 0:
                out.append((vclass, e, adj))
        return out

    villain_pairs = [_build_pairs(cc) for cc in per_villain_combo_counts]

    if top_k is not None:
        villain_pairs = [sorted(vp, key=lambda x: -x[2])[:top_k] for vp in villain_pairs]

    # Cartesian product over villains
    joint: list[tuple[float, float]] = []  # (e_joint, w_joint)
    from itertools import product as iproduct

    for combo in iproduct(*villain_pairs):
        e_joint = 1.0
        w_joint = 1.0
        for _vclass, e_i, w_i in combo:
            e_joint *= e_i
            w_joint *= w_i
        if w_joint > 0:
            joint.append((e_joint, w_joint))

    if not joint:
        return None

    joint.sort(key=lambda x: x[0])
    eqs = np.array([p[0] for p in joint], dtype=np.float64)
    wts = np.array([p[1] for p in joint], dtype=np.float64)
    total = wts.sum()
    if total <= 0:
        return None

    mean_eq = float((eqs * wts).sum() / total)
    cdf = np.cumsum(wts) / total
    deciles = [float(np.interp(p / 100.0, cdf, eqs)) for p in _PERCENTILES]
    return {
        "mean_equity": mean_eq,
        **{f"p{p}": d for p, d in zip(_PERCENTILES, deciles, strict=True)},
    }


def _combo_collision_factor(hero_class: str, villain_class: str) -> float:
    """Fraction of villain combos that survive after removing hero's 2 cards.

    Cheap rank-overlap accounting on the canonical class strings — exact enough
    for range weighting (the matrix equity itself is already collision-exact).
    """
    h = hero_class
    v = villain_class
    hero_ranks = [h[0]] if len(h) == 2 else [h[0], h[1]]
    vr = [v[0]] if len(v) == 2 else [v[0], v[1]]
    # Count how many of villain's 2 ranks are blocked by a hero rank.
    shared = sum(1 for r in vr if r in hero_ranks)
    if shared == 0:
        return 1.0
    # Coarse survival: each shared rank loses ~1 of 4 suits of availability.
    return max(0.0, 1.0 - 0.25 * shared)


def _compute_cell_equity(
    hero_class: str,
    scenario_key: str,
    combo_counts: dict[str, int],
    n_trials: int,
    seed: int,
) -> dict[str, float] | None:
    """Run MC equity for one (hero_class, scenario) cell.

    Returns dict with mean_equity + p10..p90 deciles, or None on failure.
    """
    rng = random.Random(seed)

    hero_cards = _hero_combo_for_class(hero_class)
    hero_set = set(hero_cards)

    # Build weighted sampling list: [(class, weight), ...]
    classes = list(combo_counts.keys())
    weights = [float(combo_counts[c]) for c in classes]
    total_w = sum(weights)
    if total_w == 0:
        return None
    cum_weights = []
    running = 0.0
    for w in weights:
        running += w / total_w
        cum_weights.append(running)

    scores: list[float] = []
    skipped = 0

    for _ in range(n_trials):
        # Sample villain class by weight
        r = rng.random()
        idx = 0
        for i, cw in enumerate(cum_weights):
            if r <= cw:
                idx = i
                break
        villain_class = classes[idx]

        # Sample villain combo avoiding hero cards
        villain_cards = _sample_villain_combo(villain_class, hero_set, rng)
        if villain_cards is None:
            skipped += 1
            continue

        # Remove hero + villain cards from deck
        used = hero_set | set(villain_cards)
        remaining = [c for c in _DECK if c not in used]

        if len(remaining) < 5:
            skipped += 1
            continue

        # Deal 5 random board cards
        board = rng.sample(remaining, 5)

        # Evaluate
        hero_rank, hero_tb = _best_5_from_7(hero_cards, board)
        vill_rank, vill_tb = _best_5_from_7(villain_cards, board)

        if (hero_rank, hero_tb) > (vill_rank, vill_tb):
            scores.append(1.0)
        elif (hero_rank, hero_tb) == (vill_rank, vill_tb):
            scores.append(0.5)
        else:
            scores.append(0.0)

    effective = n_trials - skipped
    if effective < 10:
        return None

    arr = np.array(scores, dtype=np.float64)
    mean_eq = float(arr.mean())
    deciles = [float(np.percentile(arr, p)) for p in _PERCENTILES]

    return {
        "mean_equity": mean_eq,
        "p10": deciles[0],
        "p20": deciles[1],
        "p30": deciles[2],
        "p40": deciles[3],
        "p50": deciles[4],
        "p60": deciles[5],
        "p70": deciles[6],
        "p80": deciles[7],
        "p90": deciles[8],
    }


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------


def _worker_init(matrix_path: Path) -> None:
    """Pool initializer: load the 169x169 matrix once per worker process."""
    global _MATRIX
    _MATRIX = _load_matrix(matrix_path)


def _worker(args_tuple: tuple) -> dict | None:
    """Pool worker: compute equity for one cell.

    Args: (hero_class, scenario_key, combo_counts_or_list, n_trials, seed)
    combo_counts_or_list may be a single dict (HU) or list[dict] (multiway).
    Returns: row dict or None on skip.
    """
    hero_class, scen_key, combo_counts_or_list, n_trials, seed = args_tuple
    if _MATRIX is not None:
        if isinstance(combo_counts_or_list, list):
            result = _matrix_cell_equity_multiway(hero_class, combo_counts_or_list, _MATRIX)
        else:
            result = _matrix_cell_equity(hero_class, combo_counts_or_list, _MATRIX)
    else:
        # MC path: flatten list to combined dict for legacy single-villain MC
        if isinstance(combo_counts_or_list, list):
            combined: dict[str, int] = {}
            for cc in combo_counts_or_list:
                for hc, cnt in cc.items():
                    combined[hc] = combined.get(hc, 0) + int(cnt)
            result = _compute_cell_equity(hero_class, scen_key, combined, n_trials, seed)
        else:
            result = _compute_cell_equity(hero_class, scen_key, combo_counts_or_list, n_trials, seed)
    if result is None:
        return None

    # Build canonical hero hole cards + empty board for parquet schema
    hero_cards = _hero_combo_for_class(hero_class)
    hole_canonical, board_canonical = joint_canonicalize(hero_cards, [])

    row_id = f"{hole_canonical}|{scen_key}"

    return {
        "id": row_id,
        "street": "preflop",
        "hole_canonical": hole_canonical,
        "board_canonical": board_canonical,
        "scenario": scen_key,
        "mean_equity": result["mean_equity"],
        "p10": result["p10"],
        "p20": result["p20"],
        "p30": result["p30"],
        "p40": result["p40"],
        "p50": result["p50"],
        "p60": result["p60"],
        "p70": result["p70"],
        "p80": result["p80"],
        "p90": result["p90"],
    }


# ---------------------------------------------------------------------------
# Scenario enumeration from hm_decisions.jsonl
# ---------------------------------------------------------------------------


def _enumerate_preflop_scenarios(decisions_path: Path) -> set[str]:
    """Scan hm_decisions.jsonl for unique preflop scenario keys."""
    scenarios: set[str] = set()
    with decisions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dp = json.loads(line)
                if dp.get("street") != "preflop":
                    continue
                scen = _compute_scenario_key(dp)
                scenarios.add(scen)
            except (json.JSONDecodeError, KeyError):
                continue
    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Build preflop equity table via MC simulation")
    ap.add_argument(
        "--decisions",
        required=True,
        type=Path,
        help="Path to hm_decisions.jsonl",
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output parquet path (e.g. tools/preflop_equity_table.parquet)",
    )
    ap.add_argument("--n-trials", type=int, default=500, help="MC trials per cell (default 500)")
    ap.add_argument(
        "--method",
        choices=["matrix", "mc"],
        default="matrix",
        help="matrix: exact weighted-quantile equity from the 169x169 table (default); mc: legacy Monte Carlo",
    )
    ap.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).resolve().parent / "preflop_hand_vs_hand_169.parquet",
        help="169x169 exact equity parquet (matrix method)",
    )
    ap.add_argument("--threads", type=int, default=12, help="Parallel worker processes")
    ap.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    ap.add_argument(
        "--limit-scenarios",
        type=int,
        default=0,
        help="Cap number of scenarios (0=all; for testing)",
    )
    ap.add_argument(
        "--only-pot-type",
        type=str,
        default=None,
        help="Build only scenarios for this pot_type (srp/3bet/4bet/5bet+/limp). For incremental rebuilds.",
    )
    ap.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="Existing parquet to merge with: its rows for OTHER pot_types are kept, "
        "rows for --only-pot-type are replaced by this run. Requires --only-pot-type.",
    )
    args = ap.parse_args()
    if args.merge_into and not args.only_pot_type:
        ap.error("--merge-into requires --only-pot-type")

    # Configure logging
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    if args.log_file:
        fh = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)

    log.info(
        "preflop_equity.start method=%s n_trials=%d threads=%d",
        args.method,
        args.n_trials,
        args.threads,
    )
    if args.method == "matrix" and not args.matrix.exists():
        log.error("matrix parquet not found: %s (build via tools/build_preflop_hand_vs_hand.py)", args.matrix)
        sys.exit(1)

    # Step 1: Enumerate preflop scenarios
    log.info("preflop_equity.enumerate_scenarios path=%s", args.decisions)
    if not args.decisions.exists():
        log.error("decisions file not found: %s", args.decisions)
        sys.exit(1)

    scenarios = _enumerate_preflop_scenarios(args.decisions)
    log.info("preflop_equity.scenarios_found n=%d", len(scenarios))

    if args.only_pot_type:
        scenarios = {s for s in scenarios if s.split("|")[0] == args.only_pot_type}
        log.info("preflop_equity.scenarios_filtered pot_type=%s n=%d", args.only_pot_type, len(scenarios))

    if args.limit_scenarios:
        scenarios = set(list(scenarios)[: args.limit_scenarios])
        log.info("preflop_equity.scenarios_capped n=%d", len(scenarios))

    # Build pooled field range once for vNA dissolve (fix #1)
    pooled_field_range = _build_pooled_field_range(PALETTE_DIR)
    log.info("preflop_equity.pooled_field_range_built n_classes=%d", len(pooled_field_range))

    # Step 2: Filter scenarios to those with resolvable palette
    scen_combos: dict[str, list[dict[str, float]]] = {}
    skipped_no_palette = 0
    for scen in scenarios:
        cc = _load_palette_combo_counts(scen, pooled_field_range=pooled_field_range)
        if cc is None:
            skipped_no_palette += 1
            continue
        scen_combos[scen] = cc

    log.info(
        "preflop_equity.scenarios_resolved n=%d skipped_no_palette=%d",
        len(scen_combos),
        skipped_no_palette,
    )

    # Step 3: Build cell list (hero_class x scenario)
    cells: list[tuple] = []
    base_seed = 42
    for scen_idx, (scen, combo_counts) in enumerate(scen_combos.items()):
        for class_idx, hero_class in enumerate(ALL_HAND_CLASSES):
            seed = base_seed + scen_idx * 10000 + class_idx
            cells.append((hero_class, scen, combo_counts, args.n_trials, seed))

    log.info("preflop_equity.cells_total n=%d", len(cells))

    # Step 4: Process cells in parallel
    t0 = time.time()
    rows: list[dict] = []
    n_ok = 0
    n_skip = 0

    pool_init = _worker_init if args.method == "matrix" else None
    pool_args = (args.matrix,) if args.method == "matrix" else ()
    with multiprocessing.Pool(processes=args.threads, initializer=pool_init, initargs=pool_args) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, cells, chunksize=50)):
            if result is None:
                n_skip += 1
            else:
                rows.append(result)
                n_ok += 1

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                eta = (len(cells) - i - 1) / max(rate, 1e-9)
                log.info(
                    "preflop_equity.progress processed=%d ok=%d skip=%d rate=%.0f/s eta=%.1fmin",
                    i + 1,
                    n_ok,
                    n_skip,
                    rate,
                    eta / 60,
                )

    elapsed = time.time() - t0
    log.info(
        "preflop_equity.done ok=%d skip=%d elapsed=%.1fs",
        n_ok,
        n_skip,
        elapsed,
    )

    if not rows:
        log.error("preflop_equity.no_rows_produced — aborting")
        sys.exit(1)

    # Step 5: Write parquet (schema matches equity_table.parquet exactly)
    ids = [r["id"] for r in rows]
    streets = [r["street"] for r in rows]
    holes = [r["hole_canonical"] for r in rows]
    boards = [r["board_canonical"] for r in rows]
    scens = [r["scenario"] for r in rows]
    means = [r["mean_equity"] for r in rows]
    decile_cols = [[r[f"p{p}"] for r in rows] for p in _PERCENTILES]

    table = pa.table(
        {
            "id": ids,
            "street": streets,
            "hole_canonical": holes,
            "board_canonical": boards,
            "scenario": scens,
            "mean_equity": means,
            "p10": decile_cols[0],
            "p20": decile_cols[1],
            "p30": decile_cols[2],
            "p40": decile_cols[3],
            "p50": decile_cols[4],
            "p60": decile_cols[5],
            "p70": decile_cols[6],
            "p80": decile_cols[7],
            "p90": decile_cols[8],
        }
    )

    if args.merge_into:
        # Keep all rows from the existing table whose pot_type differs from this run,
        # then append the freshly built rows for --only-pot-type.
        existing = pq.read_table(args.merge_into)
        keep_mask = [s.split("|")[0] != args.only_pot_type for s in existing.column("scenario").to_pylist()]
        kept = existing.filter(pa.array(keep_mask))
        n_dropped = existing.num_rows - kept.num_rows
        table = pa.concat_tables([kept, table.select(existing.column_names)])
        log.info(
            "preflop_equity.merged base=%s dropped_old=%d added_new=%d total=%d",
            args.merge_into,
            n_dropped,
            len(rows),
            table.num_rows,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    log.info(
        "preflop_equity.wrote path=%s rows=%d size_kb=%d",
        args.out,
        table.num_rows,
        args.out.stat().st_size // 1024,
    )
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB, {table.num_rows:,} rows)")


if __name__ == "__main__":
    main()
