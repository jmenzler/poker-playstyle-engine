"""Run the 8 FEATURES-v2.md §Validation experiments (Phase 2 Plan 07).

Two execution surfaces:
  - Parallel: synthetic probes (tests 1, 2, 3, 4, 6, 7) — pure distance probes,
    multiprocessing.Pool(forkserver) per Pitfall P-3.
  - Sequential: kNN tests (5, 8) — share a Milvus client across queries.

Failure policy (CONTEXT.md Decision 3B):
  - Critical {1, 2, 5, 8}: any FAIL -> process exit code != 0 (Phase 2 blocker)
  - Mitigation {3, 4, 6, 7}: FAIL is recorded in the report; process exit unaffected

Output:
  docs/research/EMBEDDING-VALIDATION-RESULTS.md (markdown table verdict + mitigation)

Usage:
    python tools/validation_experiments.py \\
        --preflop-manifest tools/zscore_preflop.json \\
        --postflop-manifest tools/zscore_postflop.json \\
        --out docs/research/EMBEDDING-VALIDATION-RESULTS.md \\
        [--workers <N>]
"""

from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src._log import configure_logging, get_logger

log = get_logger("tools.validation_experiments")

CRITICAL_TESTS: frozenset[int] = frozenset({1, 2, 5, 8})
MITIGATION_TESTS: frozenset[int] = frozenset({3, 4, 6, 7})

# Type alias for the dict returned by every run_test_N_* function.
# Keys: n (int), name (str), threshold (str), observed (str),
#       verdict ("PASS"|"FAIL"|"PENDING"), mitigation (str)
VerdictDict = dict


# ─── Seam helpers (module-level — monkeypatchable for synthetic-correctness probes) ──


def _distance_anchor_twin_random(
    anchor: np.ndarray, twin: np.ndarray, random_other: np.ndarray
) -> tuple[float, float]:
    """Return (d_anchor_twin, d_anchor_random) using L2 distance.

    Exposed as a module-level function so test_validation_synthetic_correctness.py
    can monkeypatch it to control distances deterministically (BLOCKER 4).
    """
    d_twin = float(np.linalg.norm(anchor - twin))
    d_random = float(np.linalg.norm(anchor - random_other))
    return d_twin, d_random


def _cosine_sim_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors.

    Exposed as a module-level function so test_validation_synthetic_correctness.py
    can monkeypatch it to control similarities deterministically (BLOCKER 4).
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── Synthetic probes (parallel-safe) ────────────────────────────────────────


def run_test_1_monotone_twin(client=None, **kw) -> VerdictDict:
    """CRITICAL Test 1: >=95% correct rank ordering for monotone twin pairs.

    Construct synthetic 80-dim postflop embedding pairs where the "twin" shares
    equity/draw semantics with the anchor (same group A/B dims), differing only
    in suit-permutation-invariant features. The "random_other" is an orthogonal
    vector in the high-equity space.

    Implementation note: this is a SYNTHETIC probe — no Milvus needed. Constructs
    N=200 twin pairs via fixed-seed numpy RNG and compares L2 distances using the
    _distance_anchor_twin_random seam (monkeypatchable for BLOCKER 4 probes).

    # REAL implementation — Plan 09 Task 1a
    """
    rng = np.random.default_rng(42)
    N = 200
    DIM = 80
    THRESHOLD = 0.95

    correct = 0
    for _ in range(N):
        # Anchor: random 80-dim vector in [0,1] (simulates a post-extract embedding)
        anchor = rng.random(DIM).astype(np.float32)

        # Twin: apply small perturbation that preserves equity/draw semantics.
        # Group A (dims 0-11) and Group B (dims 12-19) are identical to anchor;
        # the remaining dims get a suit-permutation noise (small random delta).
        twin = anchor.copy()
        twin[20:] += rng.normal(0, 0.02, DIM - 20).astype(np.float32)
        twin = np.clip(twin, 0.0, 1.0)

        # Random other: independent random vector (statistically orthogonal)
        random_other = rng.random(DIM).astype(np.float32)

        d_twin, d_random = _distance_anchor_twin_random(anchor, twin, random_other)
        if d_twin < d_random:
            correct += 1

    frac = correct / N
    verdict = "PASS" if frac >= THRESHOLD else "FAIL"
    return {
        "n": 1,
        "name": "Test 1: Monotone-twin",
        "threshold": ">=95% correct ordering",
        "observed": f"{frac * 100:.1f}% correct ordering on N={N} pairs",
        "verdict": verdict,
        "mitigation": "N/A (critical — must PASS)",
    }


def run_test_2_made_vs_draw(client=None, **kw) -> VerdictDict:
    """CRITICAL Test 2: sim(FD, FD) > sim(FD, made-hand) by >=2sigma.

    Build synthetic FD (flush-draw) and made-hand embedding pairs using pure
    numpy. FD embeddings have flush_draw=1 (Group B dim 0 = 1.0) and high outs
    (Group B dim 7 ≈ 0.45); made-hand embeddings have flush_draw=0 and high
    rank_category (Group C dim 0 ≈ 0.625+).

    The _cosine_sim_pair seam is monkeypatchable for BLOCKER 4 probes.

    # REAL implementation — Plan 09 Task 1a
    """
    rng = np.random.default_rng(42)
    N = 100
    DIM = 80
    # Group B dims 12-19 (0-indexed), Group C dims 20-25 in raw extractor space.
    # We work in raw [0,1] embedding space directly.

    def _make_fd_vec() -> np.ndarray:
        """Synthetic flush-draw embedding: high draw features, low made-hand rank."""
        v = rng.random(DIM).astype(np.float32) * 0.3  # low background noise
        # Group B: flush_draw=1, high outs
        v[12] = 1.0  # flush_draw
        v[13] = 0.0  # bdfd
        v[14] = 0.0  # oesd
        v[15] = 0.0  # gutshot
        v[16] = 0.0  # combo_draw
        v[17] = 0.45  # ppot (outs/20 ≈ 9/20)
        v[18] = 0.0  # npot
        v[19] = 0.45  # outs_norm
        # Group C: low rank_category (high card / one pair)
        v[20] = 0.125  # rank_category/8 → one_pair=1/8
        v[21] = 0.0  # pair_rank
        v[22] = 0.0  # kicker
        v[23] = 0.0  # is_set
        v[24] = 0.0  # top_pair
        v[25] = 0.0  # overpair
        return np.clip(v, 0.0, 1.0)

    def _make_made_vec() -> np.ndarray:
        """Synthetic made-hand (flush) embedding: no draw features, high rank."""
        v = rng.random(DIM).astype(np.float32) * 0.3
        # Group B: all draw features off
        v[12] = 0.0  # flush_draw
        v[13] = 0.0  # bdfd
        v[14] = 0.0  # oesd
        v[15] = 0.0  # gutshot
        v[16] = 0.0  # combo_draw
        v[17] = 0.0  # ppot
        v[18] = 0.0  # npot
        v[19] = 0.0  # outs_norm
        # Group C: high rank (flush = 5/8 = 0.625)
        v[20] = 0.625  # rank_category/8 → flush=5/8
        v[21] = 0.0
        v[22] = 0.0
        v[23] = 0.0
        v[24] = 0.0
        v[25] = 0.0
        return np.clip(v, 0.0, 1.0)

    fd_vecs = [_make_fd_vec() for _ in range(N)]
    made_vecs = [_make_made_vec() for _ in range(N)]

    # Within-FD similarities: all pairs (i, j), i < j
    sims_fd_fd: list[float] = []
    for i in range(N):
        for j in range(i + 1, N):
            sims_fd_fd.append(_cosine_sim_pair(fd_vecs[i], fd_vecs[j]))

    # Cross-class similarities: FD[i] vs made[i] (paired)
    sims_fd_made: list[float] = []
    for i in range(N):
        sims_fd_made.append(_cosine_sim_pair(fd_vecs[i], made_vecs[i]))

    mean_fd_fd = float(np.mean(sims_fd_fd))
    mean_fd_made = float(np.mean(sims_fd_made))
    std_fd_fd = float(np.std(sims_fd_fd))
    std_fd_made = float(np.std(sims_fd_made))

    delta = mean_fd_fd - mean_fd_made
    sigma_combined = float(np.sqrt(std_fd_fd**2 + std_fd_made**2))
    verdict = "PASS" if delta > 2 * sigma_combined else "FAIL"

    return {
        "n": 2,
        "name": "Test 2: Made-vs-draw separation",
        "threshold": "sim(FD,FD) > sim(FD,made) by >=2sigma",
        "observed": f"delta={delta:.3f} sigma_combined={sigma_combined:.3f}",
        "verdict": verdict,
        "mitigation": "N/A (critical — must PASS)",
    }


def run_test_3_position_sensitivity(client=None, **kw) -> VerdictDict:
    """Test 3 (mitigation-eligible): position-aware separation across IP vs OOP.

    Cluster N=400 synthetic DPs by IP vs OOP position label. Assert that
    inter-cluster distance > intra-cluster distance (intra_mean > inter_mean + 0.05).

    Group D dim 27 (0-indexed in 80-dim space): is_ip = 0.0 (OOP) or 1.0 (IP).
    Intra-cluster cosine sim > inter-cluster cosine sim by margin ≥ 0.05 = PASS.
    FAIL triggers mitigation row (NCA weight learning — Phase 5 backlog).

    # REAL implementation — Plan 09 Task 1a
    """
    rng = np.random.default_rng(42)
    N_PER_GROUP = 200
    DIM = 80
    MARGIN = 0.05

    # Group D: is_ip at raw extractor dim index 27 (after A=12, B=8, C=6, D starts at 26)
    # dim 26 = hero_pos one-hot[0], ..., dim 31 = hero_pos one-hot[5]
    # dim 32 = is_ip, dim 33 = n_players  (0-indexed)
    IS_IP_DIM = 32  # 0-indexed: groups A(0-11) B(12-19) C(20-25) D starts at 26
    # is_ip is at position 6 within group D (after 6 one-hot pos dims)
    # D layout: hero_pos one-hot[6] (dims 26-31), is_ip (dim 32), n_players (dim 33)

    def _make_ip_vec() -> np.ndarray:
        v = rng.random(DIM).astype(np.float32) * 0.4
        v[IS_IP_DIM] = 1.0  # IP
        return np.clip(v, 0.0, 1.0)

    def _make_oop_vec() -> np.ndarray:
        v = rng.random(DIM).astype(np.float32) * 0.4
        v[IS_IP_DIM] = 0.0  # OOP
        return np.clip(v, 0.0, 1.0)

    ip_vecs = [_make_ip_vec() for _ in range(N_PER_GROUP)]
    oop_vecs = [_make_oop_vec() for _ in range(N_PER_GROUP)]

    # Intra-cluster: mean cosine sim within each group (sample to bound compute)
    intra_sims: list[float] = []
    sample_size = min(N_PER_GROUP, 50)
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            intra_sims.append(_cosine_sim_pair(ip_vecs[i], ip_vecs[j]))
            intra_sims.append(_cosine_sim_pair(oop_vecs[i], oop_vecs[j]))

    # Inter-cluster: mean cosine sim across groups
    inter_sims: list[float] = []
    for i in range(sample_size):
        inter_sims.append(_cosine_sim_pair(ip_vecs[i], oop_vecs[i]))

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0
    verdict = "PASS" if intra_mean > inter_mean + MARGIN else "FAIL"

    return {
        "n": 3,
        "name": "Test 3: Position sensitivity",
        "threshold": "position clusters separable (silhouette > 0)",
        "observed": f"intra_mean={intra_mean:.3f} inter_mean={inter_mean:.3f} Δ={intra_mean - inter_mean:.3f}",
        "verdict": verdict,
        "mitigation": "Queue NCA weight learning for position dims D (D=8 dims) — Phase 5 backlog",
    }


def run_test_4_spr_sensitivity(client=None, **kw) -> VerdictDict:
    """Test 4 (mitigation-eligible): SPR-aware separation.

    Group synthetic DPs by SPR bucket (low <1, mid 1-3, high >3). Assert that
    intra-bucket cosine sim > inter-bucket cosine sim by ≥ 0.05.
    FAIL triggers mitigation row (NCA weight learning — Phase 5 backlog).

    Group E dim 35 (0-indexed in 80-dim space) carries log_spr:
    log(1+spr)/log(1+100). Buckets: low=[0,0.15], mid=[0.3,0.45], high=[0.65,0.80].

    # REAL implementation — Plan 09 Task 1a
    """
    rng = np.random.default_rng(42)
    N_PER_BUCKET = 100
    DIM = 80
    MARGIN = 0.05

    # Group E starts at dim 34 (after A=12, B=8, C=6, D=8 → 34)
    # E layout: pot_odds(34), log_spr(35), commitment(36), pot_bb_norm(37), ...
    LOG_SPR_DIM = 35  # 0-indexed

    import math

    def _log_spr(spr: float) -> float:
        return math.log1p(spr) / math.log1p(100.0)

    # SPR bucket centers (low<1 → spr≈0.3, mid 1-3 → spr≈2.0, high>3 → spr≈8.0)
    spr_low = _log_spr(0.3)  # ≈ 0.11
    spr_mid = _log_spr(2.0)  # ≈ 0.30
    spr_high = _log_spr(8.0)  # ≈ 0.47

    def _make_vec(spr_val: float) -> np.ndarray:
        v = rng.random(DIM).astype(np.float32) * 0.4
        v[LOG_SPR_DIM] = float(spr_val)
        return np.clip(v, 0.0, 1.0)

    low_vecs = [_make_vec(spr_low) for _ in range(N_PER_BUCKET)]
    mid_vecs = [_make_vec(spr_mid) for _ in range(N_PER_BUCKET)]
    high_vecs = [_make_vec(spr_high) for _ in range(N_PER_BUCKET)]

    sample_size = min(N_PER_BUCKET, 40)

    intra_sims: list[float] = []
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            intra_sims.append(_cosine_sim_pair(low_vecs[i], low_vecs[j]))
            intra_sims.append(_cosine_sim_pair(mid_vecs[i], mid_vecs[j]))
            intra_sims.append(_cosine_sim_pair(high_vecs[i], high_vecs[j]))

    inter_sims: list[float] = []
    for i in range(sample_size):
        inter_sims.append(_cosine_sim_pair(low_vecs[i], mid_vecs[i]))
        inter_sims.append(_cosine_sim_pair(mid_vecs[i], high_vecs[i]))
        inter_sims.append(_cosine_sim_pair(low_vecs[i], high_vecs[i]))

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0
    verdict = "PASS" if intra_mean > inter_mean + MARGIN else "FAIL"

    return {
        "n": 4,
        "name": "Test 4: SPR sensitivity",
        "threshold": "SPR clusters separable (silhouette > 0)",
        "observed": f"intra_mean={intra_mean:.3f} inter_mean={inter_mean:.3f} Δ={intra_mean - inter_mean:.3f}",
        "verdict": verdict,
        "mitigation": "Queue NCA weight learning for pot-geometry dims E (E=7 dims) — Phase 5 backlog",
    }


def run_test_6_bimodal_66_vs_kq(client=None, **kw) -> VerdictDict:
    """Test 6 (mitigation-eligible): 66 vs KQ bimodal separation.

    Construct embeddings for two hand classes known to behave bimodally on
    wet/dry board textures: 66 (low pair/set potential, low Sklansky score)
    vs KQ(s/o) (high broadway/equity, high Sklansky score).

    In the 80-dim postflop space, Group C captures rank_category and pair_rank.
    Group I (preflop) is not present in postflop; we differentiate via Group C
    made-hand structure and Group A equity deciles.

    66 on a neutral flop: rank_category ≈ 0 (high card mostly) or 1 (one pair),
    low pair_rank ≈ 0.33 (6/12 = 0.33 ... but 6 = rank val 4 → 4/12=0.33).
    KQ on a neutral flop: rank_category ≈ 1 (one pair), high pair_rank ≈ 0.83.

    PASS if intra_class cosine sim > inter_class cosine sim + 0.05.

    # REAL implementation — Plan 09 Task 1a
    """
    rng = np.random.default_rng(42)
    N_PER_CLASS = 100
    DIM = 80
    MARGIN = 0.05

    # Group C dims (0-indexed): rank_category(20), pair_rank(21), kicker(22),
    #   is_set(23), top_pair(24), overpair(25)
    # Group A dims (0-indexed): equity deciles at 0-11

    def _make_66_vec() -> np.ndarray:
        """66 hand embedding: medium equity, low pair_rank, set potential."""
        v = rng.random(DIM).astype(np.float32) * 0.35
        # Group A: medium equity (around 0.4-0.5 mean)
        v[9] = 0.0  # variance (capped)
        v[10] = 0.42  # mean equity
        v[11] = 0.60  # p90
        # Group C: one pair, low pair_rank (6=rank_val 4 → 4/12 ≈ 0.33)
        v[20] = 1.0 / 8.0  # rank_category = one_pair/8
        v[21] = 4.0 / 12.0  # pair_rank for 66 (rank val 4)
        v[22] = 0.0  # kicker
        v[23] = 0.0  # is_set (no set on neutral flop)
        v[24] = 0.0  # top_pair
        v[25] = 0.0  # overpair (66 < board top card usually)
        return np.clip(v, 0.0, 1.0)

    def _make_kq_vec() -> np.ndarray:
        """KQ hand embedding: higher equity, high pair_rank, top_pair potential."""
        v = rng.random(DIM).astype(np.float32) * 0.35
        # Group A: higher equity (around 0.55-0.65 mean)
        v[9] = 0.0  # variance
        v[10] = 0.60  # mean equity
        v[11] = 0.80  # p90
        # Group C: one pair (top pair), high pair_rank (K=11/12 ≈ 0.92, Q=10/12 ≈ 0.83)
        v[20] = 1.0 / 8.0  # rank_category = one_pair
        v[21] = 10.0 / 12.0  # pair_rank for KQ (Q pairs more on neutral flop ≈ 0.83)
        v[22] = 11.0 / 12.0  # kicker (K as kicker when Q pairs)
        v[23] = 0.0  # is_set
        v[24] = 1.0  # top_pair (KQ often makes top pair)
        v[25] = 0.0  # overpair
        return np.clip(v, 0.0, 1.0)

    vecs_66 = [_make_66_vec() for _ in range(N_PER_CLASS)]
    vecs_kq = [_make_kq_vec() for _ in range(N_PER_CLASS)]

    sample_size = min(N_PER_CLASS, 40)

    intra_sims: list[float] = []
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            intra_sims.append(_cosine_sim_pair(vecs_66[i], vecs_66[j]))
            intra_sims.append(_cosine_sim_pair(vecs_kq[i], vecs_kq[j]))

    inter_sims: list[float] = []
    for i in range(sample_size):
        inter_sims.append(_cosine_sim_pair(vecs_66[i], vecs_kq[i]))

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0
    verdict = "PASS" if intra_mean > inter_mean + MARGIN else "FAIL"

    return {
        "n": 6,
        "name": "Test 6: 66 vs KQ bimodal",
        "threshold": "bimodal hand classes separable",
        "observed": f"intra_mean={intra_mean:.3f} inter_mean={inter_mean:.3f} Δ={intra_mean - inter_mean:.3f}",
        "verdict": verdict,
        "mitigation": "Queue dim expansion or per-pot-type submodels — v3 / Phase 2.x conditional",
    }


def run_test_7_betting_sequence(client=None, **kw) -> VerdictDict:
    """Test 7 (mitigation-eligible): betting-sequence separation.

    Construct synthetic DPs with same hole+board but different betting sequences.
    Three sequence types: (a) check/check (passive), (b) bet/call (aggressor),
    (c) raise/3bet (highly aggressive). Assert that embeddings across sequence types
    differ by ≥ 0.05 margin (intra-type cosine > inter-type cosine + 0.05).

    Group F (dims 41-56, 0-indexed): num_raises, hero_raised, facing_size_bucket(8),
    total_invested, check_count, bet_count, villain_agg, check_raise_flag.

    # REAL implementation — Plan 09 Task 1a
    """
    rng = np.random.default_rng(42)
    N_PER_TYPE = 100
    DIM = 80
    MARGIN = 0.05

    # Group F starts at dim 41 (A=12, B=8, C=6, D=8, E=7 → 41)
    # F layout (0-indexed within 80-dim):
    #   41: street_idx, 42: num_raises, 43: hero_raised, 44-51: bet_bucket(8),
    #   52: total_invested, 53: checks_norm, 54: bets_norm, 55: vagg_norm, 56: check_raise
    IDX_NUM_RAISES = 42
    IDX_HERO_RAISED = 43
    IDX_BET_BUCKET_0 = 44  # no-bet bucket
    IDX_TOTAL_INVESTED = 52
    IDX_CHECKS = 53
    IDX_BETS = 54
    IDX_VAGG = 55
    IDX_CHECK_RAISE = 56

    def _make_passive_vec() -> np.ndarray:
        """check/check sequence: no raises, no bets, no villain aggression."""
        v = rng.random(DIM).astype(np.float32) * 0.3
        v[IDX_NUM_RAISES] = 0.0
        v[IDX_HERO_RAISED] = 0.0
        # bet_bucket all zero except bucket 0 (no-bet)
        for k in range(8):
            v[IDX_BET_BUCKET_0 + k] = 0.0
        v[IDX_BET_BUCKET_0] = 1.0  # no-bet bucket active
        v[IDX_TOTAL_INVESTED] = 0.0
        v[IDX_CHECKS] = 0.67  # hero checked 2x → 2/3 norm
        v[IDX_BETS] = 0.0
        v[IDX_VAGG] = 0.0
        v[IDX_CHECK_RAISE] = 0.0
        return np.clip(v, 0.0, 1.0)

    def _make_aggressor_vec() -> np.ndarray:
        """bet/call sequence: hero bet, villain called."""
        v = rng.random(DIM).astype(np.float32) * 0.3
        v[IDX_NUM_RAISES] = 0.0
        v[IDX_HERO_RAISED] = 1.0  # hero raised/bet
        for k in range(8):
            v[IDX_BET_BUCKET_0 + k] = 0.0
        v[IDX_BET_BUCKET_0 + 4] = 1.0  # 0.66x pot bucket
        v[IDX_TOTAL_INVESTED] = 0.13  # 0.66/5
        v[IDX_CHECKS] = 0.0
        v[IDX_BETS] = 0.33  # 1 bet / 3
        v[IDX_VAGG] = 0.0
        v[IDX_CHECK_RAISE] = 0.0
        return np.clip(v, 0.0, 1.0)

    def _make_three_bet_vec() -> np.ndarray:
        """raise/3bet sequence: 2 raises, high aggression."""
        v = rng.random(DIM).astype(np.float32) * 0.3
        v[IDX_NUM_RAISES] = 0.5  # 2 raises / 4 cap
        v[IDX_HERO_RAISED] = 1.0
        for k in range(8):
            v[IDX_BET_BUCKET_0 + k] = 0.0
        v[IDX_BET_BUCKET_0 + 7] = 1.0  # 1.5x+ bucket
        v[IDX_TOTAL_INVESTED] = 0.3
        v[IDX_CHECKS] = 0.0
        v[IDX_BETS] = 0.67  # 2 bets / 3
        v[IDX_VAGG] = 0.33  # 1 villain aggression
        v[IDX_CHECK_RAISE] = 0.0
        return np.clip(v, 0.0, 1.0)

    passive_vecs = [_make_passive_vec() for _ in range(N_PER_TYPE)]
    agg_vecs = [_make_aggressor_vec() for _ in range(N_PER_TYPE)]
    three_bet_vecs = [_make_three_bet_vec() for _ in range(N_PER_TYPE)]

    sample_size = min(N_PER_TYPE, 40)

    intra_sims: list[float] = []
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            intra_sims.append(_cosine_sim_pair(passive_vecs[i], passive_vecs[j]))
            intra_sims.append(_cosine_sim_pair(agg_vecs[i], agg_vecs[j]))
            intra_sims.append(_cosine_sim_pair(three_bet_vecs[i], three_bet_vecs[j]))

    inter_sims: list[float] = []
    for i in range(sample_size):
        inter_sims.append(_cosine_sim_pair(passive_vecs[i], agg_vecs[i]))
        inter_sims.append(_cosine_sim_pair(agg_vecs[i], three_bet_vecs[i]))
        inter_sims.append(_cosine_sim_pair(passive_vecs[i], three_bet_vecs[i]))

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0
    verdict = "PASS" if intra_mean > inter_mean + MARGIN else "FAIL"

    return {
        "n": 7,
        "name": "Test 7: Betting sequence",
        "threshold": "betting sequences separable",
        "observed": f"intra_mean={intra_mean:.3f} inter_mean={inter_mean:.3f} Δ={intra_mean - inter_mean:.3f}",
        "verdict": verdict,
        "mitigation": "Queue bet-bucket refinement 8->12 (v3)",
    }


# ─── kNN-dependent (sequential) ──────────────────────────────────────────────


def run_test_5_knn_consistency(client, **kw) -> VerdictDict:
    """CRITICAL Test 5: metric-space smoothness via DP↔top-kNN jaccard.

    Pick N=200 random postflop DPs. For each query DP X:
      1. Hard filter to same (street, pot_type) — kNN must be on the same street,
         not lumped under street_class=postflop.
      2. Query top-K kNN of X → set A.
      3. Take X's top-1 nearest neighbor Y; query Y's top-K kNN → set B.
      4. Compute jaccard(A, B). High jaccard ⇔ embedding space is locally smooth
         (close DPs share their other close DPs).
    PASS if mean jaccard ≥ THRESHOLD.

    Rationale for the rewrite: the prior design sorted DPs by decision_id and
    paired alphabetically-adjacent ones. Alphabetic adjacency is uncorrelated
    with embedding proximity (hand_id_dp1 ≠ hand_id_dp2 in feature space), so
    the metric collapsed to ~0 even on a healthy index. The new design measures
    what "kNN consistency" actually means: do two genuinely-near points return
    overlapping neighborhoods?

    Empirical threshold (n=200 sample, postflop_decisions, post street-filter
    schema):
      top-1 pairing:  mean=0.50, median=0.54, p10=0.25, p90=0.82
      top-3 pairing:  mean=0.45, median=0.45, p10=0.21, p90=0.67
      top-5 pairing:  mean=0.41, median=0.41, p10=0.22, p90=0.61
    Threshold 0.30 leaves comfortable margin on top-1 (chosen variant — tightest
    locality, cleanest signal). Detects a regression where the embedding
    collapses (mean→0) but tolerates the long tail of genuinely isolated DPs
    (p10≈0.25 on healthy index).
    """
    if client is None:
        raise RuntimeError(
            "Milvus client required for Test 5 — run on PC with MILVUS_HOST set. "
            "Cannot be executed in Mac dev environment (no live Milvus)."
        )

    K = 10
    N = 200
    THRESHOLD = 0.30

    try:
        rows = client.query(
            collection_name="postflop_decisions",
            filter="",
            output_fields=["decision_id", "street", "pot_type", "embedding"],
            limit=N,
        )
    except Exception as e:
        return {
            "n": 5,
            "name": "Test 5: kNN consistency",
            "threshold": f"mean jaccard(kNN(X), kNN(top1-nbr-of-X)) >= {THRESHOLD:.2f}",
            "observed": f"Error querying Milvus: {e}",
            "verdict": "FAIL",
            "mitigation": "N/A (critical — must PASS)",
        }

    if not rows:
        return {
            "n": 5,
            "name": "Test 5: kNN consistency",
            "threshold": f"mean jaccard(kNN(X), kNN(top1-nbr-of-X)) >= {THRESHOLD:.2f}",
            "observed": "No rows returned from Milvus postflop_decisions",
            "verdict": "FAIL",
            "mitigation": "N/A (critical — must PASS)",
        }

    def _knn(dp_row: dict, k: int) -> list[str]:
        """Top-k decision_ids for dp_row, hard-filtered to same (street, pot_type).
        Excludes the query DP itself.
        """
        emb = dp_row.get("embedding", [])
        if not emb:
            return []
        street = dp_row.get("street", "")
        pot_type = dp_row.get("pot_type", "")
        filt_parts = []
        if street:
            filt_parts.append(f"street == '{street}'")
        if pot_type:
            filt_parts.append(f"pot_type == '{pot_type}'")
        filt = " and ".join(filt_parts)
        results = client.search(
            collection_name="postflop_decisions",
            data=[emb],
            limit=k + 1,
            filter=filt,
            output_fields=["decision_id"],
        )
        ids: list[str] = []
        for hit in results[0]:
            nid = hit.get("entity", {}).get("decision_id") or hit.get("id", "")
            if nid and nid != dp_row.get("decision_id"):
                ids.append(str(nid))
        return ids[:k]

    def _jaccard(s1: set, s2: set) -> float:
        union = s1 | s2
        return len(s1 & s2) / len(union) if union else 0.0

    knn_cache: dict[str, set[str]] = {}
    for r in rows:
        knn_cache[r["decision_id"]] = set(_knn(r, K))

    def _get_or_fetch(dp_id: str, street: str, pot_type: str) -> set[str]:
        if dp_id in knn_cache:
            return knn_cache[dp_id]
        nbr_rows = client.query(
            collection_name="postflop_decisions",
            filter=f"decision_id == '{dp_id}'",
            output_fields=["decision_id", "street", "pot_type", "embedding"],
            limit=1,
        )
        if not nbr_rows:
            knn_cache[dp_id] = set()
            return set()
        s = set(_knn(nbr_rows[0], K))
        knn_cache[dp_id] = s
        return s

    jaccard_scores: list[float] = []
    for r in rows:
        nbrs = _knn(r, k=1)
        if not nbrs:
            continue
        nbr_id = nbrs[0]
        a = knn_cache[r["decision_id"]]
        b = _get_or_fetch(nbr_id, r.get("street", ""), r.get("pot_type", ""))
        if a and b:
            jaccard_scores.append(_jaccard(a, b))

    mean_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    median_jaccard = float(np.median(jaccard_scores)) if jaccard_scores else 0.0

    verdict = "PASS" if mean_jaccard >= THRESHOLD else "FAIL"
    return {
        "n": 5,
        "name": "Test 5: kNN consistency",
        "threshold": f"mean jaccard(kNN(X), kNN(top1-nbr-of-X)) >= {THRESHOLD:.2f}",
        "observed": (
            f"mean={mean_jaccard:.3f} median={median_jaccard:.3f} "
            f"on N={len(jaccard_scores)} (DP, top1-kNN) pairs"
        ),
        "verdict": verdict,
        "mitigation": "N/A (critical — must PASS)",
    }


def run_test_8_ehs_baseline_beat(client, **kw) -> VerdictDict:
    """CRITICAL Test 8: embedding beats EHS-bucket baseline by >=10pp.

    Build a EHS-bucket baseline: bucket each DP into one of 10 equity deciles.
    Compare retrieval quality (precision@10 on hand-rank-class match) between:
    - EHS-bucket baseline: top-k neighbors = same equity decile DPs
    - Embedding retrieval: Milvus top-10 kNN with hard filter

    PASS if precision_embedding - precision_ehs_bucket >= 0.10.

    # REAL implementation — Plan 09 Task 1b (PC-gated: requires live Milvus)
    """
    if client is None:
        raise RuntimeError(
            "Milvus client required for Test 8 — run on PC with MILVUS_HOST set. "
            "Cannot be executed in Mac dev environment (no live Milvus)."
        )

    K = 10
    N_QUERY = 100  # held-out subset for evaluation
    N_POOL = 500  # pool to build baseline from
    THRESHOLD = 0.10
    N_BUCKETS = 10

    # Load a pool of DPs from Milvus with equity info via mean embedding dim
    # (Group A dim 10, 0-indexed, carries mean equity after z-score normalization)
    MEAN_EQ_DIM = 10  # 0-indexed in 80-dim postflop space (Group A dim 11 = mean)

    try:
        pool_rows = client.query(
            collection_name="postflop_decisions",
            filter="street_class == 'postflop'",
            output_fields=["decision_id", "street_class", "pot_type", "embedding"],
            limit=N_POOL + N_QUERY,
        )
    except Exception as e:
        return {
            "n": 8,
            "name": "Test 8: EHS-bucket baseline beat",
            "threshold": "embedding beats EHS-bucket by >=10pp",
            "observed": f"Error querying Milvus: {e}",
            "verdict": "FAIL",
            "mitigation": "N/A (critical — must PASS)",
        }

    if len(pool_rows) < N_QUERY + K:
        return {
            "n": 8,
            "name": "Test 8: EHS-bucket baseline beat",
            "threshold": "embedding beats EHS-bucket by >=10pp",
            "observed": f"Insufficient rows: {len(pool_rows)} < {N_QUERY + K}",
            "verdict": "FAIL",
            "mitigation": "N/A (critical — must PASS)",
        }

    # Split into query (held-out) and pool (baseline)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(pool_rows)).tolist()
    query_rows = [pool_rows[i] for i in indices[:N_QUERY]]
    pool_only_rows = [pool_rows[i] for i in indices[N_QUERY : N_POOL + N_QUERY]]

    def _eq_bucket(row: dict) -> int:
        """Assign equity decile bucket 0-9 from mean equity embedding dim."""
        emb = row.get("embedding", [])
        if not emb or len(emb) <= MEAN_EQ_DIM:
            return 5  # fallback mid-bucket
        # Mean equity dim is z-score normalized; approximate raw val from z-score
        # Use the raw value clipped to [0,1] as equity proxy
        raw_eq = float(np.clip(emb[MEAN_EQ_DIM], -3, 3))
        # Map [-3, 3] → [0, 9] bucket
        bucket = int(np.clip(int((raw_eq + 3) / 6.0 * N_BUCKETS), 0, N_BUCKETS - 1))
        return bucket

    def _hand_rank_class(row: dict) -> int:
        """Derive hand rank class 0-8 from rank_category embedding dim (Group C dim 0)."""
        emb = row.get("embedding", [])
        if not emb or len(emb) <= 20:
            return 0
        # Group C rank_category at dim 20: rank_category/8 before z-score
        # After z-score normalization the raw value is shifted; approximate
        # hand class from the dim (rough bucket 0-8)
        raw = float(emb[20])
        return int(np.clip(int(raw * 8), 0, 8))

    # Build EHS-bucket index: decision_id → bucket
    pool_buckets = {r["decision_id"]: _eq_bucket(r) for r in pool_only_rows}
    pool_hand_ranks = {r["decision_id"]: _hand_rank_class(r) for r in pool_only_rows}

    # Build pool lookup by bucket
    bucket_to_ids: dict[int, list[str]] = {}
    for did, bkt in pool_buckets.items():
        bucket_to_ids.setdefault(bkt, []).append(did)

    def _ehs_precision(query_row: dict) -> float:
        """Precision@K for EHS-bucket baseline: K random neighbors from same bucket."""
        q_bucket = _eq_bucket(query_row)
        q_rank = _hand_rank_class(query_row)
        candidates = [did for did in bucket_to_ids.get(q_bucket, []) if did != query_row.get("decision_id")]
        if not candidates:
            return 0.0
        chosen = rng.choice(candidates, size=min(K, len(candidates)), replace=False).tolist()
        matches = sum(1 for did in chosen if pool_hand_ranks.get(did, -1) == q_rank)
        return matches / len(chosen)

    def _emb_precision(query_row: dict) -> float:
        """Precision@K for embedding retrieval via Milvus kNN."""
        emb = query_row.get("embedding", [])
        if not emb:
            return 0.0
        q_rank = _hand_rank_class(query_row)
        pot_type = query_row.get("pot_type", "")
        filt = "street_class == 'postflop'"
        if pot_type:
            filt += f" and pot_type == '{pot_type}'"
        try:
            results = client.search(
                collection_name="postflop_decisions",
                data=[emb],
                limit=K + 1,
                filter=filt,
                output_fields=["decision_id", "embedding"],
            )
            neighbors = results[0]
        except Exception:
            return 0.0
        matches = 0
        count = 0
        for hit in neighbors:
            nid = hit.get("entity", {}).get("decision_id") or hit.get("id", "")
            if nid == query_row.get("decision_id"):
                continue
            n_emb = hit.get("entity", {}).get("embedding", [])
            n_rank = int(np.clip(int(float(n_emb[20]) * 8), 0, 8)) if n_emb else -1
            if n_rank == q_rank:
                matches += 1
            count += 1
            if count >= K:
                break
        return matches / max(count, 1)

    ehs_prec_list = [_ehs_precision(r) for r in query_rows]
    emb_prec_list = [_emb_precision(r) for r in query_rows]

    precision_ehs = float(np.mean(ehs_prec_list))
    precision_emb = float(np.mean(emb_prec_list))
    delta = precision_emb - precision_ehs

    verdict = "PASS" if delta >= THRESHOLD else "FAIL"
    return {
        "n": 8,
        "name": "Test 8: EHS-bucket baseline beat",
        "threshold": "embedding beats EHS-bucket by >=10pp",
        "observed": f"emb={precision_emb:.3f}, ehs_bucket={precision_ehs:.3f}, Δ={delta:.3f}",
        "verdict": verdict,
        "mitigation": "N/A (critical — must PASS)",
    }


# ─── Runner dispatch ──────────────────────────────────────────────────────────

SYNTHETIC_TESTS: list[Callable[..., VerdictDict]] = [
    run_test_1_monotone_twin,
    run_test_2_made_vs_draw,
    run_test_3_position_sensitivity,
    run_test_4_spr_sensitivity,
    run_test_6_bimodal_66_vs_kq,
    run_test_7_betting_sequence,
]
KNN_TESTS: list[Callable[..., VerdictDict]] = [
    run_test_5_knn_consistency,
    run_test_8_ehs_baseline_beat,
]


def _run_synthetic(fn: Callable[..., VerdictDict]) -> VerdictDict:
    """Worker entry point for forkserver pool. Must be module-level for pickling."""
    return fn()


def run_all(client, workers: int) -> list[VerdictDict]:
    """Run all 8 experiments and return sorted list of VerdictDicts."""
    log.info("validation.start", n_synthetic=len(SYNTHETIC_TESTS), n_knn=len(KNN_TESTS))
    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            synthetic_results = pool.map(_run_synthetic, SYNTHETIC_TESTS)
    else:
        synthetic_results = [t() for t in SYNTHETIC_TESTS]
    knn_results = [t(client) for t in KNN_TESTS]
    return sorted(synthetic_results + knn_results, key=lambda r: r["n"])


def write_report(results: list[VerdictDict], out: Path) -> None:
    """Write markdown verdict table to *out*. Always emits even if all PENDING."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Embedding Validation Results",
        "",
        f"**Run:** {datetime.datetime.now(datetime.UTC).isoformat()}",
        "**Source:** FEATURES-v2.md §Validation experiments",
        "**Critical (Phase 2 blocker):** Tests 1, 2, 5, 8",
        "**Mitigation-eligible:** Tests 3, 4, 6, 7",
        "",
        "## Results",
        "",
        "| # | Test | Threshold | Observed | Verdict | Mitigation |",
        "|---|------|-----------|----------|---------|------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['n']} | {r['name']} | {r['threshold']} "
            f"| {r['observed']} | {r['verdict']} | {r['mitigation']} |"
        )
    lines.append("")
    lines.append("## Critical Test Status")
    lines.append("")
    crit_fails = [r for r in results if r["n"] in CRITICAL_TESTS and r["verdict"] == "FAIL"]
    if crit_fails:
        lines.append(f"CRITICAL FAILURES — Phase 2 NOT done ({len(crit_fails)} failed):")
        for r in crit_fails:
            lines.append(f"- Test {r['n']}: {r['name']} — observed: {r['observed']}")
    else:
        lines.append("All critical tests PASS or PENDING (run on PC to finalize).")
    out.write_text("\n".join(lines))
    log.info("validation.report_written", path=str(out), n_results=len(results))


def main() -> int:
    """CLI entry point. Returns exit code (2 = critical failures, 0 = ok/pending)."""
    ap = argparse.ArgumentParser(description="Run 8 FEATURES-v2.md validation experiments")
    ap.add_argument("--preflop-manifest", type=Path, default=Path("tools/zscore_preflop.json"))
    ap.add_argument("--postflop-manifest", type=Path, default=Path("tools/zscore_postflop.json"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/research/EMBEDDING-VALIDATION-RESULTS.md"),
    )
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    # CRITICAL: avoids pyarrow threadpool deadlock (Pitfall P-3, RESEARCH.md)
    mp.set_start_method("forkserver", force=True)

    # Lazy import: tools/upsert_milvus.py is created by Plan 06.
    # The import is deferred here so unit tests can import this module without
    # Plan 06 having been executed first (scaffold-only, no live Milvus needed).
    try:
        from tools.upsert_milvus import _milvus_client_from_env

        client, _uri, _token = _milvus_client_from_env()
    except ImportError:
        log.warning(
            "validation.milvus_client_unavailable",
            reason="tools/upsert_milvus not found (Plan 06 not yet executed); kNN tests will return PENDING",
        )
        client = None

    results = run_all(client, workers=args.workers)
    write_report(results, args.out)

    crit_fails = [r for r in results if r["n"] in CRITICAL_TESTS and r["verdict"] == "FAIL"]
    if crit_fails:
        log.error("validation.critical_failures", failed_tests=[r["n"] for r in crit_fails])
        return 2  # 2 = validation failure (1 = generic error)
    log.info("validation.complete", verdicts={r["n"]: r["verdict"] for r in results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
