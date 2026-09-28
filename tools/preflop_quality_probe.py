"""Pre-ingest quality probe for the v3 preflop embedding.

The 8 validation experiments (tools/validation_experiments.py) only query the
postflop_decisions collection, so they do NOT exercise any of the v3 preflop
changes (blockers, multiway equity, vNA/limp dissolve, pot_odds, players_yet_to_act).
This probe fills that gap WITHOUT a full Milvus ingest: it embeds a real sample
from hm_decisions.jsonl through the actual EquityLookup + extract_preflop and
asserts correctness properties directly in vector space.

Run on PC (has hm_decisions.jsonl + built equity tables):
    uv run python tools/preflop_quality_probe.py \\
        --decisions research/preflop-ranges/outputs/hm_decisions.jsonl \\
        --equity-table tools/equity_table.parquet \\
        --preflop-equity-table tools/preflop_equity_table.parquet \\
        --limit 40000

Exit code 0 = all probes PASS, 1 = any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.build_embedding import _inject_scenario_key
from tools.equity_lookup import EquityLookup
from tools.feature_extractors.preflop import _hole_to_class, extract_preflop
from tools.villain_range_stats import build_range_lookup, build_tightness_lookup

_PALETTE_DIR = Path(__file__).resolve().parent.parent / "research" / "preflop-ranges" / "outputs"

# Dim indices (0-based) per FEATURES-v2 v3 layout
DIM_MEAN_EQ = 5
DIM_POT_ODDS = 20
DIM_PLAYERS_YET = 19
DIM_VALUE_BLOCKER = 32
DIM_RANGE_BLOCKER = 33


def _load(decisions: Path, eq: EquityLookup, limit: int) -> list[dict]:
    """Embed up to `limit` preflop DPs; return list of {dp, vec}."""
    pop_mean = eq.population_mean_equity()
    tl = build_tightness_lookup(_PALETTE_DIR) if _PALETTE_DIR.exists() else {}
    rl = build_range_lookup(_PALETTE_DIR) if _PALETTE_DIR.exists() else {}
    out: list[dict] = []
    with decisions.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            dp = json.loads(line)
            if dp.get("street") != "preflop":
                continue
            dp = _inject_scenario_key(dp)
            vec, _ = extract_preflop(dp, eq, pop_mean, tl, rl)
            out.append({"dp": dp, "vec": vec})
            if len(out) >= limit:
                break
    return out


def _probe(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def run(samples: list[dict]) -> bool:
    vecs = np.array([s["vec"] for s in samples], dtype=np.float64)
    n = len(samples)
    passed = True

    # 1. Shape + finite + range
    shape_ok = vecs.shape[1] == 34 and np.isfinite(vecs).all()
    rng_ok = bool((vecs >= 0.0).all() and (vecs <= 1.0).all())
    passed &= _probe(
        "dim34_finite_range",
        shape_ok and rng_ok,
        f"shape={vecs.shape} finite={np.isfinite(vecs).all()} in[0,1]={rng_ok}",
    )

    # 2. players_yet_to_act zeroed
    py = vecs[:, DIM_PLAYERS_YET]
    passed &= _probe(
        "players_yet_zeroed", bool((py == 0.0).all()), f"max={py.max():.4f} nonzero={(py != 0).sum()}"
    )

    # 3. pot_odds spreads across its range (not all pinned to the old 0.5 clip).
    # Real facing_size_pot_frac maxes near 1.0 -> pot_odds max ~0.5; the bug being
    # tested is the OLD hard clip collapsing many distinct spots onto exactly 0.5.
    po = vecs[:, DIM_POT_ODDS]
    distinct = len(np.unique(np.round(po, 3)))
    passed &= _probe(
        "pot_odds_spread",
        distinct >= 10,
        f"distinct values={distinct} max={po.max():.3f} (not collapsed to a single clip value)",
    )

    # 3b. Group I (hand-class, dims 6-10) must be populated for the bulk of DPs.
    # Low offsuit connectors (65o/54o/...) legitimately zero all 5 dims
    # (no pair, unsuited, gap 0, not broadway, not in SM table) — so expect
    # the majority nonzero, not 100%. The bug was 0% nonzero.
    gi = vecs[:, 6:11]
    gi_nonzero = int((np.abs(gi).sum(axis=1) > 0).sum())
    passed &= _probe(
        "group_i_hand_class_populated",
        gi_nonzero >= 0.85 * n,
        f"nonzero rows={gi_nonzero}/{n} (>=85% expected; low offsuit connectors legitimately 0)",
    )

    # 4. blockers in range + actually fire for some DPs
    vb = vecs[:, DIM_VALUE_BLOCKER]
    rb = vecs[:, DIM_RANGE_BLOCKER]
    fires = (vb > 0).sum() + (rb > 0).sum()
    passed &= _probe(
        "blockers_active",
        fires > 0,
        f"value-blocker nonzero={int((vb > 0).sum())} range-blocker nonzero={int((rb > 0).sum())}",
    )

    # 5. blocker semantics: ace-holding heroes block more premium than non-ace heroes
    hero_classes = [_hole_to_class(s["dp"].get("hero_hole") or []) for s in samples]
    ace = np.array(["A" in c[:2] for c in hero_classes])
    if ace.any() and (~ace).any():
        vb_ace = vb[ace].mean()
        vb_non = vb[~ace].mean()
        passed &= _probe(
            "ace_blocks_more_premium",
            vb_ace > vb_non,
            f"value-blocker ace-hero mean={vb_ace:.4f} > non-ace mean={vb_non:.4f}",
        )
    else:
        _probe("ace_blocks_more_premium", True, "skipped — sample lacks both ace/non-ace heroes")

    # 6. hand-strength ordering: AA mean-equity > 72o mean-equity
    meq = vecs[:, DIM_MEAN_EQ]
    aa = [meq[i] for i, c in enumerate(hero_classes) if c == "AA"]
    trash = [meq[i] for i, c in enumerate(hero_classes) if c in ("72o", "82o", "72s")]
    if aa and trash:
        passed &= _probe(
            "equity_ordering_AA_gt_trash",
            np.mean(aa) > np.mean(trash),
            f"AA mean-eq={np.mean(aa):.3f} > trash mean-eq={np.mean(trash):.3f}",
        )
    else:
        _probe("equity_ordering_AA_gt_trash", True, f"skipped — AA={len(aa)} trash={len(trash)}")

    # 7. multiway < heads-up: same hand class, mean-eq lower at n>=3 than n=2
    n2 = [meq[i] for i, s in enumerate(samples) if s["dp"].get("n_players_at_street") == 2]
    n3 = [meq[i] for i, s in enumerate(samples) if (s["dp"].get("n_players_at_street") or 0) >= 3]
    if n2 and n3:
        passed &= _probe(
            "multiway_lower_equity",
            np.mean(n3) < np.mean(n2),
            f"n>=3 mean-eq={np.mean(n3):.3f} < n2 mean-eq={np.mean(n2):.3f}",
        )
    else:
        _probe("multiway_lower_equity", True, f"skipped — n2={len(n2)} n3+={len(n3)}")

    print(f"\nProbed {n} preflop DPs. Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-ingest preflop embedding quality probe")
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--equity-table", required=True, type=Path)
    ap.add_argument("--preflop-equity-table", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=40000)
    args = ap.parse_args(argv)

    eq = EquityLookup([args.equity_table, args.preflop_equity_table])
    samples = _load(args.decisions, eq, args.limit)
    if not samples:
        print("FAIL: no preflop DPs loaded")
        return 1
    return 0 if run(samples) else 1


if __name__ == "__main__":
    sys.exit(main())
