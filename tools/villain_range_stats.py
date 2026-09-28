"""Villain range tightness from observed preflop palettes (FEATURES-v2 §J).

The preflop embedding's Group J carries a villain-range "tightness" scalar that
the spec defines as "derived from solver palette at (pos, pot_type)". This module
computes it from the real combo_counts in research/preflop-ranges/outputs/ instead
of a hardcoded position proxy.

tightness = strength-weighted mean of the range, using Sklansky-Malmuth group
scores (premium=1.0 ... trash=0.0). A range concentrated in AA/KK/AK scores high
(tight); a wide limp/defend range full of marginal hands scores low.
"""

from __future__ import annotations

import json
from pathlib import Path

# Sklansky-Malmuth normalized strength per 169 class (mirrors preflop.py _SM_GROUPS,
# kept here to keep this module leaf — no import cycle with the extractor).
_SM_GROUPS: dict[str, float] = {
    "AA": 1.0,
    "KK": 1.0,
    "QQ": 1.0,
    "JJ": 1.0,
    "AKs": 1.0,
    "TT": 6 / 7,
    "AQs": 6 / 7,
    "AJs": 6 / 7,
    "KQs": 6 / 7,
    "AKo": 6 / 7,
    "99": 5 / 7,
    "JTs": 5 / 7,
    "QJs": 5 / 7,
    "KJs": 5 / 7,
    "ATs": 5 / 7,
    "AQo": 5 / 7,
    "88": 4 / 7,
    "QTs": 4 / 7,
    "98s": 4 / 7,
    "J9s": 4 / 7,
    "AJo": 4 / 7,
    "KQo": 4 / 7,
    "77": 3 / 7,
    "T9s": 3 / 7,
    "KTs": 3 / 7,
    "87s": 3 / 7,
    "Q9s": 3 / 7,
    "76s": 3 / 7,
    "97s": 3 / 7,
    "65s": 3 / 7,
    "KJo": 3 / 7,
    "QJo": 3 / 7,
    "66": 2 / 7,
    "55": 2 / 7,
    "86s": 2 / 7,
    "75s": 2 / 7,
    "K9s": 2 / 7,
    "T8s": 2 / 7,
    "64s": 2 / 7,
    "KTo": 2 / 7,
    "QTo": 2 / 7,
    "JTo": 2 / 7,
    "ATo": 2 / 7,
    "44": 1 / 7,
    "33": 1 / 7,
    "22": 1 / 7,
    "54s": 1 / 7,
}


def range_tightness(combo_counts: dict[str, float]) -> float:
    """Strength-weighted mean of a range's classes, in [0,1].

    Empty/None range → 0.5 (neutral). Unlisted classes score 0 (trash).
    """
    if not combo_counts:
        return 0.5
    total = 0.0
    acc = 0.0
    for cls, w in combo_counts.items():
        wf = float(w)
        if wf <= 0:
            continue
        acc += wf * _SM_GROUPS.get(cls, 0.0)
        total += wf
    return acc / total if total > 0 else 0.5


def build_tightness_lookup(palette_dir: Path) -> dict[str, float]:
    """Scan palette files → {f"{pos}/{action}": tightness}.

    Prefers the inferred GTO range when a file is low_sample (matches the
    equity builder's fallback policy), else the empirical combo_counts.
    """
    lookup: dict[str, float] = {}
    for path in palette_dir.glob("*/*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        empirical = data.get("combo_counts") or {}
        inferred = data.get("inferred_rates")
        low = bool(data.get("low_sample"))
        rng = inferred if (low or not empirical) and inferred else empirical
        if not rng:
            continue
        key = f"{path.parent.name}/{path.stem}"
        lookup[key] = range_tightness({k: float(v) for k, v in rng.items()})
    return lookup


def build_range_lookup(palette_dir: Path) -> dict[str, dict[str, float]]:
    """Scan palette files → {f"{pos}/{action}": combo_counts}.

    Returns the raw combo_counts (or inferred GTO prior for low-sample files)
    keyed by "{pos}/{action}" for blocker computation in the preflop extractor.
    """
    lookup: dict[str, dict[str, float]] = {}
    for path in palette_dir.glob("*/*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        empirical = data.get("combo_counts") or {}
        inferred = data.get("inferred_rates")
        low = bool(data.get("low_sample"))
        rng = inferred if (low or not empirical) and inferred else empirical
        if not rng:
            continue
        key = f"{path.parent.name}/{path.stem}"
        lookup[key] = {k: float(v) for k, v in rng.items()}
    return lookup
