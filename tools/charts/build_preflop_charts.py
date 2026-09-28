"""Build a swappable preflop chart tree from the inferred HH ranges.

Reads research/preflop-ranges/outputs/{POS}/*.json, emits per-combo action-freq
charts to charts/preflop/inferred_6max/ + manifest. Rate = breakdown/n_total_dealt.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from tools.preflop_combo import ALL_COMBOS_169

_REPO = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = _REPO / "research" / "preflop-ranges" / "outputs"
_OUT_DIR = _REPO / "charts" / "preflop" / "inferred_6max"

_POSITIONS = ("UTG", "MP", "CO", "BTN", "SB", "BB")
_BLIND_POS = frozenset({"SB", "BB"})

# situation -> (raise_file_pattern, call_file_pattern). {opener} substituted per villain.
# raise_file maps the "raise" stem to a sized action; call_file contributes "call" mass.
_SITUATIONS: dict[str, dict[str, str]] = {
    "RFI": {"raise_file": "open", "call_file": ""},
    "vs_RFI": {"raise_file": "3bet_vs_{opener}", "call_file": "defend_vs_{opener}"},
    "vs_3bet": {"raise_file": "4bet_vs_{opener}", "call_file": "defend_3bet_vs_{opener}"},
    "vs_4bet": {"raise_file": "5bet_vs_{opener}", "call_file": "defend_4bet_vs_{opener}"},
}


def _class_weight(combo: str) -> int:
    """Combos per 169-class: 6 (pair), 4 (suited), 12 (offsuit)."""
    if len(combo) == 2:
        return 6
    return 4 if combo.endswith("s") else 12


def _sized_raise_action(situation: str, hero_pos: str, hero_pos_rel: str) -> str:
    """Snap the raise stem to its sized canonical action (matches preflop_action_label)."""
    if situation == "RFI":
        return "open_3bb" if hero_pos in _BLIND_POS else "open_2_2bb"
    if situation == "vs_RFI":
        return "3bet_4x" if hero_pos_rel == "OOP" else "3bet_3x"
    if situation == "vs_3bet":
        return "4bet_2_5x"
    return "allin"  # vs_4bet -> 5bet jam


def _hero_pos_rel(hero: str, opener: str | None) -> str:
    """IP iff hero acts after opener postflop (SB<BB<UTG<MP<CO<BTN); seat fallback."""
    rank = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}
    if opener and opener != hero and opener in rank:
        return "IP" if rank[hero] > rank[opener] else "OOP"
    return "IP" if hero in {"CO", "BTN"} else "OOP"


def _combo_raise_rate(combo: str, info: dict[str, Any], inferred: dict[str, float] | None) -> float:
    """Per-combo raise rate, preferring inferred_rates for thin buckets."""
    if inferred is not None and combo in inferred:
        return max(0.0, min(1.0, float(inferred[combo])))
    dealt = int(info.get("n_total_dealt", 0))
    if dealt <= 0:
        return 0.0
    raised = int(info.get("breakdown", {}).get("raise", 0))
    return max(0.0, min(1.0, raised / dealt))


def _combo_call_rate(combo: str, info: dict[str, Any]) -> float:
    """Per-combo call rate from a defend file's breakdown."""
    dealt = int(info.get("n_total_dealt", 0))
    if dealt <= 0:
        return 0.0
    called = int(info.get("breakdown", {}).get("call", 0))
    return max(0.0, min(1.0, called / dealt))


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _build_one(situation: str, hero_pos: str, opener: str | None) -> dict[str, dict[str, float]] | None:
    """Assemble one chart {combo -> {action -> freq}} from component files."""
    pat = _SITUATIONS[situation]
    raise_name = pat["raise_file"].format(opener=opener)
    raise_data = _load(_SRC_DIR / hero_pos / f"{raise_name}.json")
    if raise_data is None:
        return None

    hero_pos_rel = _hero_pos_rel(hero_pos, opener)
    raise_action = _sized_raise_action(situation, hero_pos, hero_pos_rel)
    raise_pc = raise_data.get("per_combo", {})
    inferred = raise_data.get("inferred_rates")

    call_pc: dict[str, Any] = {}
    if pat["call_file"]:
        call_data = _load(_SRC_DIR / hero_pos / f"{pat['call_file'].format(opener=opener)}.json")
        if call_data is not None:
            call_pc = call_data.get("per_combo", {})

    chart: dict[str, dict[str, float]] = {}
    for combo in ALL_COMBOS_169:
        raise_f = _combo_raise_rate(combo, raise_pc.get(combo, {}), inferred)
        call_f = _combo_call_rate(combo, call_pc.get(combo, {})) if call_pc else 0.0
        # cap combined non-fold mass at 1.0; raise takes precedence over call.
        call_f = min(call_f, max(0.0, 1.0 - raise_f))
        dist: dict[str, float] = {}
        if raise_f > 0:
            dist[raise_action] = round(raise_f, 4)
        if call_f > 0:
            dist["call"] = round(call_f, 4)
        if dist:
            chart[combo] = dist
    return chart or None


def _coverage(chart: dict[str, dict[str, float]]) -> dict[str, float]:
    """Weighted non-fold mass / 1326 and combo count for a chart."""
    mass = sum(_class_weight(c) * sum(d.values()) for c, d in chart.items())
    return {"nonfold_mass": round(mass / 1326.0, 4), "n_combos": len(chart)}


def build(out_dir: Path = _OUT_DIR) -> dict[str, Any]:
    """Build the full chart tree + manifest. Returns the manifest dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, dict[str, Any]] = {}

    for situation in _SITUATIONS:
        sit_dir = out_dir / situation
        sit_dir.mkdir(parents=True, exist_ok=True)
        for hero_pos in _POSITIONS:
            if situation == "RFI":
                chart = _build_one(situation, hero_pos, None)
                if chart:
                    key = hero_pos
                    (sit_dir / f"{key}.json").write_text(json.dumps(chart, indent=0))
                    coverage[f"{situation}/{key}"] = _coverage(chart)
                continue
            for opener in _POSITIONS:
                if opener == hero_pos:
                    continue
                chart = _build_one(situation, hero_pos, opener)
                if chart:
                    key = f"{hero_pos}_vs_{opener}"
                    (sit_dir / f"{key}.json").write_text(json.dumps(chart, indent=0))
                    coverage[f"{situation}/{key}"] = _coverage(chart)

    manifest = {
        "source": "inferred-from-HH",
        "provenance": "research/preflop-ranges/outputs (per-combo breakdown / n_total_dealt; "
        "inferred_rates preferred for thin buckets)",
        "build_date": date.today().isoformat(),
        "situations": list(_SITUATIONS),
        "n_charts": len(coverage),
        "coverage": coverage,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_OUT_DIR, help="chart tree output dir")
    args = parser.parse_args()
    manifest = build(args.out)
    print(f"Built {manifest['n_charts']} charts under {args.out}")
    for sit in _SITUATIONS:
        rfi_keys = [k for k in manifest["coverage"] if k.startswith(f"{sit}/")]
        print(f"  {sit}: {len(rfi_keys)} charts")
    for pos in _POSITIONS:
        cov = manifest["coverage"].get(f"RFI/{pos}")
        if cov:
            print(f"    RFI/{pos}: open mass={cov['nonfold_mass']}")


if __name__ == "__main__":
    main()
