"""Enumerate unique (hole_canonical, board_canonical, street, scenario_id) tuples
from hm_decisions.jsonl for equity decile table build.

scenario_id derives from (hero_pos_rel, villain_pos, pot_type, n_players_at_street)
and maps to a preflop-range palette scenario (research/preflop-ranges/outputs/).

Run:
    uv run python3 tools/enumerate_equity_tuples.py <hm_decisions.jsonl> [<out.json>]

Output: JSON dump with tuple stats per street + full deduped list for downstream
solver dispatch.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from joint_canonicalize import joint_canonicalize


def _classify_villain_role(
    pos: str,
    preflop_seq: list[dict],
    preflop_aggressor: str | None,
) -> tuple[str, str | None]:
    """Classify a villain's preflop role + target.

    Returns (role, target) where role ∈ {pfr, caller, cold_caller, cold_4bettor,
    limper, unknown} and target is the position they last reacted to (or None
    for openers/limpers).

    Heuristic:
    - Walk preflop_seq, track each villain's actions in order.
    - role = derived from last meaningful action (raise vs call vs check).
    - target = position of the most recent raiser they responded to, or the
      raiser they 3-bet/4-bet.
    """
    actions = [a for a in preflop_seq if a["pos"] == pos]
    if not actions:
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
                elif role == "unknown" or role == "limper":
                    role = "caller" if len(raises_before_villain) == 1 else "cold_caller"
                    target = raises_before_villain[-1]
                else:
                    target = raises_before_villain[-1]
            elif act == "check" and not raises_before_villain:
                role = "checker"
        elif act == "raise":
            raises_before_villain.append(p)

    return (role, target)


def scenario_key(dp: dict) -> str:
    """Map a DP to a palette scenario key with per-villain role encoding.

    Format: pot_type|hero_rel|nN|<villain_spec>[+<villain_spec>...]
    Each villain_spec: POS:role[:target]  e.g. CO:pfr  or  BB:caller:BTN
    """
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
        role, target = _classify_villain_role(pos, seq, pfr)
        if target:
            specs.append(f"{pos}:{role}:{target}")
        else:
            specs.append(f"{pos}:{role}")
    specs.sort()
    return f"{pot_type}|{hero_rel}|n{n}|" + "+".join(specs)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("equity_tuples.json")

    per_street_tuples: dict[str, set[tuple[str, str, str]]] = {
        "flop": set(),
        "turn": set(),
        "river": set(),
    }
    scenarios: Counter[str] = Counter()
    bad = 0
    skipped_preflop = 0

    with in_path.open() as f:
        for line_no, line in enumerate(f, 1):
            try:
                dp = json.loads(line)
                street = dp["street"]
                if street == "preflop":
                    skipped_preflop += 1
                    continue
                hole = dp["hero_hole"]
                board = dp["board"]
                hc, bc = joint_canonicalize(hole, board)
                scen = scenario_key(dp)
                per_street_tuples[street].add((hc, bc, scen))
                scenarios[scen] += 1
            except (KeyError, ValueError) as e:
                bad += 1
                if bad <= 5:
                    print(f"  skip line {line_no}: {e}", file=sys.stderr)

    stats = {street: len(s) for street, s in per_street_tuples.items()}
    print("\nUnique (hole_canon, board_canon, scenario) tuples per street:")
    for street, n in stats.items():
        print(f"  {street:>7}: {n:>8,}")
    print(f"\n  total: {sum(stats.values()):,}")
    print(f"  preflop DPs skipped (no postflop board): {skipped_preflop:,}")
    print(f"  bad lines skipped: {bad}")
    print("\nTop 20 scenarios by DP count:")
    for scen, count in scenarios.most_common(20):
        print(f"  {count:>8,}  {scen}")
    print(f"\n  unique scenarios: {len(scenarios)}")

    out = {
        "stats": stats,
        "total_unique_tuples": sum(stats.values()),
        "n_scenarios": len(scenarios),
        "tuples": {street: sorted([list(t) for t in s]) for street, s in per_street_tuples.items()},
    }
    out_path.write_text(json.dumps(out))
    print(f"\nWrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
