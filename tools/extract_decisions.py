"""Decision Point extractor for hm_events.jsonl → hm_decisions.jsonl.

Walks per-hand event streams from hm-replayer output, emits one record per
hero decision point per DECISIONS-SCHEMA.md v1.

Schema-level decisions:
  - Skip whole hand if: tournament, RIT, straddle, HU table, 3+ players at flop
  - Skip preflop blind post events (not decisions)
  - Continuous action sizes (no bucketing here; defer to query-time tool)

Usage:
  python3 tools/extract_decisions.py \\
      --in research/preflop-ranges/outputs/hm_events.jsonl \\
      --out research/preflop-ranges/outputs/hm_decisions.jsonl

Run on either Mac or PC. ~5 min for 213k hands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src._log import configure_logging, get_logger
from tools.position import last_raiser_pos as _last_raiser_pos
from tools.position import preflop_pos_rel, preflop_pot_type

log = get_logger("tools.extract_decisions")

SCHEMA_VERSION = 2  # v1.1: multiway supported, villain pos/stacks explicit
SKIP_ACTIONS = {"post_sb", "post_bb", "post_ante", "post_straddle"}


def hand_excluded(hand: dict) -> tuple[bool, str | None]:
    """Return (excluded, reason). v1.1 exclusions per DECISIONS-SCHEMA.md."""
    if hand.get("table_size", 0) == 2:
        return True, "hu_table"
    if hand.get("table_size", 0) > 6:
        return True, "table_size_gt_6"
    pre = hand.get("preflop", []) or []
    if any(e.get("action") == "post_straddle" for e in pre):
        return True, "straddle"
    # Multiway now KEPT (was excluded in v1).
    return False, None


# Unused helper retained for reference; v1.1 computes hero_pos_rel inline.


def derive_facing(events_so_far: list[dict]) -> tuple[str, str | None, int | None]:
    """Look at events on this street so far, before hero's pending decision.
    Returns (facing_type, facing_pos, facing_size_cents).
    facing_type in {"check_to", "bet_to", "raise_to", "cold"} — "cold" means no prior street action."""
    if not events_so_far:
        return "cold", None, None
    last = events_so_far[-1]
    a = last.get("action")
    if a == "check":
        return "check_to", last.get("actor_pos"), None
    if a in ("bet",):
        return "bet_to", last.get("actor_pos"), last.get("amount_cents")
    if a in ("raise",):
        return "raise_to", last.get("actor_pos"), last.get("to_amount_cents")
    if a == "call":
        return "check_to", last.get("actor_pos"), None  # treat call-after-call as quasi-check-to
    return "cold", None, None


def is_decision_event(event: dict) -> bool:
    """Is this event hero making a non-trivial decision?"""
    if not event.get("is_hero"):
        return False
    return event.get("action") not in SKIP_ACTIONS


def n_board_cards_for_street(street: str) -> int:
    return {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]


def get_pot_type(hand: dict) -> str:
    return hand.get("pot_type", "unknown")


def stake_label(bb_cents: int) -> str:
    """Map bb_cents → 'NL5' / 'NL10' / 'NL25' / ..."""
    if bb_cents <= 0:
        return f"NL{bb_cents}c"
    return f"NL{bb_cents}"


def to_bb(cents: int | float | None, bb_cents: int) -> float | None:
    if cents is None or bb_cents <= 0:
        return None
    return round(cents / bb_cents, 3)


def _normalize_hole(hole: str | list[str] | None) -> list[str] | None:
    if hole is None:
        return None
    if isinstance(hole, str):
        return hole.split()
    return list(hole)


def extract_dps_from_hand(hand: dict) -> list[dict]:
    """Walk this hand's events, yield per-DP dict per schema v1.1.

    All monetary values stored as fractions of BB.
    Multiway supported: villains_active list + hero_position_order.
    """
    dps = []
    hero_alias = hand.get("hero_alias")
    seats = hand.get("seats") or {}  # pos -> player_name
    btn_seat = hand.get("btn_seat")
    bb_cents = hand.get("bb_cents", 0)
    sb_cents = hand.get("sb_cents", 0)
    site = hand.get("site")
    hand_id = hand.get("hand_id")

    # Hero pos
    hero_pos = next((pos for pos, name in seats.items() if name == hero_alias), None)
    if hero_pos is None:
        return []

    # Preflop summary (for postflop DPs)
    pre_events = hand.get("preflop", []) or []
    preflop_action_seq = []
    last_raiser_pos = None
    preflop_aggressor_was_hero = False
    for e in pre_events:
        if e.get("action") not in SKIP_ACTIONS:
            preflop_action_seq.append(
                {
                    "pos": e.get("actor_pos"),
                    "action": e.get("action"),
                    "size_cents": e.get("to_amount_cents") or e.get("amount_cents") or 0,
                }
            )
            if e.get("action") in ("bet", "raise"):
                last_raiser_pos = e.get("actor_pos")
                preflop_aggressor_was_hero = e.get("is_hero", False)

    preflop_aggressor_pos = last_raiser_pos
    pot_type = get_pot_type(hand)
    full_board = hand.get("board", []) or []
    stake = stake_label(bb_cents)

    preflop_action_seq_bb = [
        {"pos": e["pos"], "action": e["action"], "size_bb": to_bb(e["size_cents"], bb_cents)}
        for e in preflop_action_seq
    ]

    # ── Track per-player state continuously across streets ───────────────────
    # Walk EVERY event in the hand to maintain stacks + active state.
    # We compute these incrementally as we iterate; for each hero decision,
    # we snapshot the state at that moment.
    player_stack_bb: dict[str, float] = {}  # pos -> stack in BB
    player_active: dict[str, bool] = {}  # pos -> still in hand
    player_last_action_this_street: dict[str, str | None] = {}

    # Initialize stacks from preflop event stream (each event has stack_after for actor)
    # First pass: find each player's initial stack from any "post" or first action
    # Simpler: walk preflop, take stack_after + amount_cents at first appearance
    seen_initial: set[str] = set()
    for street_name in ["preflop", "flop", "turn", "river"]:
        for e in hand.get(street_name, []) or []:
            pos = e.get("actor_pos")
            if pos and pos not in seen_initial:
                # Reconstruct pre-action stack: stack_after + amount taken
                init = e.get("stack_after_cents", 0) + e.get("amount_cents", 0)
                player_stack_bb[pos] = to_bb(init, bb_cents) or 0.0
                player_active[pos] = True
                player_last_action_this_street[pos] = None
                seen_initial.add(pos)

    decision_idx = 0

    for street in ["preflop", "flop", "turn", "river"]:
        events = hand.get(street, []) or []
        if not events:
            continue
        # Reset last-action-this-street tracker
        for p in player_last_action_this_street:
            player_last_action_this_street[p] = None
        action_so_far_street = []

        for event in events:
            if is_decision_event(event):
                facing_type, facing_pos, facing_size_cents = derive_facing(action_so_far_street)
                pot_at_decision_cents = (
                    action_so_far_street[-1].get("pot_after_cents", 0)
                    if action_so_far_street
                    else (
                        pre_events[-1].get("pot_after_cents", 0)
                        if street != "preflop" and pre_events
                        else (sb_cents + bb_cents)
                    )
                )

                n_board = n_board_cards_for_street(street)
                board_at_decision = full_board[:n_board] if full_board else []

                # hero_pos_rel: IP iff hero acts last on this street.
                # Computed after we know hero_position_order + active count below.
                pos_rel = "?"  # placeholder; set after villains_active built

                pot_bb = to_bb(pot_at_decision_cents, bb_cents)
                eff_stack_cents = event.get("effective_stack_cents")
                if eff_stack_cents is None:
                    eff_stack_cents = event.get("stack_after_cents", 0) + event.get("amount_cents", 0)
                eff_stack_bb = to_bb(eff_stack_cents, bb_cents)
                hero_stack_pre_cents = event.get("stack_after_cents", 0) + event.get("amount_cents", 0)
                hero_stack_bb = to_bb(hero_stack_pre_cents, bb_cents)

                hero_action_amount_cents = event.get("amount_cents", 0)
                hero_action_to_cents = event.get("to_amount_cents", 0)

                # Active villains (everyone except hero, still in hand)
                villains_active = [
                    {
                        "pos": p,
                        "stack_bb": round(player_stack_bb.get(p, 0.0), 3),
                        "last_action_this_street": player_last_action_this_street.get(p),
                    }
                    for p, active in player_active.items()
                    if active and p != hero_pos
                ]
                # Sort by position canonical order (UTG, MP, CO, BTN, SB, BB)
                POS_ORDER = {"UTG": 0, "MP": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}
                villains_active.sort(key=lambda v: POS_ORDER.get(v["pos"], 99))

                # Single villain shortcut (HU case)
                villain_pos = villains_active[0]["pos"] if len(villains_active) == 1 else None

                # Hero's order to act this street (1-indexed; counts unique actors before hero)
                actors_this_street_so_far = []
                for e in action_so_far_street:
                    p = e.get("actor_pos")
                    if p not in actors_this_street_so_far:
                        actors_this_street_so_far.append(p)
                hero_position_order = len(actors_this_street_so_far) + 1

                n_active = sum(1 for a in player_active.values() if a)
                # Skip walks (hero is sole remaining player — auto-win).
                # We use a sentinel flag instead of continue to ensure state update runs below.
                skip_emit = n_active <= 1

                # hero_pos_rel = postflop-relative IP/OOP (button advantage)
                if street == "preflop":
                    pos_rel = preflop_pos_rel(hero_pos, _last_raiser_pos(action_so_far_street))
                else:
                    POSTFLOP_ORDER = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
                    active_positions = [p for p, a in player_active.items() if a]
                    active_in_order = [p for p in POSTFLOP_ORDER if p in active_positions]
                    last_to_act = active_in_order[-1] if active_in_order else hero_pos
                    pos_rel = "IP" if hero_pos == last_to_act else "OOP"

                dp = {
                    "_schema_version": SCHEMA_VERSION,
                    "id": f"{hand_id}_dp{decision_idx}",
                    "hand_id": str(hand_id),
                    "decision_idx": decision_idx,
                    "site": site,
                    "stake": stake,
                    "bb_cents": bb_cents,
                    "table_size": hand.get("table_size"),
                    "btn_seat": btn_seat,
                    "street": street,
                    "hero_pos": hero_pos,
                    "hero_pos_rel": pos_rel,
                    "hero_position_order": hero_position_order,
                    "n_players_at_street": n_active,
                    "pot_type": preflop_pot_type(action_so_far_street) if street == "preflop" else pot_type,
                    "preflop_aggressor": preflop_aggressor_pos,
                    "preflop_action_seq": preflop_action_seq_bb
                    if street != "preflop"
                    else preflop_action_seq_bb[: len(action_so_far_street)],
                    "preflop_aggressor_was_hero": preflop_aggressor_was_hero,
                    "villain_pos": villain_pos,
                    "villains_active": villains_active,
                    "pot_bb": pot_bb,
                    "effective_stack_bb": eff_stack_bb,
                    "hero_stack_bb": hero_stack_bb,
                    "spr": event.get("spr_at_decision"),
                    "board": board_at_decision,
                    "board_canonical": "".join(board_at_decision),
                    "hero_hole": _normalize_hole(hand.get("hero_hole")),
                    "action_so_far_street": [
                        {
                            "pos": e["actor_pos"],
                            "action": e["action"],
                            "size_bb": to_bb(e.get("amount_cents", 0), bb_cents),
                        }
                        for e in action_so_far_street
                    ],
                    "facing": facing_type,
                    "facing_pos": facing_pos,
                    "facing_size_bb": to_bb(facing_size_cents, bb_cents),
                    "facing_size_pot_frac": (facing_size_cents / pot_at_decision_cents)
                    if (facing_size_cents and pot_at_decision_cents > 0)
                    else None,
                    "hero_action_type": event.get("action"),
                    "hero_action_size_bb": to_bb(hero_action_amount_cents, bb_cents),
                    "hero_action_to_bb": to_bb(hero_action_to_cents, bb_cents),
                    "hero_action_size_pot_frac": (
                        hero_action_amount_cents / pot_at_decision_cents
                        if pot_at_decision_cents > 0
                        else None
                    ),
                    "hero_action_allin": event.get("allin", False),
                }
                if not skip_emit:
                    dps.append(dp)
                    decision_idx += 1

            # Update tracked state AFTER processing event (whether DP or not)
            event_pos = event.get("actor_pos")
            action_name = event.get("action")
            if event_pos:
                # Update stack
                stack_after = event.get("stack_after_cents", 0)
                player_stack_bb[event_pos] = to_bb(stack_after, bb_cents) or 0.0
                # Update active state
                if action_name == "fold":
                    player_active[event_pos] = False
                # Track last action
                if action_name not in SKIP_ACTIONS:
                    player_last_action_this_street[event_pos] = action_name

            action_so_far_street.append(event)
    return dps


def _count_active(pre_events: list, street_events: list, street_so_far: list) -> int:
    """Estimate active players at this decision point."""
    folded = set()
    for e in pre_events + street_events[: len(street_so_far)]:
        if e.get("action") == "fold":
            folded.add(e.get("actor_pos"))
    # crude: 6 seats - folded
    return max(2, 6 - len(folded))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    # Accept both --in (legacy) and --events (new alias) for the input path
    p.add_argument(
        "--in", "--events", dest="in_path", default="research/preflop-ranges/outputs/hm_events.jsonl"
    )
    p.add_argument("--out", dest="out_path", default="research/preflop-ranges/outputs/hm_decisions.jsonl")
    p.add_argument("--limit", type=int, default=0, help="limit hand count for testing")
    args = p.parse_args(argv)

    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        log.error("extract_decisions.input_not_found", path=str(in_path))
        return 1

    n_hands = 0
    n_dps = 0
    exclusions: dict[str, int] = {}

    # BOOT-06 skip counters — tracked separately for structured end-of-run summary
    n_straddle = 0
    n_rit = 0  # RIT detection is a known gap (HH-INGEST.md §Run-It-Twice); always 0

    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            n_hands += 1
            if args.limit and n_hands > args.limit:
                break
            hand = json.loads(line)
            excluded, reason = hand_excluded(hand)
            if excluded:
                exclusions[reason] = exclusions.get(reason, 0) + 1
                # Route to BOOT-06 counters
                if reason == "straddle":
                    n_straddle += 1
                elif reason == "rit":
                    n_rit += 1
                continue
            dps = extract_dps_from_hand(hand)
            for dp in dps:
                f_out.write(json.dumps(dp) + "\n")
                n_dps += 1
            if n_hands % 50000 == 0:
                log.info(
                    "extract_decisions.progress",
                    n_hands=n_hands,
                    n_dps=n_dps,
                )

    # BOOT-06: emit unconditional end-of-run INFO summary with skip counts
    # (emitted even when both counts are 0 so verifiers can grep deterministically)
    log.info(
        "extract_decisions.skip_summary",
        straddle_skip=int(n_straddle),
        rit_skip=int(n_rit),
        n_hands=n_hands,
        n_dps=n_dps,
        exclusions=exclusions,
    )

    log.info(
        "extract_decisions.complete",
        n_hands=n_hands,
        n_dps=n_dps,
        wrote=str(out_path),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
