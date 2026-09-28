"""Turn one hand's replay rows into a flop-rooted harvest spec.

One hand's get_replay_by_hand rows → SolverSpot (flop-rooted) + one nav_line
per hero postflop DP + a parallel per-DP metadata list.
"""

from __future__ import annotations

from typing import Any

import msgspec

from src._log import get_logger
from src.solver.flop_spot import flop_start_stack_bb, is_multiway
from src.solver.nav_line import parse_nav_line
from src.solver.postflop_cli import SolverSpot
from src.solver.queue_driver import PALETTE_DIR, _derive_preflop_info
from src.solver.range_resolver import _resolve_ranges, build_range_lookup

__all__ = ["DpMeta", "HandHarvestSpec", "build_hand_harvest"]

log = get_logger("solver.hand_to_harvest")

_POSTFLOP_STREETS = {"flop", "turn", "river"}


class DpMeta(msgspec.Struct, frozen=True, kw_only=True):
    decision_id: str
    street: str
    hero_seat: int
    multiway: bool
    nav_ok_expected: bool


class HandHarvestSpec(msgspec.Struct, frozen=True, kw_only=True):
    spot: SolverSpot
    nav_lines: list[dict[str, object]]
    dp_meta: list[DpMeta]


def build_hand_harvest(
    hand_rows: list[dict[str, Any]],
    *,
    palette_lookup: dict[str, Any] | None = None,
    target_exploitability_pct: float = 1.0,
    max_iterations: int | None = None,
    memory_budget_mb: int | None = None,
) -> HandHarvestSpec:
    """Build a flop-rooted SolverSpot + parallel nav_lines/dp_meta for one hand."""
    palette_lookup = (
        palette_lookup
        if palette_lookup is not None
        else (build_range_lookup(PALETTE_DIR) if PALETTE_DIR.exists() else {})
    )

    flop_row = next(
        (r for r in hand_rows if (r.get("felt_snapshot") or {}).get("street") == "flop"),
        None,
    )
    if flop_row is None:
        raise ValueError("hand has no flop DP — cannot flop-root a harvest spec")

    rep_felt = flop_row["felt_snapshot"]
    hand_id: str = flop_row["hand_id"]
    hero_pos: str = rep_felt.get("hero_position") or rep_felt.get("hero_pos") or "BTN"

    flop_cards = list(rep_felt["board_cards"][:3])
    flop_stack_bb = flop_start_stack_bb(rep_felt, {hand_id: float(rep_felt["effective_stack_bb"])}, hand_id)
    flop_pot_bb = float(rep_felt.get("pot_size_bb") or 0.0)

    pot_type, villain_pos, opener_pos, bettor_pos, _ = _derive_preflop_info(
        rep_felt["action_sequence"], hero_pos
    )
    range_ip, range_oop = _resolve_ranges(
        rep_felt,
        hero_pos,
        villain_pos,
        pot_type,
        palette_lookup,
        opener_pos=opener_pos,
        bettor_pos=bettor_pos,
    )

    spot = SolverSpot(
        board=flop_cards,
        pot=round(flop_pot_bb * 100),
        effective_stack=round(flop_stack_bb * 100),
        range_ip=range_ip,
        range_oop=range_oop,
        target_exploitability_pct=target_exploitability_pct,
        max_iterations=max_iterations,
        memory_budget_mb=memory_budget_mb,
    )

    nav_lines: list[dict[str, object]] = []
    dp_meta: list[DpMeta] = []

    for row in hand_rows:
        felt = row.get("felt_snapshot") or {}
        street = felt.get("street") or ""
        if street not in _POSTFLOP_STREETS:
            continue

        m_hero_pos: str = felt.get("hero_position") or felt.get("hero_pos") or hero_pos
        _, m_villain_pos, _, _, _ = _derive_preflop_info(felt["action_sequence"], m_hero_pos)
        steps, hero_player, turn_card, river_card = parse_nav_line(
            felt["action_sequence"], m_hero_pos, m_villain_pos, list(felt["board_cards"])
        )
        multiway = is_multiway(felt)
        nav_ok_expected = not (multiway and street in ("turn", "river"))

        nav_lines.append(
            {
                "steps": steps,
                "hero_player": hero_player,
                "turn_card": turn_card,
                "river_card": river_card,
            }
        )
        dp_meta.append(
            DpMeta(
                decision_id=row["decision_id"],
                street=street,
                hero_seat=hero_player,
                multiway=multiway,
                nav_ok_expected=nav_ok_expected,
            )
        )

    return HandHarvestSpec(spot=spot, nav_lines=nav_lines, dp_meta=dp_meta)
