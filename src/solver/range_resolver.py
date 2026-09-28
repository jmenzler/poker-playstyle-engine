"""Preflop range resolution from HM3 palette files.  long-ok

Converts palette combo_counts to postflop-cli notation strings and resolves
IP/OOP ranges for a solver spot using the fallback chain:
  1. obs_spot_features['range_ip'/'range_oop'] (stored from prior analysis)
  2. Palette lookup via ip_key/oop_key
  3. Placeholder "AA-22,AKs-A2s" with a WARNING log so the canary detects misses
"""

from __future__ import annotations

from typing import Any

from src._log import get_logger
from tools.villain_range_stats import build_range_lookup

__all__ = ["_combos_to_postflop_cli_notation", "_resolve_ranges", "build_range_lookup"]

log = get_logger("solver.range_resolver")

_PLACEHOLDER = "AA-22,AKs-A2s"

_POSTFLOP_RANK: dict[str, int] = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}


def _combos_to_postflop_cli_notation(combo_counts: dict[str, float]) -> str:
    """Convert a {hand_class: freq_float} dict to postflop-cli range notation.  long-ok

    freq == 1.0 (within 0.001) → bare hand (e.g. "AA")
    freq < 1.0 → "HAND:{freq:.2f}" (e.g. "KK:0.90")
    freq <= 0 → excluded
    Result: comma-separated, no spaces.
    """
    parts: list[str] = []
    for hand, freq in combo_counts.items():
        if freq <= 0:
            continue
        if abs(freq - 1.0) < 0.001:
            parts.append(hand)
        else:
            parts.append(f"{hand}:{freq:.2f}")
    return ",".join(parts)


def _normalize_counts(raw: dict[str, float]) -> dict[str, float]:
    """Normalize raw combo counts to [0, 1] by dividing by max count."""
    if not raw:
        return {}
    max_count = max(raw.values())
    if max_count <= 0:
        return {}
    return {k: v / max_count for k, v in raw.items()}


def _postflop_ip_player(pos_a: str, pos_b: str) -> str:
    """Return the IP player (higher postflop rank = acts last = IP)."""
    rank_a = _POSTFLOP_RANK.get(pos_a, -1)
    rank_b = _POSTFLOP_RANK.get(pos_b, -1)
    return pos_a if rank_a >= rank_b else pos_b


def _palette_keys_for(
    pot_type: str,
    hero_pos: str,
    villain_pos: str,
    opener: str,
    bettor: str,
) -> dict[str, str] | None:
    """Map each player (hero, villain) to its palette range key per pot_type.

    Action-specific names mirror sample_palette.enumerate_scenarios; returns
    {pos: key} or None when positions can't be mapped.
    """

    def _other(p: str) -> str:
        return villain_pos if p == hero_pos else hero_pos

    players = (hero_pos, villain_pos)
    if pot_type == "limp":
        return {hero_pos: f"{hero_pos}/limp", villain_pos: f"{villain_pos}/limp"}
    if pot_type == "srp":
        if opener not in players:
            return None
        caller = _other(opener)
        return {opener: f"{opener}/open", caller: f"{caller}/defend_vs_{opener}"}
    if pot_type == "3bet":
        if opener not in players or bettor not in players or opener == bettor:
            return None
        return {bettor: f"{bettor}/3bet_vs_{opener}", opener: f"{opener}/defend_3bet_vs_{bettor}"}
    if pot_type == "4bet":
        if opener not in players:
            return None
        tb = _other(opener)
        return {opener: f"{opener}/4bet_vs_{tb}", tb: f"{tb}/defend_4bet_vs_{opener}"}
    return None


def _resolve_ranges(
    obs_spot_features: dict[str, Any],
    hero_pos: str,
    villain_pos: str,
    pot_type: str,
    palette_lookup: dict[str, dict[str, float]],
    *,
    opener_pos: str | None = None,
    bettor_pos: str | None = None,
) -> tuple[str, str]:
    """Resolve IP/OOP range strings for a solver spot.  long-ok

    Fallback chain:
      1. obs_spot_features keys 'range_ip'/'range_oop' if both present
      2. Palette lookup — range_ip is the IP player's range, range_oop the OOP player's
         (IP = closer to button by postflop action order, regardless of which is hero).
         Per-pot_type keys via _palette_keys_for (srp→open/defend_vs, 3bet→3bet_vs/
         defend_3bet_vs, 4bet→4bet_vs/defend_4bet_vs, limp→limp).
      3. Placeholder with WARNING log on every miss
    """
    stored_ip = obs_spot_features.get("range_ip")
    stored_oop = obs_spot_features.get("range_oop")
    if stored_ip and stored_oop:
        return str(stored_ip), str(stored_oop)

    eff_opener = opener_pos or villain_pos
    eff_bettor = bettor_pos or hero_pos

    keys = _palette_keys_for(pot_type, hero_pos, villain_pos, eff_opener, eff_bettor)
    if keys:
        raws = {pos: r for pos, k in keys.items() if (r := palette_lookup.get(k)) is not None}
        if len(raws) == len(keys):
            ranges = {
                pos: _combos_to_postflop_cli_notation(_normalize_counts(raw)) for pos, raw in raws.items()
            }
            ip_player = _postflop_ip_player(hero_pos, villain_pos)
            oop_player = villain_pos if ip_player == hero_pos else hero_pos
            return ranges[ip_player], ranges[oop_player]

    log.warning(
        "solver.range_resolver.placeholder_fallback",
        hero_pos=hero_pos,
        villain_pos=villain_pos,
        pot_type=pot_type,
        keys=keys,
    )
    return _PLACEHOLDER, _PLACEHOLDER
