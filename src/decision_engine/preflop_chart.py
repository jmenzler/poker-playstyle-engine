"""Chart-driven preflop strategy: lookup from inferred ranges, kNN is the fallback.

Replaces the diluting kNN preflop blend with a direct chart lookup keyed by the
hero's 169-combo + situation. Miss (multiway/missing chart/fold combo) -> None.
"""

from __future__ import annotations

import json
from pathlib import Path

from src._log import get_logger
from src.canonicalizer.encoder import _parse_action_tokens
from src.decision_engine.blending import CANONICAL_ACTIONS
from src.protocols.game_state import GameState
from tools.position import last_raiser_pos
from tools.preflop_combo import ALL_COMBOS_169, combo_169

log = get_logger("decision_engine.preflop_chart")

_CANONICAL: frozenset[str] = frozenset(CANONICAL_ACTIONS)
_VALID_COMBOS: frozenset[str] = frozenset(ALL_COMBOS_169)

# pot_type -> situation directory. limp/unopened = hero is unopened (RFI).
_POT_TYPE_TO_SITUATION: dict[str, str] = {
    "limp": "RFI",
    "unopened": "RFI",
    "srp": "vs_RFI",
    "3bet": "vs_3bet",
    "4bet": "vs_4bet",
}

# Situations that require a clean heads-up line; > 2 active players -> fall to kNN.
_HU_SITUATIONS: frozenset[str] = frozenset({"vs_RFI", "vs_3bet", "vs_4bet"})

_DIST_SUM_TOL = 1e-4


class PreflopChartStrategy:
    """Loads a chart tree and answers per-combo preflop action distributions."""

    def __init__(self, charts: dict[str, dict[str, dict[str, float]]]) -> None:
        """charts: {f"{situation}/{key}" -> {combo -> {action -> freq}}}."""
        self._charts = charts
        self.chart_hits = 0
        self.chart_misses = 0

    @property
    def n_charts(self) -> int:
        return len(self._charts)

    @classmethod
    def from_dir(cls, path: str | Path) -> PreflopChartStrategy:
        """Load + validate every {situation}/{key}.json under path. Fail loud."""
        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"preflop chart dir not found: {root}")
        charts: dict[str, dict[str, dict[str, float]]] = {}
        for sit_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for chart_file in sorted(sit_dir.glob("*.json")):
                key = f"{sit_dir.name}/{chart_file.stem}"
                chart = json.loads(chart_file.read_text())
                cls._validate_chart(key, chart)
                charts[key] = chart
        if not charts:
            raise ValueError(f"no charts found under {root}")
        log.info("preflop_chart.loaded", n_charts=len(charts), root=str(root))
        return cls(charts)

    @staticmethod
    def _validate_chart(key: str, chart: dict[str, dict[str, float]]) -> None:
        for combo, dist in chart.items():
            if combo not in _VALID_COMBOS:
                raise ValueError(f"{key}: invalid combo key {combo!r}")
            total = 0.0
            for action, freq in dist.items():
                if action not in _CANONICAL:
                    raise ValueError(f"{key}:{combo}: non-vocab action {action!r}")
                if freq < 0:
                    raise ValueError(f"{key}:{combo}: negative freq {action}={freq}")
                total += float(freq)
            if total > 1.0 + _DIST_SUM_TOL:
                raise ValueError(f"{key}:{combo}: dist sum {total} > 1")

    def lookup(self, gs: GameState, hard_filter: dict[str, object]) -> dict[str, float] | None:
        """Per-combo action dist for this spot, or None (miss -> kNN fallback).

        None when not preflop, unknown pot_type, multiway facing-spot, missing
        chart, or the combo is absent (pure fold). Dists exclude fold (remainder).
        """
        if str(hard_filter.get("street_class", "")).lower() != "preflop":
            self.chart_misses += 1
            return None

        situation = _POT_TYPE_TO_SITUATION.get(str(hard_filter.get("pot_type", "")))
        if situation is None:
            self.chart_misses += 1
            return None

        n_active_raw = hard_filter.get("n_players_active", 2)
        try:
            n_active = int(n_active_raw)  # type: ignore[call-overload]
        except (ValueError, TypeError):
            n_active = 2
        if situation in _HU_SITUATIONS and n_active > 2:
            self.chart_misses += 1
            return None

        chart_key = self._chart_key(situation, gs)
        if chart_key is None:
            self.chart_misses += 1
            return None
        chart = self._charts.get(chart_key)
        if chart is None:
            self.chart_misses += 1
            return None

        combo = combo_169(gs.hero_hole_cards[0], gs.hero_hole_cards[1])
        dist = chart.get(combo)
        if not dist:
            self.chart_misses += 1
            return None

        self.chart_hits += 1
        return dict(dist)

    @staticmethod
    def _chart_key(situation: str, gs: GameState) -> str | None:
        """Build the chart lookup key: hero seat (RFI) or '{hero}_vs_{opener}'."""
        hero = gs.hero_position
        if situation == "RFI":
            return f"{situation}/{hero}"
        opener = last_raiser_pos(_parse_action_tokens(gs.action_sequence))
        if opener is None or opener == hero:
            return None
        return f"{situation}/{hero}_vs_{opener}"
