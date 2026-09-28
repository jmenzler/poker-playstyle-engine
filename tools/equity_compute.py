"""On-the-fly equity for query spots that miss the sparse corpus table (fail-soft to pop_mean)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from src._log import get_logger

log = get_logger("canonicalizer.equity_compute")

_DECILE_KEYS = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]

# Default villain continuing ranges per pot_type (postflop-solver Range syntax),
# approximating corpus palette ranges so query equity stays in-distribution.
_DEFAULT_RANGES: dict[str, str] = {
    "srp": "22+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,86s+,75s+,A8o+,KTo+,QTo+,JTo",
    "3bet": "55+,A5s,A7s+,K9s+,Q9s+,JTs,AJo+,KQo",
    "4bet": "TT+,AJs+,AKo,A5s",
    "5bet": "QQ+,AKs,AKo",
    "limp": "22+,A2s+,K6s+,Q8s+,J7s+,T7s+,96s+,85s+,75s+,64s+,A3o+,K8o+,Q9o+,J9o+,T8o+,98o",
}
_FALLBACK_RANGE = _DEFAULT_RANGES["srp"]

_TIMEOUT_S = 20.0


def _default_range_for_scenario(scenario: str) -> str:
    """Villain range for a scenario key, selected by its pot_type prefix."""
    pot_type = scenario.split("|", 1)[0]
    return _DEFAULT_RANGES.get(pot_type, _FALLBACK_RANGE)


class EquityComputer:
    """Compute equity deciles on demand via the equity-decile binary (per-instance cache)."""

    def __init__(self, binary_path: Path) -> None:
        self._bin = Path(binary_path)
        self._cache: dict[tuple[str, str, str], dict[str, float] | None] = {}

    def is_available(self) -> bool:
        return self._bin.exists() and self._bin.is_file()

    def compute(self, hole_canonical: str, board_canonical: str, scenario: str) -> dict[str, float] | None:
        key = (hole_canonical, board_canonical, scenario)
        if key not in self._cache:
            self._cache[key] = self._compute_uncached(hole_canonical, board_canonical, scenario)
        return self._cache[key]

    def _compute_uncached(self, hole: str, board: str, scenario: str) -> dict[str, float] | None:
        if not self.is_available():
            log.warning("equity_compute.binary_unavailable", path=str(self._bin))
            return None
        payload = json.dumps(
            {"id": "q", "hole": hole, "board": board, "villain_range": _default_range_for_scenario(scenario)}
        )
        try:
            proc = subprocess.run(
                [str(self._bin)],
                input=payload + "\n",
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except (subprocess.SubprocessError, OSError) as e:
            log.warning("equity_compute.subprocess_failed", error=str(e), hole=hole, board=board)
            return None
        lines = proc.stdout.strip().splitlines()
        if not lines:
            log.warning("equity_compute.empty_output", returncode=proc.returncode, stderr=proc.stderr[:200])
            return None
        try:
            resp = json.loads(lines[0])
        except json.JSONDecodeError:
            log.warning("equity_compute.bad_json", line=lines[0][:200])
            return None
        if resp.get("error") or resp.get("deciles") is None:
            log.warning("equity_compute.binary_error", error=resp.get("error"), hole=hole, board=board)
            return None
        deciles = [float(x) for x in resp["deciles"]]
        row: dict[str, float] = {k: deciles[i] for i, k in enumerate(_DECILE_KEYS)}
        row["mean"] = float(resp.get("mean", float(np.mean(deciles))))
        row["variance"] = float(np.var(deciles))
        return row
