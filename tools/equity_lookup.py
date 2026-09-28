"""Equity decile table loader + (hole_canonical, board_canonical, scenario) → 9 deciles + mean.

The equity table is produced by tools/build_equity_table.py and indexed by a
composite key "{hole_canonical}|{board_canonical}|{scenario}".  This module
loads the full Parquet once into RAM (54k rows x 16 cols ~7 MB) and exposes
fast O(1) dict lookups.

Fallback chain on scenario miss:
    1. Exact key: "{hole}|{board}|{scenario}"
    2. Pot-type family: drop villain detail, keep pot_type/hero_rel/nN prefix
       e.g. "srp|IP|n2|BTN:pfr" → try any scenario that starts with "srp|IP|n2|"
       (pick the lexicographically smallest full key; deterministic across merges)
    3. None — caller logs warning, uses population mean.

Usage:
    eq = EquityLookup("tools/equity_table.parquet")
    result = eq.get("AhKh", "AcJd5d", "srp|IP|n2|BTN:pfr")
    # result: {"p10": ..., "p20": ..., ..., "p90": ..., "mean": ..., "variance": ...}
    # or None on double miss

    # Multi-table usage (postflop + preflop merged):
    eq = EquityLookup(["tools/equity_table.parquet", "tools/preflop_equity_table.parquet"])
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# Double-miss handler (hole, board, scenario) → equity row or None. Serve path computes
# equity on-the-fly; build path leaves it None (miss → ledger).
OnMiss = Callable[[str, str, str], "dict[str, float] | None"]

# Decile column names in table order
_DECILE_COLS = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]


class EquityLookup:
    """In-memory equity table for fast O(1) lookup.

    Accepts a single path or a list of paths. When a list is provided, all
    tables are loaded and merged into a single in-memory index. Missing files
    in a list are skipped with a warning; at least one file must exist.
    """

    def __init__(self, parquet_path: str | Path | list[str | Path], *, on_miss: OnMiss | None = None) -> None:
        self._table: dict[str, dict[str, float]] = {}
        # Maps partial key prefix "hole|board|pot_type|hero_rel|nN|" → list of full keys
        self._prefix_index: dict[str, list[str]] = {}
        self._on_miss = on_miss
        self._n_total = 0
        self._miss_count = 0
        self._hit_count = 0

        if isinstance(parquet_path, list):
            paths = [Path(p) for p in parquet_path]
            if not paths:
                raise ValueError("EquityLookup: parquet_path list must not be empty")
            n_loaded = 0
            for p in paths:
                if not p.exists():
                    log.warning("EquityLookup: skipping missing file %s", p)
                    continue
                self._load_one(p)
                n_loaded += 1
            if n_loaded == 0:
                raise FileNotFoundError(f"EquityLookup: no valid parquet files found in list {parquet_path}")
        else:
            self._load_one(Path(parquet_path))

    def _load_one(self, path: Path) -> None:
        """Load a single Parquet file and merge into self._table / self._prefix_index."""
        t = pq.read_table(path).to_pydict()
        n = len(t["hole_canonical"])
        self._n_total += n
        for i in range(n):
            hole = t["hole_canonical"][i]
            board = t["board_canonical"][i]
            scen = t["scenario"][i]
            deciles = [float(t[col][i]) for col in _DECILE_COLS]
            mean = float(t["mean_equity"][i])
            # Approximate variance from decile values
            variance = float(np.var(deciles))
            row = {
                "p10": deciles[0],
                "p20": deciles[1],
                "p30": deciles[2],
                "p40": deciles[3],
                "p50": deciles[4],
                "p60": deciles[5],
                "p70": deciles[6],
                "p80": deciles[7],
                "p90": deciles[8],
                "mean": mean,
                "variance": variance,
            }
            key = f"{hole}|{board}|{scen}"
            if key in self._table:
                raise ValueError(f"EquityLookup: duplicate exact key {key!r} across merged files {path}")
            self._table[key] = row

            # Build prefix index: hole|board|pot_type|hero_rel|nN (first 3 scenario parts)
            scen_parts = scen.split("|")
            if len(scen_parts) >= 3:
                prefix = f"{hole}|{board}|{'|'.join(scen_parts[:3])}|"
                if prefix not in self._prefix_index:
                    self._prefix_index[prefix] = []
                self._prefix_index[prefix].append(key)

        log.info("EquityLookup loaded %d rows from %s", n, path)

    def get(
        self,
        hole_canonical: str,
        board_canonical: str,
        scenario: str,
    ) -> dict[str, float] | None:
        """Return equity dict or None on double miss.

        On miss, logs a warning and tries the fallback chain.
        """
        exact_key = f"{hole_canonical}|{board_canonical}|{scenario}"

        # 1. Exact hit
        if exact_key in self._table:
            self._hit_count += 1
            return self._table[exact_key]

        # 2. Scenario-family fallback: same pot_type/hero_rel/nN prefix
        scen_parts = scenario.split("|")
        if len(scen_parts) >= 3:
            prefix = f"{hole_canonical}|{board_canonical}|{'|'.join(scen_parts[:3])}|"
            candidates = self._prefix_index.get(prefix)
            if candidates:
                self._miss_count += 1
                # Deterministic regardless of file/row load order across merged tables.
                chosen = min(candidates)
                log.debug(
                    "equity_lookup scenario fallback: %s → %s (used %s)",
                    scenario,
                    chosen.split("|", 4)[-1],  # just the scenario part
                    prefix,
                )
                return self._table[chosen]

        # 3. Double miss — on-the-fly compute if a handler is wired (serve path), else None
        self._miss_count += 1
        if self._on_miss is not None:
            computed = self._on_miss(hole_canonical, board_canonical, scenario)
            if computed is not None:
                return computed
        log.warning("equity_lookup MISS: hole=%s board=%s scen=%s", hole_canonical, board_canonical, scenario)
        return None

    def miss_rate(self) -> float:
        """Fraction of get() calls that could not be served from exact key."""
        total = self._miss_count + self._hit_count
        return self._miss_count / total if total > 0 else 0.0

    def population_mean_equity(self) -> dict[str, float]:
        """Return population-average equity dict for use when lookup returns None."""
        if not self._table:
            return {col: 0.5 for col in _DECILE_COLS} | {"mean": 0.5, "variance": 0.0}
        # Compute mean over all rows
        sums: dict[str, float] = {k: 0.0 for k in next(iter(self._table.values()))}
        n = len(self._table)
        for row in self._table.values():
            for k, v in row.items():
                sums[k] += v
        return {k: v / n for k, v in sums.items()}

    @property
    def n_total(self) -> int:
        return self._n_total
