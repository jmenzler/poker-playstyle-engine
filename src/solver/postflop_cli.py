"""postflop-cli subprocess wrapper — Phase 6: OQ-1 resolved.

Implements the SolverBackend protocol from docs/research/SOLVER.md.

Phase 6 contract (this file):
- ``PostflopCliBackend.is_available()`` works and is fully tested.
- ``PostflopCliBackend.solve()`` parses the binary's solve_mode JSON
  (``actions_root`` + ``aggregate_freq_root``) into a SolverResult whose
  ``action_dist`` is mapped to the project's 15-action canonical vocab.
- Stage B (`src/study/solver_verify.py`, Plan 06-03) is the first non-test caller.

OQ-1 resolution (Phase 6 — see .planning/phases/06-study-tool/06-02-PLAN.md):
    postflop-cli's ``solve_mode`` Output struct already emits ``actions_root`` and
    ``aggregate_freq_root`` (verified 2026-05-19 in tools/postflop-cli/src/main.rs
    lines 67-77). We parse those fields and map BET/RAISE chip amounts to the
    project's 15-action vocab via ``_map_solver_to_canonical`` with unconditional
    snap-to-nearest-bucket (the vocab is intentionally coarse).

OQ-1 lock tables (locked by planner; do NOT mutate without a documented decision):
    _BET_RATIO_BUCKETS   — ratio = chips / pot_at_decision
    _RAISE_RATIO_BUCKETS — ratio = (TO_chips - villain_bet_total) / to_call
    _SNAP_TOLERANCE      — 0.20; sizes _BET_OVERBET_FLOOR (bet_150's edge)

The catch-all ``bet_overbet`` absorbs any BET ratio >= _BET_OVERBET_FLOOR
(bet_150's tolerance edge, 1.80). Interior BET ratios snap to the nearest bin.

Design notes preserved from Phase 4:
    - subprocess.run, not Popen (one spot at a time; synchronous).
    - msgspec.Struct(frozen=True, kw_only=True) for SolverSpot / SolverResult
      matches project-wide convention (src/_config.py TsdbEnv / MilvusEnv).
    - SolverSpot uses postflop-cli-native per-street bet_sizes_* dicts; Phase 5
      can layer a TexasSolver translation later if needed.
"""

# long-ok-file

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import msgspec

from src._errors import NoStrategyError, SolverParseError
from src._log import get_logger

log = get_logger("solver.postflop_cli")


# ----------------------------------------------------------------------------
# OQ-1 Bucket Snap Tables (LOCKED — see 06-02-PLAN.md <oq1_lock>)
# ----------------------------------------------------------------------------
#
# postflop-cli returns absolute chip amounts (e.g. "BET 50", "RAISE 200",
# "ALLIN 500"). We snap to the project's 15-action vocab via nearest-neighbor
# ratio match. CANONICAL_ACTIONS is defined in src/decision_engine/blending.py
# — these bucket names MUST be a subset of that tuple.

_BET_RATIO_BUCKETS: tuple[tuple[float, str], ...] = (
    (0.25, "bet_25"),
    (0.33, "bet_33"),
    (0.50, "bet_50"),
    (0.75, "bet_75"),
    (1.00, "bet_100"),
    (1.50, "bet_150"),
    (2.50, "bet_overbet"),  # any ratio >= 2.0 maps here (catch-all)
)

_RAISE_RATIO_BUCKETS: tuple[tuple[float, str], ...] = (
    # ratio = (raise_TO_chips - villain_bet_total) / to_call (increment over prev bet)
    (1.00, "raise_min"),  # min-raise relative to prev bet
    (2.50, "raise_2_5x"),
    (3.00, "raise_3x"),
    (4.00, "raise_pot"),  # also the catch-all: any ratio >= 4.0 maps here
)

# Sizes the bet_overbet floor (see below). No longer a rejection gate — interior
# BET ratios snap to the nearest coarse bucket unconditionally.
_SNAP_TOLERANCE: float = 0.20

# bet_overbet absorbs anything above the highest finite bucket's tolerance edge.
# Without this, ratios in (bet_150*1.2, 2.0) miss bet_150 by >tolerance AND fall
# short of a hardcoded 2.0 overbet floor → the whole node is rejected.
_BET_OVERBET_FLOOR: float = _BET_RATIO_BUCKETS[-2][0] * (1.0 + _SNAP_TOLERANCE)


class SolverSpot(msgspec.Struct, frozen=True, kw_only=True):
    """Input to a postflop-cli solve.

    Required fields match the postflop-cli JSON stdin schema documented in
    docs/research/SOLVER.md §postflop-cli Interface §Input schema.
    """

    pot: int
    """Starting pot in chips (integer)."""

    effective_stack: int
    """Effective stack in chips (integer)."""

    board: list[str]
    """Board cards as a list of card strings, e.g. ["Ah", "7c", "2d"]."""

    range_ip: str
    """In-position range in standard notation, e.g. "AA,KK,AKs:0.75"."""

    range_oop: str
    """Out-of-position range in standard notation."""

    # Optional previous-bet hero is facing at this decision. Used by
    # _map_solver_to_canonical to compute RAISE ratios. None / 0 means
    # no prior bet pending (open-action node).
    prev_bet: int | None = None

    # Optional per-street bet size arrays (postflop-cli-native shape).
    # Default is None — omitted from JSON payload; postflop-cli applies its
    # own defaults (typically ["33%", "66%", "e", "a"] per street).
    bet_sizes_flop_oop: list[str] | None = None
    bet_sizes_flop_ip: list[str] | None = None
    bet_sizes_turn_oop: list[str] | None = None
    bet_sizes_turn_ip: list[str] | None = None
    bet_sizes_river_oop: list[str] | None = None
    bet_sizes_river_ip: list[str] | None = None

    max_iterations: int | None = None
    """Maximum CFR iterations. None → postflop-cli default."""

    target_exploitability_pct: float | None = None
    """Stop when exploitability ≤ this percentage of pot. None → default."""

    memory_budget_mb: int | None = None
    """Per-spot budget for the binary's adaptive memory gate. None → no gate."""


class SolverResult(msgspec.Struct, frozen=True, kw_only=True):
    """Output from postflop-cli: solved result.

    ``action_dist`` is the canonical Phase 5 target shape per
    docs/research/FEATURES-v2.md decide-time action distribution. Phase 6
    populates this dict by parsing solve_mode JSON (OQ-1 resolved).
    """

    action_dist: dict[str, float]
    """Solved action probability distribution, keyed by canonical action label.

    Phase 6: populated by _map_solver_to_canonical; sums to 1.0 ± 0.001.
    """

    exploitability_pct: float
    """Final exploitability percentage from postflop-cli stdout."""

    solve_time_ms: int
    """Wall-clock time reported by postflop-cli (time_ms field)."""

    distortion: str = "none"
    """Adaptive memory gate distortion level: none/light/medium/heavy."""

    applied_prune: float = 0.0
    """Range-prune weight threshold the gate applied (0.0 = no prune)."""

    applied_n_sizes: int = 0
    """Per-street bet-size count after gate trimming (0 = binary omitted the field)."""

    final_effective_stack: int = 0
    """Effective stack the solve actually used after any cap (chips; 0 = field absent)."""

    mem_estimate_mb: int = 0
    """Compressed memory estimate of the chosen rung (MB)."""


class PostflopCliBackend:
    """Implements SolverBackend protocol from docs/research/SOLVER.md.

    Phase 6: solve() is fully implemented — parses solve_mode JSON output and
    returns a populated SolverResult.

    Usage:
        backend = PostflopCliBackend(binary_path=Path("~/postflop-cli/...").expanduser())
        if not backend.is_available():
            raise RuntimeError("postflop-cli binary not found")
        result = backend.solve(spot, timeout_s=600.0)
    """

    def __init__(self, binary_path: Path) -> None:
        self._bin = binary_path

    def is_available(self) -> bool:
        """Return True only when the binary path points to an existing file."""
        return self._bin.exists() and self._bin.is_file()

    def solve(self, spot: SolverSpot, *, timeout_s: float = 600.0) -> SolverResult:
        """Run postflop-cli and parse solve_mode output into a SolverResult.

        OQ-1 resolution (Phase 6): postflop-cli already emits actions_root +
        aggregate_freq_root in its solve_mode Output struct. We parse and map
        to the project's 15-action vocab via _map_solver_to_canonical with
        snap-to-nearest-bucket (20% relative-error tolerance).

        Args:
            spot: SolverSpot with pot, prev_bet, ranges, board, etc.
            timeout_s: subprocess timeout in seconds (default 600).

        Returns:
            SolverResult with action_dist mapped to canonical vocab,
            sum(action_dist.values()) == 1.0 ± 0.001.

        Raises:
            NoStrategyError: subprocess exited non-zero (solver crashed/timed out).
            SolverParseError: stdout not valid JSON OR a BET/RAISE ratio is
                > _SNAP_TOLERANCE (20%) off the nearest canonical bucket OR an
                unknown solver label was returned OR
                len(actions_root) != len(aggregate_freq_root).
            subprocess.TimeoutExpired: subprocess ran longer than ``timeout_s``.
        """
        payload = _spot_to_payload(spot)
        input_json = json.dumps(payload)

        log.info(
            "solver.postflop_cli.solve.start",
            pot=spot.pot,
            effective_stack=spot.effective_stack,
            board=spot.board,
            timeout_s=timeout_s,
        )

        result = subprocess.run(
            [str(self._bin)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if result.returncode != 0:
            log.error(
                "solver.postflop_cli.nonzero_exit",
                returncode=result.returncode,
                stderr=result.stderr[:500],
            )
            raise NoStrategyError(f"postflop-cli exited {result.returncode}: {result.stderr[:200]}")

        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            log.error("solver.postflop_cli.invalid_json", stdout_head=result.stdout[:200])
            raise SolverParseError(f"postflop-cli stdout not valid JSON: {e}") from e

        prev_bet = spot.prev_bet if spot.prev_bet is not None else 0
        # At the root hero has no street wager yet, so villain's total IS the prev bet.
        action_dist = _map_solver_to_canonical(
            parsed["actions_root"],
            parsed["aggregate_freq_root"],
            pot_at_decision=float(spot.pot),
            prev_bet_at_decision=float(prev_bet),
            villain_bet_total=float(prev_bet),
        )
        log.info(
            "solver.postflop_cli.solve.done",
            exploitability_pct=parsed["exploitability_pct"],
            time_ms=parsed["time_ms"],
            n_canonical_actions=len(action_dist),
        )
        return SolverResult(
            action_dist=action_dist,
            exploitability_pct=float(parsed["exploitability_pct"]),
            solve_time_ms=int(parsed["time_ms"]),
            distortion=str(parsed.get("distortion", "none")),
            applied_prune=float(parsed.get("applied_prune", 0.0)),
            applied_n_sizes=int(parsed.get("applied_n_sizes", 0)),
            final_effective_stack=int(parsed.get("final_effective_stack", 0)),
            mem_estimate_mb=int(parsed.get("mem_estimate_mb", 0)),
        )

    def solve_harvest(
        self,
        spot: SolverSpot,
        nav_lines: list[dict[str, object]],
        *,
        timeout_s: float = 600.0,
    ) -> list[SolverResult | None]:
        """Solve one flop-rooted spot once, navigate every nav_line, return per-line results.

        nav_ok lines map via _map_solver_to_canonical with pot_at_decision=result.pot_at_node
        (the real node pot, NOT spot.pot) so deep bets snap right; nav_ok=false -> None.
        Returns a list of length len(nav_lines). Raises NoStrategyError on non-zero exit,
        SolverParseError on bad JSON / off-bucket label, TimeoutExpired past timeout_s.
        """  # long-ok
        parsed, results = self.solve_harvest_raw(spot, nav_lines, timeout_s=timeout_s)
        out = [raw_harvest_to_solver_result(result, parsed) for result in results]

        log.info(
            "solver.postflop_cli.solve_harvest.done",
            exploitability_pct=parsed["exploitability_pct"],
            time_ms=parsed["time_ms"],
            n_results=len(out),
        )
        return out

    def solve_harvest_raw(
        self,
        spot: SolverSpot,
        nav_lines: list[dict[str, object]],
        *,
        timeout_s: float = 600.0,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Solve one flop-rooted spot and return the raw per-DP harvest dicts.

        Returns (parsed_top_level, parsed["results"]) — the raw list includes
        the 17-01 grid blocks (hero_grid/villain_grid) plus nav_ok/actions and
        the untouched full-set fields.  Use this for the range-viewer contract;
        use solve_harvest for the corpus/canonical-action path.

        Raises NoStrategyError on non-zero exit, SolverParseError on bad JSON,
        TimeoutExpired past timeout_s.
        """  # long-ok
        payload = _spot_to_payload(spot)
        payload["mode"] = "solve_harvest"
        payload["nav_lines"] = nav_lines
        input_json = json.dumps(payload)

        log.info(
            "solver.postflop_cli.solve_harvest_raw.start",
            pot=spot.pot,
            effective_stack=spot.effective_stack,
            board=spot.board,
            n_lines=len(nav_lines),
            timeout_s=timeout_s,
        )

        proc = subprocess.run(
            [str(self._bin)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if proc.returncode != 0:
            log.error(
                "solver.postflop_cli.harvest_raw_nonzero_exit",
                returncode=proc.returncode,
                stderr=proc.stderr[:500],
            )
            raise NoStrategyError(f"postflop-cli exited {proc.returncode}: {proc.stderr[:200]}")

        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            log.error("solver.postflop_cli.harvest_raw_invalid_json", stdout_head=proc.stdout[:200])
            raise SolverParseError(f"postflop-cli stdout not valid JSON: {e}") from e

        log.info(
            "solver.postflop_cli.solve_harvest_raw.done",
            exploitability_pct=parsed.get("exploitability_pct"),
            time_ms=parsed.get("time_ms"),
            n_results=len(parsed.get("results", [])),
        )
        return parsed, parsed["results"]


def raw_harvest_to_solver_result(result: dict[str, Any], parsed_top: dict[str, Any]) -> SolverResult | None:
    """Map one raw harvest result dict to a SolverResult; nav_ok=false -> None.

    An all-zero action_dist (hero's range carries no weight at the navigated
    node) is a degenerate navigation, not a mapping error -> None as well.
    """
    if not result["nav_ok"]:
        return None
    labels = [pair[0] for pair in result["action_dist"]]
    freqs = [float(pair[1]) for pair in result["action_dist"]]
    if sum(freqs) <= 0.0:
        return None
    action_dist = _map_solver_to_canonical(
        labels,
        freqs,
        pot_at_decision=float(result["pot_at_node"]),
        prev_bet_at_decision=float(result.get("to_call") or 0.0),
        villain_bet_total=float(result.get("villain_bet_total") or 0.0),
    )
    return SolverResult(
        action_dist=action_dist,
        exploitability_pct=float(parsed_top["exploitability_pct"]),
        solve_time_ms=int(parsed_top["time_ms"]),
        distortion=str(parsed_top.get("distortion", "none")),
        applied_prune=float(parsed_top.get("applied_prune", 0.0)),
        applied_n_sizes=int(parsed_top.get("applied_n_sizes", 0)),
        final_effective_stack=int(
            result.get("final_effective_stack", parsed_top.get("final_effective_stack", 0))
        ),
        mem_estimate_mb=int(parsed_top.get("mem_estimate_mb", 0)),
    )


# ----------------------------------------------------------------------------
# Module-level helpers — OQ-1 bucket-snap + canonical mapping
# ----------------------------------------------------------------------------


def _snap_to_bucket(
    ratio: float,
    buckets: tuple[tuple[float, str], ...],
    action_label: str,
) -> str:
    """Find nearest bucket by absolute ratio distance (unconditional snap).

    The 15-action vocab is intentionally coarse: geometric solver sizing ('e')
    routinely lands between buckets, so an interior BET snaps to the nearest bin
    rather than rejecting the node. The catch-all ``bet_overbet`` (the only
    "overbet" suffix in _BET_RATIO_BUCKETS) short-circuits any ratio >=
    _BET_OVERBET_FLOOR. True garbage (negative / non-finite chips) is rejected by
    the caller before this is reached.

    Args:
        ratio: observed chips / reference (pot or prev_bet).
        buckets: ordered tuple of ``(reference_ratio, bucket_name)`` pairs.
        action_label: original solver label (for error context only).

    Returns:
        The matching bucket name from ``buckets``.
    """
    _last_ratio, last_name = buckets[-1]
    if last_name == "bet_overbet" and ratio >= _BET_OVERBET_FLOOR:
        return last_name

    return min(buckets, key=lambda b: abs(b[0] - ratio))[1]


def _map_solver_to_canonical(
    solver_actions: list[str],
    solver_freqs: list[float],
    pot_at_decision: float,
    prev_bet_at_decision: float,
    villain_bet_total: float = 0.0,
) -> dict[str, float]:
    """Map postflop-cli action labels to the project's 15-action canonical vocab.

    Solver emits one of:
        - "CHECK", "FOLD", "CALL"  — exact label match
        - "BET <chips>"             — snap chips/pot to _BET_RATIO_BUCKETS
        - "RAISE <chips>"           — chips is the TO-amount; snap
          (chips - villain_bet_total) / prev_bet to _RAISE_RATIO_BUCKETS
        - "ALLIN <chips>"           — always maps to "allin"

    Multiple solver actions may map to the same project bucket (different
    chip amounts both snap to "bet_50"); their frequencies are summed.

    Args:
        solver_actions: list of label strings from ``parsed["actions_root"]``.
        solver_freqs: list of frequencies in [0, 1], same length as
            ``solver_actions``.
        pot_at_decision: pot size when hero faces the decision (chips).
        prev_bet_at_decision: chips hero must call (0 if no bet pending).
        villain_bet_total: villain's total street wager (RAISE TO-amounts are
            relative to it). 0 falls back to prev_bet_at_decision.

    Returns:
        dict ``{canonical_action: freq}``. Sums to 1.0 ± 0.001;
        re-normalized defensively if the solver-returned freqs do not
        already sum to ~1.0.

    Raises:
        SolverParseError: a BET label carries non-positive / non-finite chips,
            a RAISE label appears at a node with prev_bet 0, an unknown solver
            label is returned, or ``len(solver_actions) != len(solver_freqs)``.
            Legitimate interior BET/RAISE ratios snap to the nearest coarse
            bucket and never raise.
    """
    if len(solver_actions) != len(solver_freqs):
        raise SolverParseError(
            f"actions_root len {len(solver_actions)} != aggregate_freq_root len {len(solver_freqs)}"
        )

    out: dict[str, float] = {}
    for action, freq in zip(solver_actions, solver_freqs, strict=True):
        action_upper = action.upper().strip()
        bucket: str
        if action_upper == "CHECK":
            bucket = "check"
        elif action_upper == "FOLD":
            bucket = "fold"
        elif action_upper == "CALL":
            bucket = "call"
        elif action_upper.startswith("BET "):
            chips = float(action.split()[1])
            if not math.isfinite(chips) or chips <= 0:
                raise SolverParseError(f"BET label {action!r} has non-positive/non-finite chips")
            ratio = chips / max(pot_at_decision, 1e-9)
            # Below-floor catch-all mirroring bet_overbet: deep pots produce real
            # sub-25%-pot probes the vocab can only floor to bet_25.
            if ratio <= _BET_RATIO_BUCKETS[0][0]:
                bucket = _BET_RATIO_BUCKETS[0][1]
            else:
                bucket = _snap_to_bucket(ratio, _BET_RATIO_BUCKETS, action_label=action)
        elif action_upper.startswith("RAISE "):
            if prev_bet_at_decision <= 0:
                raise SolverParseError(
                    f"RAISE label {action!r} at a node with prev_bet 0 — "
                    "to_call/villain_bet_total missing (stale postflop-cli binary?)"
                )
            chips = float(action.split()[1])
            if not math.isfinite(chips) or chips <= 0:
                raise SolverParseError(f"RAISE label {action!r} has non-positive/non-finite chips")
            base = villain_bet_total if villain_bet_total > 0 else prev_bet_at_decision
            # RAISE labels are TO-amounts; the buckets are increment-over-prev-bet ratios.
            # Nearest-bucket without the tolerance gate: the raise buckets are sparse
            # (1.0/2.5/3.0/4.0), so legitimate sizes (e.g. 1.5x) fall in the gaps; the
            # canonical vocab can only approximate them.
            ratio = (chips - base) / prev_bet_at_decision
            bucket = min(_RAISE_RATIO_BUCKETS, key=lambda b: abs(b[0] - ratio))[1]
        elif action_upper.startswith("ALLIN"):
            bucket = "allin"
        else:
            raise SolverParseError(f"unknown solver action label: {action!r}")
        out[bucket] = out.get(bucket, 0.0) + float(freq)

    # Renormalize defensively: solver freqs should already sum to ~1.0, but
    # bucket merging may amplify rounding noise; protect downstream callers
    # that assert sum == 1.0 ± 0.001.
    total = sum(out.values())
    if total <= 0:
        raise SolverParseError(f"action_dist sums to {total}; expected ~1.0")
    if abs(total - 1.0) > 0.001:
        out = {k: v / total for k, v in out.items()}
    return out


def _spot_to_payload(spot: SolverSpot) -> dict[str, object]:
    """Serialize SolverSpot to the postflop-cli JSON input payload.

    Only required fields and explicitly set optional fields are included.
    None-valued fields are omitted so postflop-cli applies its own defaults.

    ``prev_bet`` is consumed by _map_solver_to_canonical (RAISE ratios) and
    is NOT forwarded to postflop-cli — the binary's own street walk derives
    the betting tree from ranges + board.
    """
    payload: dict[str, object] = {
        "pot": spot.pot,
        "effective_stack": spot.effective_stack,
        "board": spot.board,
        "range_ip": spot.range_ip,
        "range_oop": spot.range_oop,
    }
    # Optional per-street bet sizes
    for field in (
        "bet_sizes_flop_oop",
        "bet_sizes_flop_ip",
        "bet_sizes_turn_oop",
        "bet_sizes_turn_ip",
        "bet_sizes_river_oop",
        "bet_sizes_river_ip",
    ):
        value = getattr(spot, field)
        if value is not None:
            payload[field] = value
    if spot.max_iterations is not None:
        payload["max_iterations"] = spot.max_iterations
    if spot.target_exploitability_pct is not None:
        payload["target_exploitability_pct"] = spot.target_exploitability_pct
    if spot.memory_budget_mb is not None:
        payload["memory_budget_mb"] = spot.memory_budget_mb
    return payload
