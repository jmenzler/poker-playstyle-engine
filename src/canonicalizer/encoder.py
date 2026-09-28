"""Canonicalizer: encode GameState → EncodeResult for Phase 3 DecisionEngine.  long-ok

Public API
----------
    from src.canonicalizer import Canonicalizer, EncodeResult

    c = Canonicalizer.default()
    result = c.encode(game_state)  # game_state: GameState
    # result.embedding: np.ndarray, shape (34,) preflop or (80,) postflop
    # result.hard_filter: dict with street_class, pot_type, hero_pos_rel, n_players_active
    # result.schema_version: int = 3

Architecture
------------
encode() is the single entry point. It:
  1. Converts GameState → DP-dict via _adapt() (one-way adapter)
  2. Dispatches by street to extract_preflop or extract_postflop
  3. Returns EncodeResult (NamedTuple — stable Phase 3 contract)

encode() is PURE: zero network I/O. The equity table Parquet is loaded once
at __init__ time; subsequent encode() calls touch only in-memory structures.

GameState → DP-dict Field Mapping
----------------------------------
The tools/ feature extractors expect a DP-dict shaped per DECISIONS-SCHEMA.md.
GameState uses BB-denominated floats; DP-dict uses both BB and cents fields.
We normalise to bb_cents=100 (NL10 equivalent) for DP-dict fields that need cents.

    GameState.street                 → dp["street"]
    GameState.hero_position          → dp["hero_pos"]
    GameState.hero_hole_cards        → dp["hero_hole"]         (list[str])
    GameState.board_cards            → dp["board"]             (list[str], empty for preflop)
    GameState.pot_size_bb            → dp["pot_bb"]
                                       dp["pot_cents"]         (pot_bb * 100)
    GameState.effective_stack_bb     → dp["spr"]               (eff_stack / pot, clamped ≥ 0)
                                       dp["hero_stack_bb"]
                                       dp["hero_stack_cents"]  (hero_stack_bb * 100)
    GameState.hero_facing_bet_bb     → dp["facing_size_pot_frac"] (facing_bet_bb / pot_bb, 0 if pot=0)
                                       dp["facing"]            ("check_to" or "bet_to" / "raise_to")
    GameState.hero_bet_size_bb       → dp["hero_action_to_bb"]
                                       dp["hero_action_to_cents"]
    GameState.action_sequence        → dp["action_so_far_street"]
                                       (each token "POS:verb" → {"pos": POS, "action": verb_stem})
    GameState.opponents_remaining    → dp["n_players_at_street"]  (opponents + 1)
    GameState.prior_street_aggressor → dp["preflop_aggressor"]  (Position str or None)

Derived / defaulted fields:
    dp["hero_pos_rel"]      = postflop-relative IP/OOP from preflop aggressor seat (preflop + postflop)
    dp["hero_hole_class"]   = joint-canonical hole string (used by preflop Group I features)
    dp["board_canonical"]   = joint-canonical board string
    dp["scenario_key"]      = "<pot_type>|<hero_pos_rel>|n<n_players>" (approximate)
    dp["pot_type"]          = inferred from action_sequence raise count
    dp["bb_cents"]          = 100  (normalisation constant)
    dp["preflop_action_seq"]= []   (no preflop sequence available from GameState v1)
    dp["facing_pos"]        = prior_street_aggressor (villain seat, proxy for facing_pos)

Edge Cases
----------
- preflop + empty board: board=[] accepted by joint_canonicalize; board_canonical=""
- equity table miss: extractor uses population_mean fallback (no exception)
- equity table absent at init: raises ConfigError with actionable message
- malformed cards: ValueError propagates from joint_canonicalize (no swallowing)
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src._errors import ConfigError
from src._log import get_logger
from src.protocols.game_state import GameState
from tools.equity_lookup import EquityLookup, OnMiss
from tools.feature_extractors.postflop import extract_postflop
from tools.feature_extractors.preflop import extract_preflop
from tools.joint_canonicalize import joint_canonicalize
from tools.position import last_raiser_pos, preflop_pos_rel, preflop_pot_type

log = get_logger("canonicalizer.encoder")


def _resolve_equity_on_miss() -> OnMiss | None:
    """Wire serve-time on-the-fly equity when EQUITY_DECILE_BIN points at the binary.

    Returns None (→ pop_mean fallback) when unset or unavailable. The corpus build path
    constructs EquityLookup directly without this, so its misses still go to the ledger.
    """
    bin_path = os.environ.get("EQUITY_DECILE_BIN")
    if not bin_path:
        return None
    from tools.equity_compute import EquityComputer

    computer = EquityComputer(Path(bin_path))
    if not computer.is_available():
        log.warning("encoder.equity_decile_unavailable", path=bin_path)
        return None
    log.info("encoder.equity_decile_wired", path=bin_path)
    return computer.compute


# Candidate paths for equity_table.parquet relative to project root / tools/
_EQUITY_TABLE_CANDIDATES: tuple[str, ...] = (
    "tools/equity_table.parquet",
    "equity_table.parquet",
)

# Pattern to parse action tokens "POS:verb_suffix" → {"pos": POS, "action": verb_stem}
_TOKEN_RE = re.compile(r"^([A-Z]+):(.+)$")

# Normalisation constant: treat 1 BB = 100 cents throughout the adapter
_BB_CENTS = 100


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class EncodeResult(NamedTuple):
    """Stable output contract for Phase 3 DecisionEngine consumption.

    Fields:
        embedding:      float32 ndarray, shape (34,) for preflop or (80,) for postflop
        hard_filter:    dict with keys street_class, pot_type, hero_pos_rel, n_players_active
        schema_version: always 3 (FEATURES-v2.md v3 amendment)
    """

    embedding: np.ndarray
    hard_filter: dict  # type: ignore[type-arg]
    schema_version: int = 3


# ---------------------------------------------------------------------------
# Internal adapter
# ---------------------------------------------------------------------------


_STREET_DELIM = "/"


def _infer_pot_type(action_sequence: tuple[str, ...]) -> str:
    """Infer pot_type from the PREFLOP raise count (opens count as the first raise).

    Slices to the segment before the first "/" street delimiter so postflop
    raises (raise_half_pot/raise_pot) can't inflate the preflop raise count.
    """
    preflop: list[str] = []
    for tok in action_sequence:
        if tok == _STREET_DELIM:
            break
        preflop.append(tok)
    return preflop_pot_type(_parse_action_tokens(tuple(preflop)))


def _parse_action_tokens(action_sequence: tuple[str, ...]) -> list[dict]:  # type: ignore[type-arg]
    """Convert action_sequence tokens to DP-dict action_so_far_street format.

    Each token has format "POS:verb" (e.g. "BTN:open_2.5", "BB:call").
    Returns a list of {"pos": str, "action": str} dicts.
    The verb is mapped to its canonical stem for extractor consumption:
        open_* / raise_* → "raise"
        call            → "call"
        check           → "check"
        fold            → "fold"
        bet_*           → "bet"
        allin           → "allin"
    """
    result: list[dict] = []  # type: ignore[type-arg]
    for tok in action_sequence:
        m = _TOKEN_RE.match(tok)
        if not m:
            continue
        pos, verb = m.group(1), m.group(2).lower()
        if (
            verb.startswith("open")
            or verb.startswith("raise")
            or verb.startswith("3b")
            or verb.startswith("4b")
            or verb.startswith("5b")
        ):
            action = "raise"
        elif verb.startswith("bet"):
            action = "bet"
        elif verb == "call":
            action = "call"
        elif verb == "check":
            action = "check"
        elif verb == "fold":
            action = "fold"
        elif verb == "allin":
            action = "allin"
        else:
            action = verb  # pass through unknown verbs unchanged
        result.append({"pos": pos, "action": action})
    return result


def _adapt(game_state: GameState) -> dict:  # type: ignore[type-arg]
    """Convert a GameState into a DP-dict compatible with tools/ feature extractors.

    See module docstring for the full field mapping. This is a one-way adapter;
    only the fields the extractors consume are populated.
    """
    gs = game_state

    # --- Cards ---
    hole = list(gs.hero_hole_cards)
    board = list(gs.board_cards)

    # Joint-canonical forms (used for equity lookup + preflop hole_class)
    h_can, b_can = joint_canonicalize(hole, board)

    # --- Sizes ---
    pot_bb = float(gs.pot_size_bb)
    eff_stack_bb = float(gs.effective_stack_bb)
    facing_bet_bb = float(gs.hero_facing_bet_bb)
    hero_bet_bb = float(gs.hero_bet_size_bb)

    spr = eff_stack_bb / pot_bb if pot_bb > 0 else 0.0
    facing_size_pot_frac = facing_bet_bb / pot_bb if pot_bb > 0 else 0.0

    # --- Facing classification ---
    facing = "bet_to" if facing_bet_bb > 0 else "check_to"

    # --- Action sequence ---
    action_so_far = _parse_action_tokens(gs.action_sequence)

    # Refine facing: if any raise in action_so_far, it's raise_to
    if any(a["action"] in ("raise", "allin") for a in action_so_far):
        facing = "raise_to"

    # --- Players ---
    n_players = gs.opponents_remaining + 1  # hero + opponents

    # --- Pot type ---
    pot_type = gs.pot_type if gs.pot_type is not None else _infer_pot_type(gs.action_sequence)

    # --- Scenario key ---
    # hero_pos_rel derived from aggressor seat so build and serve agree
    if gs.street == "preflop":
        hero_pos_rel = preflop_pos_rel(gs.hero_position, last_raiser_pos(action_so_far))
    else:
        # postflop pos_rel approximated from prior_street_aggressor; exact multi-way
        # derivation deferred to GameState v2
        hero_pos_rel = preflop_pos_rel(gs.hero_position, gs.prior_street_aggressor)
    scenario_key = f"{pot_type}|{hero_pos_rel}|n{n_players}"

    # --- Prior street aggressor ---
    preflop_aggressor = gs.prior_street_aggressor  # Position str or None

    return {
        # Street + position
        "street": gs.street,
        "hero_pos": gs.hero_position,
        "hero_pos_rel": hero_pos_rel,
        "n_players_at_street": n_players,
        # Cards
        "hero_hole": hole,
        "hero_hole_class": h_can,  # joint-canonical used as hole class approximation
        "board": board,
        "board_canonical": b_can,
        # Pot / stack geometry
        "pot_bb": pot_bb,
        "pot_cents": pot_bb * _BB_CENTS,
        "bb_cents": _BB_CENTS,
        "spr": spr,
        "hero_stack_bb": eff_stack_bb,
        "hero_stack_cents": eff_stack_bb * _BB_CENTS,
        # Facing info
        "facing": facing,
        "facing_size_pot_frac": facing_size_pot_frac,
        "facing_pos": gs.prior_street_aggressor,
        # Hero bet
        "hero_action_to_bb": hero_bet_bb,
        "hero_action_to_cents": hero_bet_bb * _BB_CENTS,
        # Action history
        "action_so_far_street": action_so_far,
        "preflop_action_seq": [],  # not available from GameState v1
        "preflop_aggressor": preflop_aggressor,
        # Pot type
        "pot_type": pot_type,
        # Equity lookup
        "scenario_key": scenario_key,
    }


# ---------------------------------------------------------------------------
# Canonicalizer
# ---------------------------------------------------------------------------


class Canonicalizer:
    """Encode a GameState into a fixed-dimension embedding vector.

    Usage:
        c = Canonicalizer.default()
        result = c.encode(game_state)

    The equity table is loaded once at construction; encode() is pure (no I/O).
    """

    def __init__(self, equity_table_path: Path | None = None) -> None:
        """Initialise with explicit equity table path.

        Args:
            equity_table_path: Path to equity_table.parquet. Must exist.
                               Use Canonicalizer.default() for automatic resolution.

        Raises:
            ConfigError: if equity_table_path does not exist.
        """
        if equity_table_path is None:
            raise ConfigError(
                "equity_table_path is required. Use Canonicalizer.default() for automatic path resolution."
            )
        if not equity_table_path.exists():
            raise ConfigError(
                f"equity table not found at {equity_table_path}. "
                "Run tools/build_equity_table.py first to generate it."
            )
        # Merge postflop + preflop equity tables (preflop optional — warn if absent)
        equity_paths: list[str | Path] = [equity_table_path]
        preflop_table = equity_table_path.parent / "preflop_equity_table.parquet"
        if preflop_table.exists():
            equity_paths.append(preflop_table)
        self._equity_lookup = EquityLookup(
            equity_paths if len(equity_paths) > 1 else equity_paths[0],
            on_miss=_resolve_equity_on_miss(),
        )
        self._pop_mean = self._equity_lookup.population_mean_equity()

        # Build tightness + range lookups for preflop Group J/K features
        from tools.villain_range_stats import build_range_lookup, build_tightness_lookup

        here = Path(__file__).resolve()
        palette_candidates = [
            here.parent.parent.parent / "research" / "preflop-ranges" / "outputs",
        ]
        self._tightness_lookup: dict[str, float] = {}
        self._range_lookup: dict[str, dict[str, float]] = {}
        for candidate in palette_candidates:
            if candidate.exists():
                self._tightness_lookup = build_tightness_lookup(candidate)
                self._range_lookup = build_range_lookup(candidate)
                break

    @classmethod
    def _build_default(cls) -> Canonicalizer:
        """Auto-resolve equity_table.parquet from the repo root or a parent worktree.

        Walks up the directory tree from this file looking for tools/equity_table.parquet.
        This handles both normal checkout (file is 3 dirs up from encoder.py) and worktree
        layouts where the file may be in the parent main-repo (e.g. nested-checkouts/…).

        Raises ConfigError with an actionable message if not found anywhere in the chain.
        """
        here = Path(__file__).resolve()
        searched: list[str] = []

        # Walk each ancestor up to filesystem root
        for ancestor in [here, *here.parents]:
            for candidate in _EQUITY_TABLE_CANDIDATES:
                path = ancestor / candidate
                searched.append(str(path))
                if path.exists():
                    return cls(equity_table_path=path)

        # Also try tools/ relative to cwd (for scripts running from repo root)
        for candidate in _EQUITY_TABLE_CANDIDATES:
            path = Path(candidate).resolve()
            if path.exists():
                return cls(equity_table_path=path)

        raise ConfigError(
            "equity table not found. Run tools/build_equity_table.py to generate it. "
            f"Searched {len(searched)} candidate paths (first few): {searched[:4]}"
        )

    @classmethod
    def default(cls) -> Canonicalizer:
        """Return the process-scoped singleton. Parquet loaded once; failed build retried next call."""
        global _default_canonicalizer
        if _default_canonicalizer is None:
            with _default_lock:
                if _default_canonicalizer is None:
                    _default_canonicalizer = cls._build_default()
        return _default_canonicalizer

    def encode(self, game_state: GameState) -> EncodeResult:
        """Encode a GameState into a fixed-dimension embedding + hard_filter dict.

        Args:
            game_state: Immutable decision point snapshot.

        Returns:
            EncodeResult with:
                embedding:      float32 ndarray of shape (34,) for preflop, (80,) for postflop
                hard_filter:    {street_class, pot_type, hero_pos_rel, n_players_active}
                schema_version: 3

        Raises:
            ValueError:  if card strings are malformed (propagated from joint_canonicalize)
            ConfigError: never raised here (only at __init__ time)
        """
        dp = _adapt(game_state)

        if game_state.street == "preflop":
            vector, hard_filter = extract_preflop(
                dp,
                self._equity_lookup,
                self._pop_mean,
                self._tightness_lookup or None,
                self._range_lookup or None,
            )
        else:
            vector, hard_filter = extract_postflop(dp, self._equity_lookup, self._pop_mean)

        return EncodeResult(
            embedding=vector,
            hard_filter=hard_filter,
            schema_version=3,
        )


_default_canonicalizer: Canonicalizer | None = None
_default_lock = threading.Lock()
