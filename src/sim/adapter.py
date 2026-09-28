"""SimAdapter: RLCard no-limit-holdem 6-max wrapper (ENGN-06).

Translates between RLCard env state and the project's GameState protocol.
Two responsibilities:
    1. raw_obs (RLCard dict) -> GameState (project frozen struct) — used by harness
    2. project_action (str)  -> RLCard action int (0-4) — used by harness step

CRITICAL PRE-CONDITIONS (RESEARCH.md):
    - Pitfall 2: env config key is ``game_num_players``, NOT ``num_players``.
        Wrong key silently falls back to 2-player.
    - Pitfall 5: RLCard 5-action set is coarser than project 15-action vocab.
        ``map_to_rlcard_action`` falls back to nearest legal action; never crashes.
    - RLCard 1.2.0 emits ``Action`` enums (not ints) in ``raw_obs['legal_actions']``;
        normalisation handles this transparently. ``stage`` is a ``Stage`` enum.
    - ``hand`` is suit-first uppercase (e.g. 'HK' = King of Hearts).
        ``_normalize_rlcard_card`` converts to project rank-first lowercase-suit form ('Kh').
"""

from __future__ import annotations

from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.nolimitholdem.round import Action

from src._log import get_logger
from src.protocols.game_state import GameState, Position
from src.sim.sized_env import make_sized_env
from src.sim.sizing import intended_raise_to_chips, nominal_raise_to_chips
from tools.position import aggressor_for_prior_street

log = get_logger("sim.adapter")

# RLCard Action enum → action_sequence verb. CHECK_CALL emits "call": the
# check-vs-call distinction never affects pot_type/aggressor/facing inference.
_ACTION_TO_VERB: dict[Action, str] = {
    Action.FOLD: "fold",
    Action.CHECK_CALL: "call",
    Action.RAISE_HALF_POT: "raise_half_pot",
    Action.RAISE_POT: "raise_pot",
    Action.ALL_IN: "allin",
}

_IN_HAND = (PlayerStatus.ALIVE, PlayerStatus.ALLIN)

# Pitfall 2: correct config key is game_num_players. RLCard NLHE big_blind=2 chips,
# so chips_for_each=200 -> 100bb starting stacks (standard depth, matches the HM3 corpus).
_DEFAULT_CONFIG: dict = {
    "game_num_players": 6,
    "chips_for_each": 200,
}

# RESEARCH.md Pattern 3 — 15-action vocab to RLCard 5-action set.
PROJECT_TO_RLCARD: dict[str, int] = {
    "fold": 0,
    "check": 1,
    "call": 1,
    # Preflop opens = RAISE_POT(3) not RAISE_HALF_POT(2): RLCard makes a half-pot
    # first-in open illegal (below min-raise), so 2 silently fell back to a limp.
    "open_2_2bb": 3,
    "open_3bb": 3,
    "3bet_3x": 3,
    "3bet_4x": 3,
    "4bet_2_5x": 3,
    "bet_25": 2,
    "bet_33": 2,
    "bet_50": 2,
    "bet_75": 3,
    "bet_100": 3,
    "bet_150": 3,
    "bet_overbet": 4,
    "raise_min": 2,
    "raise_2_5x": 3,
    "raise_3x": 3,
    "raise_pot": 3,
    "allin": 4,
}

# Passive fallback: CHECK_CALL > HALF_POT > FOLD > ALL_IN > POT (cheapest first).
_PASSIVE_FALLBACK_ORDER: tuple[int, ...] = (1, 2, 0, 4, 3)
# Aggressive fallback: largest legal raise / ALL_IN BEFORE CHECK_CALL, so a
# raise/bet/shove intent never silently downgrades to a passive call (C3/H1).
_AGGRESSIVE_FALLBACK_ORDER: tuple[int, ...] = (3, 4, 2, 1, 0)

# Verbs whose intent is aggression; everything else (fold/check/call) is passive.
_AGGRESSIVE_VERBS: frozenset[str] = frozenset(
    {
        "open_2_2bb",
        "open_3bb",
        "3bet_3x",
        "3bet_4x",
        "4bet_2_5x",
        "bet_25",
        "bet_33",
        "bet_50",
        "bet_75",
        "bet_100",
        "bet_150",
        "bet_overbet",
        "raise_min",
        "raise_2_5x",
        "raise_3x",
        "raise_pot",
        "allin",
    }
)


def intent_class(verb: str) -> str:
    """Classify a project verb's intent as ``"aggressive"`` or ``"passive"``."""
    return "aggressive" if verb in _AGGRESSIVE_VERBS else "passive"


# Seat names in postflop-action order; UTG sits 3 seats after the button.
_POSITION_BY_INDEX: tuple[Position, ...] = ("UTG", "MP", "CO", "BTN", "SB", "BB")

_STREETS_ORDER: tuple[str, ...] = ("preflop", "flop", "turn", "river")


def _hero_position(player_id: int, dealer_id: int, n_players: int = 6) -> Position:
    """Map an RLCard player_id to its seat, accounting for button rotation.
    RLCard seats SB at dealer+1, BB at dealer+2, UTG at dealer+3. Heads-up is the
    exception: button posts SB and acts first, so dealer=SB / other=BB (the 6-seat
    ring would mislabel them BTN/CO).
    """
    if n_players == 2:
        return "SB" if player_id == dealer_id else "BB"
    idx = (player_id - dealer_id - 3) % len(_POSITION_BY_INDEX)
    return _POSITION_BY_INDEX[idx]


def _action_tokens(
    action_record: list,
    dealer_id: int,
    street_boundaries: list[int] | None = None,
    n_players: int = 6,
) -> tuple[str, ...]:
    """Build "POS:verb" tokens from an RLCard record; insert "/" at each street
    boundary (first action of a new street) so consumers can slice preflop vs
    postflop. Without it streets flatten and preflop-raise counts (pot_type) inflate.
    """
    boundaries = set(street_boundaries or ())
    tokens: list[str] = []
    for i, (pid, action) in enumerate(action_record):
        if i in boundaries:
            tokens.append("/")
        pos = _hero_position(int(pid), dealer_id, n_players)
        tokens.append(f"{pos}:{_ACTION_TO_VERB.get(action, 'call')}")
    return tuple(tokens)


def _normalize_rlcard_card(rlcard_str: str) -> str:
    """Convert RLCard card format to project format.

    RLCard 1.2.0 emits suit-first uppercase (e.g. 'HK' = King of Hearts). The
    project format is rank-first with lowercase suit (e.g. 'Kh', 'Th' for tens).

    Args:
        rlcard_str: 2-char card string in either RLCard convention.

    Returns:
        Project format: rank (uppercase, T for 10) + suit (lowercase).
    """
    if len(rlcard_str) != 2:
        raise ValueError(f"Unexpected RLCard card length: {rlcard_str!r}")
    a, b = rlcard_str[0], rlcard_str[1]
    # If first char is a suit letter (H/D/C/S), swap rank/suit.
    if a.upper() in ("H", "D", "C", "S"):
        rank, suit = b, a
    else:
        rank, suit = a, b
    return rank.upper() + suit.lower()


def _coerce_int(value: object) -> int:
    """Coerce an RLCard ``Action`` enum (or int) to a plain Python int."""
    if hasattr(value, "value"):
        return int(value.value)  # type: ignore[attr-defined]
    return int(value)  # type: ignore[arg-type]


def _normalize_stage(stage: object) -> str:
    """RLCard 1.2.0 stage is a ``Stage`` enum (e.g. Stage.PREFLOP)."""
    s = str(stage.name).lower() if hasattr(stage, "name") else str(stage).lower()  # type: ignore[attr-defined]
    if s not in ("preflop", "flop", "turn", "river"):
        return "preflop"
    return s


def rlcard_state_to_gamestate(
    state: dict,
    *,
    player_id: int | None = None,
    dealer_id: int | None = None,
    active_count: int | None = None,
    prior_street_aggressor: Position | None = None,
) -> GameState:
    """Convert a raw RLCard ``env.get_state(player_id)`` dict to a project ``GameState``.

    Standalone counterpart of ``SimAdapter.next_game_state``. ``dealer_id`` and
    ``active_count`` are absent from the state dict; callers with env access
    should pass them, else position assumes button-at-seat-0 and field size is
    inferred from folds in the action record.
    """
    raw_obs = state.get("raw_obs", {}) if isinstance(state, dict) else {}
    if player_id is None:
        player_id = int(raw_obs.get("current_player", 0))

    stage = _normalize_stage(raw_obs.get("stage", "preflop"))

    hand = raw_obs.get("hand", []) or []
    hole_list = [_normalize_rlcard_card(c) for c in hand[:2]]
    hole = ("As", "Kh") if len(hole_list) != 2 else (hole_list[0], hole_list[1])

    public = raw_obs.get("public_cards", []) or []
    board = tuple(_normalize_rlcard_card(c) for c in public)

    all_chips = list(raw_obs.get("all_chips", []))
    stakes = list(raw_obs.get("stakes", []))
    n_players = max(len(all_chips), len(stakes), 2)
    if not all_chips:
        all_chips = [0] * n_players
    if not stakes:
        stakes = [0] * n_players
    pot = int(raw_obs.get("pot", sum(all_chips)))
    max_chips_in_pot = max(all_chips) if all_chips else 0

    action_record = state.get("action_record", []) if isinstance(state, dict) else []
    folded_ids = {int(pid) for pid, a in action_record if a == Action.FOLD}

    bb_chips = 2.0
    pot_bb = pot / bb_chips
    # Exclude folded players: a folded seat keeps its remaining chips in `stakes`,
    # so counting it understates effective stack / SPR (mirrors next_game_state).
    active_stacks = [s for i, s in enumerate(stakes) if s > 0 and i not in folded_ids]
    eff_stack_bb = (min(active_stacks) if active_stacks else 0) / bb_chips
    my_in_pot = int(all_chips[player_id]) if player_id < len(all_chips) else 0
    facing_bb = max(0.0, (max_chips_in_pot - my_in_pot) / bb_chips)
    bet_size_bb = my_in_pot / bb_chips

    _dealer = dealer_id if dealer_id is not None else 3  # seat-0=UTG fallback
    hero_position = _hero_position(player_id, _dealer, n_players)

    action_sequence = _action_tokens(action_record, _dealer, n_players=n_players)

    if active_count is not None:
        active = active_count
    else:
        n_seats = max(len(all_chips), len(stakes), 2)
        active = max(2, n_seats - len(folded_ids))
    opponents = max(0, active - 1)

    return GameState(
        street=stage,
        hero_position=hero_position,
        hero_hole_cards=hole,
        board_cards=board,
        pot_size_bb=float(pot_bb),
        effective_stack_bb=float(eff_stack_bb),
        hero_facing_bet_bb=float(facing_bb),
        hero_bet_size_bb=float(bet_size_bb),
        action_sequence=action_sequence,
        opponents_remaining=opponents,
        prior_street_aggressor=prior_street_aggressor,
    )


class SimAdapter:
    """Wraps RLCard no-limit-holdem 6-max environment; implements GameStateSource.

    Construction does NOT call ``env.reset()`` — the harness controls reset timing
    so that ``env.seed()`` can be called immediately before each reset (Pitfall 1).
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config if config is not None else _DEFAULT_CONFIG
        self._env = make_sized_env(cfg)
        self._last_state: dict | None = None
        self._last_player_id: int | None = None
        self._prev_stage: str | None = None
        self._street_start_idx: int = 0
        self._street_boundaries: list[int] = []
        self._street_aggressor_cache: dict[str, str | None] = {}
        # M3 observability: count executed-vs-intended size/action divergences by
        # intent class so silent downgrades are visible in harness metrics.
        self._divergence: dict[str, int] = {"aggressive": 0, "passive": 0}
        expected = cfg.get("game_num_players", 2)
        if self._env.num_players != expected:
            log.warning(
                "sim.adapter.player_count_mismatch",
                expected=expected,
                got=self._env.num_players,
            )

    def divergence_counts(self) -> dict[str, int]:
        """Executed-vs-intended divergence tally by intent class (M3)."""
        return dict(self._divergence)

    def map_to_rlcard_action(self, project_action: str, legal_actions: list) -> int:
        """Map a project verb to a legal RLCard enum int. Never crashes.

        Aggressive intents prefer the largest legal raise / ALL_IN before
        CHECK_CALL so aggression is never downgraded to a passive call (C3/H1).
        """
        if not legal_actions:
            raise ValueError("map_to_rlcard_action: legal_actions is empty")
        legal_ints = [_coerce_int(a) for a in legal_actions]
        mapped = PROJECT_TO_RLCARD.get(project_action, 1)
        if mapped in legal_ints:
            return mapped
        order = (
            _AGGRESSIVE_FALLBACK_ORDER
            if intent_class(project_action) == "aggressive"
            else _PASSIVE_FALLBACK_ORDER
        )
        for fallback in order:
            if fallback in legal_ints:
                return fallback
        return legal_ints[0]

    def execute_action(self, project_action: str) -> dict:
        """Drive the env for ``project_action`` with real bet-size fidelity.

        Aggressive verbs apply an intended raise-TO via the sized env when legal,
        else fall through the aggressive enum path. Returns intent_class/diverged.
        """
        cls = intent_class(project_action)
        legal_ints = self.current_legal_actions()
        if not legal_ints:
            raise ValueError("execute_action: no legal actions")

        target = self._intended_raise_to_chips(project_action)
        diverged = False
        executed_int: int | None = None
        executed_chips: int | None = None

        sized_raise_legal = any(
            a in legal_ints for a in (Action.RAISE_HALF_POT.value, Action.RAISE_POT.value)
        )
        if target is not None and sized_raise_legal:
            # Apply the sized raise-TO. The sized env clamps to all-in internally.
            rnd = self._env.game.round
            all_in_to = rnd.max_raise_to(self._env.game.players)
            executed_chips = min(target, all_in_to)
            # Divergence = realized chips differ from the verb's NOMINAL (pre-clamp)
            # target beyond round()'s <=0.5 error (`target` is already clamped).
            nominal = self._nominal_raise_to_chips(project_action)
            if nominal is not None and abs(executed_chips - nominal) > 0.5:
                diverged = True
            self._env.step_raise_to(executed_chips)
        else:
            # No sized raise achievable (or passive verb): enum-action path.
            executed_int = self.map_to_rlcard_action(project_action, legal_ints)
            faithful_all_in = project_action == "allin" and executed_int == Action.ALL_IN.value
            # Aggressive intent that lands on a non-raise enum is a downgrade.
            if cls == "aggressive" and executed_int in (
                Action.FOLD.value,
                Action.CHECK_CALL.value,
            ):
                diverged = True
            elif target is not None and not faithful_all_in:
                # Wanted a sized raise but only ALL_IN / enum raise available. A
                # shove verb that executes ALL_IN got exactly its target, not a divergence.
                diverged = True
            self._env.step(executed_int)

        if diverged:
            self._divergence[cls] += 1
        return {
            "intent_class": cls,
            "diverged": diverged,
            "executed_action": executed_int,
            "executed_chips": executed_chips,
        }

    def _live_chip_state(self) -> dict:
        """Live RLCard chip values for the current actor (pot from in_chips since
        dealer.pot is only refreshed inside get_state)."""
        game = self._env.game
        rnd = game.round
        gp = game.game_pointer
        max_raised = max(rnd.raised)
        # A true raise exists only once a commitment exceeds the BB (init_raise_amount).
        # Preflop first-in, raised holds the blinds (SB<BB); their gap is NOT a raise
        # increment — the min legal raise is the BB, so default to init_raise_amount.
        if max_raised > rnd.init_raise_amount:
            sorted_raised = sorted(set(rnd.raised))
            last_raise_size = sorted_raised[-1] - sorted_raised[-2]
        else:
            last_raise_size = rnd.init_raise_amount
        return {
            "pot": int(sum(p.in_chips for p in game.players)),
            "max_raised": int(max_raised),
            "my_raised": int(rnd.raised[gp]),
            "my_remained": int(game.players[gp].remained_chips),
            "bb_chips": int(game.big_blind),
            "init_raise_amount": int(rnd.init_raise_amount),
            "last_raise_size": int(last_raise_size),
        }

    def _intended_raise_to_chips(self, project_action: str) -> int | None:
        """Intended (clamped) raise-TO chips for ``project_action`` from live state."""
        if intent_class(project_action) == "passive":
            return None
        return intended_raise_to_chips(project_action, **self._live_chip_state())

    def _nominal_raise_to_chips(self, project_action: str) -> float | None:
        """The verb's nominal (pre-clamp) raise-TO chips, for divergence checks (M3)."""
        s = self._live_chip_state()
        return nominal_raise_to_chips(
            project_action,
            pot=s["pot"],
            max_raised=s["max_raised"],
            my_raised=s["my_raised"],
            bb_chips=s["bb_chips"],
        )

    def has_more(self) -> bool:
        """True while the current hand is in progress."""
        try:
            return not self._env.is_over()
        except Exception:
            return False

    def current_legal_actions(self) -> list[int]:
        """Return the legal RLCard action ints for the current decision point."""
        try:
            player_id = self._env.get_player_id()
            state = self._env.get_state(player_id)
        except Exception:
            return []
        raw = state.get("raw_obs", {})
        legal = raw.get("legal_actions", [])
        if isinstance(legal, dict):
            return [_coerce_int(k) for k in legal]
        return [_coerce_int(a) for a in legal]

    def next_game_state(self) -> GameState:
        """Build a GameState from the current env raw_obs.

        Must be called AFTER ``env.reset()`` or ``env.step()``; reads the most
        recent per-player state. The harness is responsible for calling reset/step
        at the right time.

        action_sequence is returned as an empty tuple in v1 — RLCard's per-street
        action history is not trivially reconstructed without additional bookkeeping.
        Plan 04's harness will track this externally when needed.
        """
        player_id = self._env.get_player_id()
        state = self._env.get_state(player_id)
        raw_obs = state.get("raw_obs", {})
        self._last_state = state
        self._last_player_id = player_id

        dealer_id = self._env.game.dealer_id

        if len(self._env.action_recorder) < self._street_start_idx:
            self._prev_stage = None
            self._street_start_idx = 0
            self._street_boundaries = []
            self._street_aggressor_cache = {}

        stage = _normalize_stage(raw_obs.get("stage", "preflop"))

        if stage != self._prev_stage:
            if self._prev_stage is not None:
                prior_slice = self._env.action_recorder[self._street_start_idx :]
                self._street_aggressor_cache[self._prev_stage] = aggressor_for_prior_street(
                    prior_slice,
                    dealer_id,
                    lambda pid, did: _hero_position(pid, did, self._env.num_players),
                    _ACTION_TO_VERB,
                )
                self._street_boundaries.append(len(self._env.action_recorder))
            self._street_start_idx = len(self._env.action_recorder)
            self._prev_stage = stage

        prior_idx = _STREETS_ORDER.index(stage) - 1
        prior_street_aggressor = (
            self._street_aggressor_cache.get(_STREETS_ORDER[prior_idx]) if prior_idx >= 0 else None
        )

        hand = raw_obs.get("hand", []) or []
        hole_list = [_normalize_rlcard_card(c) for c in hand[:2]]
        # Defensive — empty hand should never happen post-reset; fall back to a placeholder.
        hole = ("As", "Kh") if len(hole_list) != 2 else (hole_list[0], hole_list[1])

        public = raw_obs.get("public_cards", []) or []
        board = tuple(_normalize_rlcard_card(c) for c in public)

        n_players = self._env.num_players
        all_chips = list(raw_obs.get("all_chips", [0] * n_players))
        pot = int(raw_obs.get("pot", sum(all_chips)))
        max_chips_in_pot = max(all_chips) if all_chips else 0

        # BB = 2 chips in default RLCard NLHE (1 SB + 1 BB).
        bb_chips = 2.0
        pot_bb = pot / bb_chips
        # Effective stack proxy: min remaining stack across players STILL IN THE
        # HAND. A folded player keeps its remaining chips, so including it (e.g. a
        # folded short stack) understates SPR.
        active_stacks = [
            p.remained_chips for p in self._env.game.players if p.status in _IN_HAND and p.remained_chips > 0
        ]
        eff_stack_bb = (min(active_stacks) if active_stacks else 0) / bb_chips
        my_in_pot = int(all_chips[player_id]) if player_id < len(all_chips) else 0
        facing_bb = max(0.0, (max_chips_in_pot - my_in_pot) / bb_chips)
        bet_size_bb = my_in_pot / bb_chips

        hero_position = _hero_position(player_id, dealer_id, n_players)

        # Active = players not yet folded. stakes[i] is REMAINING stack (a folded
        # player keeps it), so it cannot be used to count who is still in the hand.
        active = sum(1 for p in self._env.game.players if p.status in _IN_HAND)
        opponents = max(0, active - 1)

        action_sequence = _action_tokens(
            self._env.action_recorder, dealer_id, self._street_boundaries, n_players
        )

        # RLCard raw_obs chips are numpy scalars; cast to plain float at the
        # boundary so the GameState is msgspec-encodable for felt capture.
        return GameState(
            street=stage,
            hero_position=hero_position,
            hero_hole_cards=hole,
            board_cards=board,
            pot_size_bb=float(pot_bb),
            effective_stack_bb=float(eff_stack_bb),
            hero_facing_bet_bb=float(facing_bb),
            hero_bet_size_bb=float(bet_size_bb),
            action_sequence=action_sequence,
            opponents_remaining=opponents,
            prior_street_aggressor=prior_street_aggressor,
        )
