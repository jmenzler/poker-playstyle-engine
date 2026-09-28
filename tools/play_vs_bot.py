"""Interactive CLI: human vs KNNDecisionEngine bot.

Plays heads-up no-limit hold'em where bot uses src.decision_engine and human
types actions at the prompt. Prints bot's action + facing situation each turn.

Usage:
    set -a && source .env && set +a
    uv run python tools/play_vs_bot.py --n-hands 5
    uv run python tools/play_vs_bot.py --n-hands 5 --human-seat 0 --session-seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision_engine.engine import engine_from_env
from src.sim.adapter import SimAdapter

# RLCard 5-action vocab labels (rlcard.games.nolimitholdem.round.Action).
RLCARD_LABELS: dict[int, str] = {
    0: "FOLD",
    1: "CHECK_CALL",
    2: "RAISE_HALF_POT",
    3: "RAISE_POT",
    4: "ALL_IN",
}

# Human input shortcuts.
HUMAN_SHORTCUTS: dict[str, int] = {
    "f": 0,
    "fold": 0,
    "c": 1,
    "check": 1,
    "call": 1,
    "h": 2,
    "half": 2,
    "p": 3,
    "pot": 3,
    "a": 4,
    "allin": 4,
    "all-in": 4,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play heads-up vs the kNN bot.")
    p.add_argument("--n-hands", type=int, default=5)
    p.add_argument("--session-seed", type=int, default=1234)
    p.add_argument(
        "--human-seat", type=int, default=0, help="Player index controlled by human (0 or 1 in heads-up)."
    )
    p.add_argument("--engine-k", type=int, default=10)
    p.add_argument("--no-warmup", action="store_true")
    return p.parse_args()


def render_hand(state: dict, hero_id: int) -> str:
    raw = state.get("raw_obs", {})
    hand = raw.get("hand", [])
    board = raw.get("public_cards", [])
    pot = raw.get("pot", 0)
    all_chips = raw.get("all_chips", [])
    stakes = raw.get("stakes", [])
    legal = raw.get("legal_actions", [])
    legal_ints = []
    for a in legal:
        legal_ints.append(int(a.value) if hasattr(a, "value") else int(a))
    legal_labels = [f"{i}:{RLCARD_LABELS.get(i, '?')}" for i in legal_ints]
    return (
        f"  hero_id={hero_id} hand={hand} board={board}\n"
        f"  pot={pot} all_chips={all_chips} stakes={stakes}\n"
        f"  legal={legal_labels}"
    )


def human_turn(state: dict, hero_id: int) -> int:
    print("\n[YOUR TURN]")
    print(render_hand(state, hero_id))
    raw = state.get("raw_obs", {})
    legal = raw.get("legal_actions", [])
    legal_ints = [int(a.value) if hasattr(a, "value") else int(a) for a in legal]
    while True:
        try:
            entry = input("action> ").strip().lower()
        except EOFError:
            sys.exit(0)
        if entry in HUMAN_SHORTCUTS:
            chosen = HUMAN_SHORTCUTS[entry]
        elif entry.isdigit():
            chosen = int(entry)
        else:
            print(f"  unknown action {entry!r}; try f/c/h/p/a or int")
            continue
        if chosen not in legal_ints:
            print(f"  {chosen} not legal; choose from {legal_ints}")
            continue
        return chosen


def bot_turn(adapter: SimAdapter, engine, hero_id: int) -> tuple[int, str, str]:
    """Returns (rlcard_int, project_action, cluster_key)."""
    state = adapter._env.get_state(hero_id)
    print("\n[BOT TURN]")
    print(render_hand(state, hero_id))
    gs = adapter.next_game_state()
    project_action, _flagged_sparse, _max_dist, enc = engine.decide_with_encoding(gs)
    legal_ints = adapter.current_legal_actions()
    rlcard_int = adapter.map_to_rlcard_action(project_action, legal_ints)
    cluster_key = "|".join(f"{k}={v}" for k, v in sorted(enc.hard_filter.items()))
    print(f"  bot.project_action={project_action!r}")
    print(f"  bot.rlcard={rlcard_int}:{RLCARD_LABELS.get(rlcard_int, '?')}")
    print(f"  bot.cluster_key={cluster_key}")
    return rlcard_int, project_action, cluster_key


def play_one_hand(adapter: SimAdapter, engine, human_seat: int, hand_seed: int, hand_idx: int) -> None:
    print(f"\n{'=' * 60}\nHAND {hand_idx + 1}  (seed={hand_seed})\n{'=' * 60}")
    adapter._env.seed(hand_seed)
    adapter._env.reset()
    while not adapter._env.is_over():
        pid = adapter._env.get_player_id()
        if pid == human_seat:
            action = human_turn(adapter._env.get_state(pid), pid)
        else:
            action, _, _ = bot_turn(adapter, engine, pid)
        adapter._env.step(action)
    payoffs = adapter._env.get_payoffs()
    print(f"\n[HAND OVER] payoffs={payoffs}  (you={payoffs[human_seat]})")


def main() -> int:
    args = parse_args()
    print(f"Loading bot (engine_k={args.engine_k}, warmup={not args.no_warmup})...")
    engine = engine_from_env(k=args.engine_k, rng_seed=args.session_seed, warmup=not args.no_warmup)
    adapter = SimAdapter(config={"game_num_players": 2, "chips_for_each": 100})
    print(f"\nHEADS-UP: human=seat{args.human_seat}, bot=seat{1 - args.human_seat}")
    print("Shortcuts: f=fold c=check/call h=half-pot-raise p=pot-raise a=allin  (or int)")
    session_rng = np.random.default_rng(args.session_seed)
    human_total = 0.0
    for i in range(args.n_hands):
        hand_seed = int(session_rng.integers(0, 2**31 - 1))
        play_one_hand(adapter, engine, args.human_seat, hand_seed, i)
        payoffs = adapter._env.get_payoffs()
        human_total += float(payoffs[args.human_seat])
    print(f"\n{'=' * 60}\nSESSION OVER\n{'=' * 60}")
    print(f"Total human payoff over {args.n_hands} hands: {human_total:.2f} chips ({human_total / 2:.2f} bb)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
