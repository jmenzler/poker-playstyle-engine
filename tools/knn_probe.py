"""Ad-hoc kNN retrieval probe: hand-built GameStates -> encode -> Milvus search -> print neighbors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.canonicalizer.encoder import Canonicalizer
from src.decision_engine.engine import _build_filter, _milvus_client_from_env
from src.protocols.game_state import GameState
from tools.upsert_milvus import build_preflop_weight_vector, normalize
from tools.zscore_fit import load_manifest

_OUT = [
    "decision_id",
    "hero_action_type",
    "pot_type",
    "hero_pos_rel",
    "n_players_active",
    "confidence",
    "gto_score",
]


SPOTS: list[tuple[str, GameState]] = [
    (
        "AKs BTN, facing UTG open (srp, IP) — premium, should call/3bet",
        GameState(
            street="preflop",
            hero_position="BTN",
            hero_hole_cards=("As", "Ks"),
            board_cards=(),
            pot_size_bb=4.5,
            effective_stack_bb=100.0,
            hero_facing_bet_bb=3.0,
            hero_bet_size_bb=0.0,
            action_sequence=("UTG:open_3", "MP:fold", "CO:fold"),
            opponents_remaining=1,
            prior_street_aggressor="UTG",
        ),
    ),
    (
        "72o UTG open decision (unopened) — trash, should fold",
        GameState(
            street="preflop",
            hero_position="UTG",
            hero_hole_cards=("7d", "2c"),
            board_cards=(),
            pot_size_bb=1.5,
            effective_stack_bb=100.0,
            hero_facing_bet_bb=0.0,
            hero_bet_size_bb=0.0,
            action_sequence=(),
            opponents_remaining=5,
            prior_street_aggressor=None,
        ),
    ),
    (
        "QQ facing 3bet, hero was CO opener (3bet pot, OOP) — should 4bet/call",
        GameState(
            street="preflop",
            hero_position="CO",
            hero_hole_cards=("Qh", "Qd"),
            board_cards=(),
            pot_size_bb=13.0,
            effective_stack_bb=100.0,
            hero_facing_bet_bb=9.0,
            hero_bet_size_bb=3.0,
            action_sequence=("CO:open_3", "BTN:3b_9"),
            opponents_remaining=1,
            prior_street_aggressor="BTN",
        ),
    ),
    (
        "AJs SB facing BTN open (srp, OOP) — flat/3bet spot",
        GameState(
            street="preflop",
            hero_position="SB",
            hero_hole_cards=("Ah", "Jh"),
            board_cards=(),
            pot_size_bb=4.5,
            effective_stack_bb=100.0,
            hero_facing_bet_bb=3.0,
            hero_bet_size_bb=0.5,
            action_sequence=("BTN:open_3",),
            opponents_remaining=1,
            prior_street_aggressor="BTN",
        ),
    ),
    (
        "55 BTN, unopened (srp, IP) — open or fold",
        GameState(
            street="preflop",
            hero_position="BTN",
            hero_hole_cards=("5s", "5c"),
            board_cards=(),
            pot_size_bb=1.5,
            effective_stack_bb=100.0,
            hero_facing_bet_bb=0.0,
            hero_bet_size_bb=0.0,
            action_sequence=("UTG:fold", "MP:fold", "CO:fold"),
            opponents_remaining=4,
            prior_street_aggressor=None,
        ),
    ),
]


def main() -> None:
    canon = Canonicalizer.default()
    client = _milvus_client_from_env()
    p_mean, p_std = load_manifest(Path("tools/zscore_preflop.json"))
    weights = build_preflop_weight_vector()

    for desc, gs in SPOTS:
        enc = canon.encode(gs)
        vec = normalize(np.asarray(enc.embedding, dtype=np.float64), p_mean, p_std, weights)
        sf = {"street": "preflop", "spr_x100": -1}
        filt = _build_filter(enc.hard_filter, include_active=True, spot_features=sf)
        res = client.search(
            collection_name="preflop_decisions",
            data=[vec.tolist()],
            limit=8,
            filter=filt,
            search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            output_fields=_OUT,
        )
        print(f"\n=== {desc}")
        print(f"    filter: {filt}")
        row = res[0] if res else []
        if not row:
            print("    NO NEIGHBORS (fallback)")
            continue
        from collections import Counter

        acts = Counter(h["entity"]["hero_action_type"] for h in row)
        print(
            f"    n={len(row)} dist[{row[0]['distance']:.3f}..{row[-1]['distance']:.3f}] actions={dict(acts)}"
        )
        for h in row[:6]:
            e = h["entity"]
            print(
                f"      d={h['distance']:.3f} {e['hero_action_type']:6} "
                f"{e['pot_type']:5}|{e['hero_pos_rel']:3}|n{e['n_players_active']} "
                f"conf={e['confidence']:.2f} gto={e['gto_score']:.2f} {e['decision_id']}"
            )


if __name__ == "__main__":
    main()
