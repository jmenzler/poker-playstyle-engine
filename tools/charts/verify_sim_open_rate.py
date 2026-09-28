"""In-process sim verification: preflop open/walk rate with charts ON vs OFF.
Drives the real SimAdapter (RLCard) with a mock Milvus (no DB needed): OFF falls
back to the kNN blend, ON uses the chart. Reports preflop open/walk rate + the
postflop mix (must be unchanged). Run: python -m tools.charts.verify_sim_open_rate
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.decision_engine.engine import KNNDecisionEngine
from src.sim import SimAdapter

_OPEN_ACTIONS = {"open_2_2bb", "open_3bb", "3bet_3x", "3bet_4x", "4bet_2_5x", "allin"}
_MANIFEST_DIR = Path("tools")


def _mock_milvus() -> MagicMock:
    """Mock Milvus whose neighbors blend to a fold-dominated preflop dist (the
    documented over-fold artifact), so OFF mode reproduces the kNN baseline."""
    client = MagicMock()
    client.search.return_value = [
        [
            {"distance": 0.05, "entity": {"hero_action_type": "fold", "confidence": 1.0, "gto_score": 1.0}},
            {"distance": 0.10, "entity": {"hero_action_type": "fold", "confidence": 1.0, "gto_score": 1.0}},
            {"distance": 0.40, "entity": {"hero_action_type": "raise", "confidence": 1.0, "gto_score": 1.0}},
            {"distance": 0.50, "entity": {"hero_action_type": "call", "confidence": 1.0, "gto_score": 1.0}},
        ]
    ]
    return client


def _run(n_hands: int, seed: int, use_charts: bool) -> dict[str, object]:
    engine = KNNDecisionEngine(
        _mock_milvus(),
        _MANIFEST_DIR / "zscore_preflop.json",
        _MANIFEST_DIR / "zscore_postflop.json",
        rng_seed=seed,
        use_preflop_charts=use_charts,
    )
    adapter = SimAdapter()
    pre: Counter[str] = Counter()
    post: Counter[str] = Counter()
    session_rng = np.random.default_rng(seed)
    for _ in range(n_hands):
        adapter._env.seed(int(session_rng.integers(0, 2**31)))
        adapter._env.reset()
        guard = 0
        while adapter.has_more() and guard < 200:
            guard += 1
            gs = adapter.next_game_state()
            try:
                action, *_ = engine.decide_with_encoding(gs)
            except Exception:
                action = "fold"
            if gs.street == "preflop":
                pre[action] += 1
            else:
                post[action] += 1
            legal = adapter.current_legal_actions()
            adapter._env.step(adapter.map_to_rlcard_action(action, legal))
    pre_total = sum(pre.values()) or 1
    opens = sum(pre[a] for a in _OPEN_ACTIONS)
    walks = pre.get("check", 0) + pre.get("fold", 0)
    return {
        "preflop_decisions": sum(pre.values()),
        "open_rate": round(opens / pre_total, 3),
        "walk_fold_rate": round(walks / pre_total, 3),
        "preflop_mix": dict(pre.most_common()),
        "postflop_mix": dict(post.most_common()),
        "chart_hits": engine._preflop_chart.chart_hits if engine._preflop_chart else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-hands", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    off = _run(args.n_hands, args.seed, use_charts=False)
    on = _run(args.n_hands, args.seed, use_charts=True)
    print("=== charts OFF (kNN blend baseline) ===")
    print(f"  open_rate={off['open_rate']}  walk/fold_rate={off['walk_fold_rate']}")
    print(f"  preflop_mix={off['preflop_mix']}")
    print("=== charts ON (chart lookup) ===")
    print(
        f"  open_rate={on['open_rate']}  walk/fold_rate={on['walk_fold_rate']}  chart_hits={on['chart_hits']}"
    )
    print(f"  preflop_mix={on['preflop_mix']}")
    print("=== postflop mix (should be ~unchanged) ===")
    print(f"  OFF={off['postflop_mix']}")
    print(f"  ON ={on['postflop_mix']}")


if __name__ == "__main__":
    main()
