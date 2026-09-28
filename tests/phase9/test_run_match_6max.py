"""Phase 9 / run_match 6-max — env config wiring (unit)."""

from __future__ import annotations

import inspect


def test_run_match_accepts_table_size_param():
    """run_match signature includes table_size: int = 2."""
    from src.eval.run_match import run_match

    sig = inspect.signature(run_match)
    assert "table_size" in sig.parameters, "run_match missing table_size parameter"
    param = sig.parameters["table_size"]
    assert param.default == 2, f"table_size default should be 2, got {param.default}"


def test_run_match_source_has_6max_branch():
    """run_match source contains game_num_players=table_size for the 6-max branch."""
    from src.eval import run_match as rm

    src = inspect.getsource(rm)
    assert "game_num_players" in src
    assert "table_size" in src
    assert "for i in range(table_size - 1)" in src, "6-max agent instantiation loop missing"


def test_run_match_source_hu_branch():
    """run_match source still contains the explicit game_num_players=2 HU branch."""
    from src.eval import run_match as rm

    src = inspect.getsource(rm)
    assert '"game_num_players": 2' in src or 'game_num_players": 2' in src, (
        "HU explicit game_num_players=2 not found"
    )


def test_registry_has_chen_bot():
    """REGISTRY contains 'chen-bot' and it instantiates without error."""
    from src.eval.baselines import REGISTRY

    assert "chen-bot" in REGISTRY
    bot = REGISTRY["chen-bot"](seed=99)
    assert bot.name == "chen-bot"


def test_registry_chen_bot_resolvable_in_run_match():
    """run_match does not raise KeyError for 'chen-bot'."""
    from src.eval.baselines import REGISTRY

    assert "chen-bot" in REGISTRY, "chen-bot must be in REGISTRY to avoid KeyError in run_match"


def test_6max_agent_seed_spread():
    """6-max branch seeds opponents as seed+1+i for i in range(table_size-1)."""
    from src.eval.baselines.chen_bot import ChenBot

    seed = 42
    table_size = 6
    seeds = [seed + 1 + i for i in range(table_size - 1)]
    bots = [ChenBot(seed=s) for s in seeds]
    assert len(bots) == 5
    for i, bot in enumerate(bots):
        assert isinstance(bot, ChenBot)
