"""Phase 9 / ChenBot — Chen formula preflop + postflop hand class (unit)."""

from __future__ import annotations

from src.eval.baselines.chen_bot import ChenBot, chen_score, postflop_hand_class

from src.eval.strategy import CHECK_CALL, FOLD, HALF_POT, POT


def test_chen_formula_scores():
    """Chen formula scores known hands correctly (ordering gate)."""
    aa = chen_score(("As", "Ah"))
    aks = chen_score(("As", "Kh"))
    trash = chen_score(("7s", "2c"))

    # AA must score highest, AKs in the middle, 72o in the trash region
    assert aa >= 20, f"AA expected >= 20, got {aa}"
    assert trash < 0, f"72o expected < 0, got {trash}"
    assert aa > aks > trash, f"ordering failed: AA={aa}, AKs={aks}, 72o={trash}"


def test_postflop_hand_class():
    """ChenBot postflop uses treys to classify made-hand tier."""
    hole = ["Ah", "Kh"]
    board = ["Qh", "Jh", "Th"]
    hand_class = postflop_hand_class(hole, board)
    assert hand_class in ("strong", "medium", "weak", "draw")


def test_chen_bot_preflop_raises_premium():
    """ChenBot raises with premium hands preflop."""
    bot = ChenBot(seed=42)
    state = {
        "hand": ["As", "Ah"],
        "public_cards": [],
        "legal_actions": [FOLD, CHECK_CALL, HALF_POT, POT],
    }
    action = bot.decide(state)
    assert action in (HALF_POT, POT), f"Expected raise with AA, got {action}"


def test_chen_bot_preflop_folds_trash():
    """ChenBot folds trash hands preflop when fold is legal."""
    bot = ChenBot(seed=42)
    state = {
        "hand": ["7s", "2c"],
        "public_cards": [],
        "legal_actions": [FOLD, CHECK_CALL, HALF_POT, POT],
    }
    action = bot.decide(state)
    assert action == FOLD, f"Expected fold with 72o, got {action}"


def test_chen_bot_postflop_strong_hand_does_not_fold():
    """ChenBot does NOT fold a set on a dry board postflop."""
    bot = ChenBot(seed=42)
    state = {
        "hand": ["Ah", "As"],
        "public_cards": ["Ad", "7c", "2h"],
        "legal_actions": [FOLD, CHECK_CALL, HALF_POT, POT],
    }
    action = bot.decide(state)
    assert action != FOLD, f"Expected non-fold with set of aces, got {action}"


def test_chen_bot_postflop_air_folds():
    """ChenBot folds air on a paired board with no outs."""
    bot = ChenBot(seed=42)
    state = {
        "hand": ["2s", "3h"],
        "public_cards": ["Kd", "Kh", "As", "Jc", "Td"],
        "legal_actions": [FOLD, CHECK_CALL],
    }
    action = bot.decide(state)
    assert action == FOLD, f"Expected fold with air facing bet, got {action}"


def test_chen_bot_returns_legal_action():
    """ChenBot always returns a legal action even with restricted legal_actions."""
    bot = ChenBot(seed=42)
    state = {
        "hand": ["As", "Kh"],
        "public_cards": [],
        "legal_actions": [CHECK_CALL],
    }
    action = bot.decide(state)
    assert action == CHECK_CALL


def test_chen_bot_deterministic():
    """ChenBot is deterministic given same seed."""
    state = {
        "hand": ["As", "Kh"],
        "public_cards": ["Qd", "Jc", "Ts"],
        "legal_actions": [FOLD, CHECK_CALL, HALF_POT, POT],
    }
    bot1 = ChenBot(seed=7)
    bot2 = ChenBot(seed=7)
    assert bot1.decide(state) == bot2.decide(state)


def test_chen_bot_name():
    """ChenBot has the correct registry name."""
    assert ChenBot.name == "chen-bot"
