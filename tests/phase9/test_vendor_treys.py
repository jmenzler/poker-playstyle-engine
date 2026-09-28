"""Phase 9 / treys vendor — import and basic sanity (unit)."""

from __future__ import annotations


def test_import():
    from src.vendor.treys import Card, Evaluator

    assert Card is not None and Evaluator is not None


def test_royal_flush_beats_pair():
    from src.vendor.treys import Card, Evaluator

    ev = Evaluator()
    # treys rank: lower = stronger (1 = royal flush, 7462 = worst)
    royal = ev.evaluate(
        [Card.new("Ah"), Card.new("Kh"), Card.new("Qh")],
        [Card.new("Jh"), Card.new("Th")],
    )
    pair = ev.evaluate(
        [Card.new("Ah"), Card.new("Kd"), Card.new("2c")],
        [Card.new("As"), Card.new("7h")],
    )
    assert royal < pair


def test_card_format_matches_project():
    from src.vendor.treys import Card

    # project format = rank.upper() + suit.lower() (adapter.py _normalize_rlcard_card)
    assert isinstance(Card.new("Ah"), int)
    assert isinstance(Card.new("Kd"), int)


def test_no_dangerous_imports():
    import pathlib
    import re

    src = pathlib.Path("src/vendor/treys")
    # Simple substring tokens (not partial-identifier false positives)
    substring_forbidden = (
        "import socket",
        "import subprocess",
        "urllib",
        "requests",
        "import pickle",
        "os.environ",
    )
    # Regex tokens: must be standalone (not part of a longer identifier like five_card_eval)
    regex_forbidden = (
        r"(?<![A-Za-z_])eval\(",  # eval( not preceded by identifier char
        r"(?<![A-Za-z_])exec\(",  # exec( not preceded by identifier char
    )
    for f in src.glob("*.py"):
        text = f.read_text()
        for tok in substring_forbidden:
            assert tok not in text, f"{f.name} contains forbidden token {tok!r}"
        for pat in regex_forbidden:
            assert not re.search(pat, text), f"{f.name} contains forbidden pattern {pat!r}"
