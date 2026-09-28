"""BOOT-06 — INFO-level summary of straddle_skip + rit_skip counts — activated by Plan 04.

Confirms tools/extract_decisions.py emits the INFO summary unconditionally,
even when both skip counts are 0 (zero-case test).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _make_normal_hand(hand_id: str) -> dict:
    """Minimal HM events hand record that passes hand_excluded (no straddle, HU, or RIT).

    Uses table_size=3 (not HU), no post_straddle event in preflop.
    Enough structure for extract_decisions.py to parse without crashing.
    """
    return {
        "hand_id": hand_id,
        "table_size": 3,
        "pot_type": "srp",
        "site": "PokerStars",
        "bb_cents": 10,
        "sb_cents": 5,
        "btn_seat": 1,
        "hero_alias": "hero",
        "seats": {"BTN": "hero", "SB": "villain1", "BB": "villain2"},
        "board": [],
        "preflop": [
            {
                "actor_pos": "SB",
                "action": "post_sb",
                "amount_cents": 5,
                "stack_after_cents": 995,
                "pot_after_cents": 5,
                "is_hero": False,
            },
            {
                "actor_pos": "BB",
                "action": "post_bb",
                "amount_cents": 10,
                "stack_after_cents": 990,
                "pot_after_cents": 15,
                "is_hero": False,
            },
            {
                "actor_pos": "BTN",
                "action": "fold",
                "amount_cents": 0,
                "stack_after_cents": 1000,
                "pot_after_cents": 15,
                "is_hero": True,
            },
        ],
        "flop": [],
        "turn": [],
        "river": [],
    }


def _make_straddle_hand(hand_id: str) -> dict:
    """Minimal hand with a straddle — hand_excluded returns True, reason='straddle'."""
    return {
        "hand_id": hand_id,
        "table_size": 4,
        "pot_type": "srp",
        "site": "PokerStars",
        "bb_cents": 10,
        "sb_cents": 5,
        "btn_seat": 1,
        "hero_alias": "hero",
        "seats": {"BTN": "hero", "SB": "p1", "BB": "p2", "UTG": "p3"},
        "board": [],
        "preflop": [
            {
                "actor_pos": "SB",
                "action": "post_sb",
                "amount_cents": 5,
                "stack_after_cents": 995,
                "pot_after_cents": 5,
                "is_hero": False,
            },
            {
                "actor_pos": "BB",
                "action": "post_bb",
                "amount_cents": 10,
                "stack_after_cents": 990,
                "pot_after_cents": 15,
                "is_hero": False,
            },
            {
                "actor_pos": "UTG",
                "action": "post_straddle",
                "amount_cents": 20,
                "stack_after_cents": 980,
                "pot_after_cents": 35,
                "is_hero": False,
            },
        ],
        "flop": [],
        "turn": [],
        "river": [],
    }


def _write_events_jsonl(path: Path, hands: list[dict]) -> None:
    """Write a list of hand dicts to a JSONL file."""
    with open(path, "w") as f:
        for hand in hands:
            f.write(json.dumps(hand) + "\n")


def test_skip_count_logging(capfd, caplog, tmp_path: Path) -> None:
    """BOOT-06: extract_decisions.py emits INFO summary with straddle_skip + rit_skip counts.

    Uses a fixture with >=1 straddle hand and >=1 normal hand (rit_skip = 0 per known gap).
    Asserts that captured output contains both straddle_skip and rit_skip tokens with values.
    Checks both capfd (raw stdout/stderr) and caplog (Python logging capture) for maximum
    coverage regardless of how structlog routes output in test environment.
    """
    events_path = tmp_path / "hm_events.jsonl"
    out_path = tmp_path / "hm_decisions.jsonl"

    # 2 straddle hands + 1 normal hand
    hands = [
        _make_straddle_hand("hand_straddle_1"),
        _make_straddle_hand("hand_straddle_2"),
        _make_normal_hand("hand_normal_1"),
    ]
    _write_events_jsonl(events_path, hands)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from tools.extract_decisions import main as extract_main

    with caplog.at_level("INFO"):
        rc = extract_main(["--events", str(events_path), "--out", str(out_path)])
    assert rc == 0

    captured = capfd.readouterr()
    # Combine: capfd raw output + caplog text (structlog may route through stdlib logging)
    combined = captured.out + captured.err + caplog.text

    assert re.search(r"straddle_skip[=:\s\"]+\d+", combined), (
        f"Expected straddle_skip count in log output; got: {combined[-800:]}"
    )
    assert re.search(r"rit_skip[=:\s\"]+\d+", combined), (
        f"Expected rit_skip count in log output; got: {combined[-800:]}"
    )


def test_skip_count_logging_zero_case(capfd, caplog, tmp_path: Path) -> None:
    """BOOT-06 zero-case: extractor still emits the INFO summary even if both counts are 0.

    Plan 04 MUST NOT conditionally suppress the summary when no straddles or RIT hands
    are encountered — observability requires a deterministic end-of-run log line.
    """
    events_path = tmp_path / "hm_events.jsonl"
    out_path = tmp_path / "hm_decisions.jsonl"

    # Only normal hands — no straddle, no RIT
    hands = [
        _make_normal_hand("hand_normal_1"),
        _make_normal_hand("hand_normal_2"),
    ]
    _write_events_jsonl(events_path, hands)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from tools.extract_decisions import main as extract_main

    with caplog.at_level("INFO"):
        rc = extract_main(["--events", str(events_path), "--out", str(out_path)])
    assert rc == 0

    captured = capfd.readouterr()
    # Combine: capfd raw output + caplog text (structlog may route through stdlib logging)
    combined = captured.out + captured.err + caplog.text

    # Both counts present even at zero — unconditional summary contract
    assert "straddle_skip" in combined, (
        f"BOOT-06: summary must fire even when count=0. Output: {combined[-800:]}"
    )
    assert "rit_skip" in combined, f"BOOT-06: summary must fire even when count=0. Output: {combined[-800:]}"
