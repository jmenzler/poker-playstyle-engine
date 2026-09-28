"""End-to-end solve_harvest integration: one flop-rooted solve, navigate to the flop
and turn DPs of a line, assert ranges narrow + pot escalates. Binary-gated — skips
when the postflop-cli binary is absent (Mac); runs on the PC.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_BIN = Path(
    os.environ.get("POSTFLOP_CLI_BIN", str(Path.home() / "projects/poker-engine/target/release/postflop-cli"))
)
pytestmark = pytest.mark.skipif(
    not _BIN.exists(), reason=f"postflop-cli binary not found: {_BIN} — run on PC"
)


def _harvest(payload: dict) -> dict:
    proc = subprocess.run([str(_BIN)], input=json.dumps(payload), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[:500]
    return json.loads(proc.stdout)


def test_harvest_narrows_and_escalates() -> None:
    """Turn DP (after flop bet+call) has a tighter villain range + larger pot than the flop DP."""
    sizes = ["50%", "a"]
    payload = {
        "mode": "solve_harvest",
        "pot": 600,
        "effective_stack": 4000,
        "board": ["Ah", "7c", "2d"],
        "range_oop": "22+,AJs+,KQs,AQo+",
        "range_ip": "22+,ATs+,KJs+,AJo+,KQo",
        "target_exploitability_pct": 2.0,
        "max_iterations": 120,
        "bet_sizes_flop_oop": sizes,
        "bet_sizes_flop_ip": sizes,
        "bet_sizes_turn_oop": sizes,
        "bet_sizes_turn_ip": sizes,
        "bet_sizes_river_oop": sizes,
        "bet_sizes_river_ip": sizes,
        "nav_lines": [
            {"steps": [], "hero_player": 0},
            {
                "steps": [
                    {"seat": 0, "kind": "check"},
                    {"seat": 1, "kind": "bet", "frac": 0.5},
                    {"seat": 0, "kind": "call"},
                ],
                "hero_player": 0,
                "turn_card": "Ts",
            },
        ],
    }
    out = _harvest(payload)
    flop, turn = out["results"]

    assert flop["nav_ok"] and turn["nav_ok"]
    # Flop root = villain has not acted → full preflop range; turn = narrowed by the bet+call.
    assert turn["villain_live_combos"] < flop["villain_live_combos"]
    # Real node pot escalates (the pot_at_node snap reference, not the flop-start pot).
    assert turn["pot_at_node"] > flop["pot_at_node"]
    # Both seats' surviving ranges are emitted at the node.
    assert turn["oop_range"] and turn["ip_range"]
