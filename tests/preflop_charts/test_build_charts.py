"""Converter: build_preflop_charts emits a valid, GTO-shaped chart tree."""

from __future__ import annotations

import json

import pytest

from src.decision_engine.blending import CANONICAL_ACTIONS
from tools.charts.build_preflop_charts import _class_weight, _sized_raise_action, build


@pytest.fixture(scope="module")
def chart_tree(tmp_path_factory):
    out = tmp_path_factory.mktemp("charts")
    manifest = build(out)
    return out, manifest


def _nonfold_mass(chart: dict[str, dict[str, float]]) -> float:
    mass = sum(_class_weight(c) * sum(d.values()) for c, d in chart.items())
    return mass / 1326.0


def test_all_dists_use_canonical_vocab_and_sum_le_one(chart_tree):
    out, _ = chart_tree
    for f in out.glob("*/*.json"):
        chart = json.loads(f.read_text())
        for combo, dist in chart.items():
            assert sum(dist.values()) <= 1.0001, f"{f.name}:{combo} sum>1"
            for action in dist:
                assert action in CANONICAL_ACTIONS, f"{f.name}:{combo} bad action {action!r}"


def test_rfi_btn_open_mass_in_range(chart_tree):
    out, _ = chart_tree
    chart = json.loads((out / "RFI" / "BTN.json").read_text())
    mass = _nonfold_mass(chart)
    assert 0.35 <= mass <= 0.50, f"BTN open mass {mass} outside [0.35,0.50]"


def test_rfi_utg_open_mass_in_range(chart_tree):
    out, _ = chart_tree
    chart = json.loads((out / "RFI" / "UTG.json").read_text())
    mass = _nonfold_mass(chart)
    assert 0.13 <= mass <= 0.22, f"UTG open mass {mass} outside [0.13,0.22]"
    # the over-fold blend artifact was ~0.04 — chart must NOT reproduce it
    assert mass > 0.10, f"UTG mass {mass} looks like the ~4% blend artifact"


def test_rfi_uses_open_actions(chart_tree):
    out, _ = chart_tree
    chart = json.loads((out / "RFI" / "BTN.json").read_text())
    # AA opens for sure; the action must be a sized open, never 3bet/call
    assert "open_2_2bb" in chart["AA"] or "open_3bb" in chart["AA"]


def test_vs_rfi_has_3bet_and_call_mass(chart_tree):
    out, _ = chart_tree
    chart = json.loads((out / "vs_RFI" / "BTN_vs_CO.json").read_text())
    # BTN is IP vs CO -> 3bet_3x stem; medium pairs flat, premiums 3bet
    assert "3bet_3x" in chart["AA"]
    assert "call" in chart["44"], "44 should mostly flat-call vs an open"


def test_sized_raise_action_mapping():
    assert _sized_raise_action("RFI", "BTN", "IP") == "open_2_2bb"
    assert _sized_raise_action("RFI", "SB", "OOP") == "open_3bb"
    assert _sized_raise_action("vs_RFI", "BTN", "IP") == "3bet_3x"
    assert _sized_raise_action("vs_RFI", "SB", "OOP") == "3bet_4x"
    assert _sized_raise_action("vs_3bet", "CO", "OOP") == "4bet_2_5x"
    assert _sized_raise_action("vs_4bet", "UTG", "OOP") == "allin"


def test_manifest_provenance(chart_tree):
    _, manifest = chart_tree
    assert manifest["source"] == "inferred-from-HH"
    assert manifest["n_charts"] > 0
    assert "coverage" in manifest
