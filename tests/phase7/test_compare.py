"""Head-to-head corpus-version A/B (compare) + CLI shim.

compare pits the v_new engine against the v_old engine (paired seed); factory +
runner are injectable so the unit test uses stubs (no Milvus/TSDB).
"""

from __future__ import annotations

import argparse
from unittest.mock import patch


def _fake_match(*, bb_per_100, ci):
    """Minimal stand-in for MatchResult exposing the fields compare reads."""

    class _R:
        def __init__(self):
            self.bb_per_100 = bb_per_100
            self.ci_low, self.ci_high = ci

    return _R()


def test_compare_pits_new_against_old_head_to_head():
    from src.study.compare import compare

    seen: list = []
    engines = {1: object(), 2: object()}

    def _factory(version):
        return engines[version]

    def _h2h(engine_new, engine_old, *, hands, seed):
        seen.append((engine_new, engine_old, hands, seed))
        return _fake_match(bb_per_100=7.0, ci=(2.0, 12.0))

    result = compare(1, 2, hands=500, seed=99, _engine_factory=_factory, _head_to_head=_h2h)

    assert result["v_old"] == 1
    assert result["v_new"] == 2
    assert result["n_hands"] == 500
    assert result["mode"] == "head_to_head"
    # bb_per_100 is v_new's edge over v_old, straight from the one match.
    assert result["bb_per_100"] == 7.0
    assert result["ci"] == (2.0, 12.0)
    # Exactly ONE head-to-head match: hero = v_new engine, villain = v_old engine.
    assert len(seen) == 1
    en, eo, hands, seed = seen[0]
    assert en is engines[2] and eo is engines[1]
    assert hands == 500 and seed == 99


def test_compare_resolves_both_versions_via_engine_factory():
    from src.study.compare import compare

    requested: list = []

    def _factory(version):
        requested.append(version)
        return object()

    def _h2h(engine_new, engine_old, *, hands, seed):
        return _fake_match(bb_per_100=0.0, ci=(0.0, 0.0))

    compare(3, 8, _engine_factory=_factory, _head_to_head=_h2h)
    assert requested == [3, 8]


def test_compare_default_engine_factory_uses_engine_from_env():
    """Without an injected factory, compare builds pinned engines via env."""
    from src.study import compare as compare_mod

    captured: list = []

    def _fake_env(*, as_of_version):
        captured.append(as_of_version)
        return object()

    def _h2h(engine_new, engine_old, *, hands, seed):
        return _fake_match(bb_per_100=1.0, ci=(0.0, 2.0))

    with patch("src.decision_engine.engine.engine_from_env", side_effect=_fake_env):
        compare_mod.compare(4, 5, _head_to_head=_h2h)
    assert captured == [4, 5]


# CLI shim


def _args(*, old=1, new=2, hands=2000, seed=42, format="text"):
    return argparse.Namespace(old=old, new=new, hands=hands, seed=seed, format=format)


def test_cli_compare_forwards_args_to_compare():
    captured = {}

    def _capture(v_old, v_new, **kw):
        captured["v_old"] = v_old
        captured["v_new"] = v_new
        captured.update(kw)
        return {
            "v_old": v_old,
            "v_new": v_new,
            "bb_per_100": 1.0,
            "ci": [0.0, 2.0],
            "n_hands": kw["hands"],
            "mode": "head_to_head",
        }

    with patch("src.study.compare.compare", side_effect=_capture):
        from src.cli.compare import run

        rc = run(_args(old=1, new=2, hands=500, seed=7, format="json"))

    assert rc == 0
    assert captured["v_old"] == 1
    assert captured["v_new"] == 2
    assert captured["hands"] == 500
    assert captured["seed"] == 7


def test_cli_compare_text_format_renders_result(capsys):
    delta = {
        "v_old": 1,
        "v_new": 2,
        "bb_per_100": 4.0,
        "ci": [3.0, 5.0],
        "n_hands": 2000,
        "mode": "head_to_head",
    }
    with patch("src.study.compare.compare", return_value=delta):
        from src.cli.compare import run

        rc = run(_args(format="text"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "4.0" in out
    assert "v1" in out and "v2" in out


def test_cli_compare_unexpected_exception_returns_exit_2():
    with patch("src.study.compare.compare", side_effect=RuntimeError("boom")):
        from src.cli.compare import run

        rc = run(_args())
    assert rc == 2
