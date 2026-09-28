from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


class _FakeCursor:
    """psycopg-faithful cursor for the empty-DB case: row-returning SELECTs
    (strategy_nodes, embedding) yield None, while aggregate SELECTs
    (COUNT/AVG/bool_or) yield a single-row tuple as real psycopg does."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, *args, **kwargs):
        s = sql.upper()
        if "COUNT(" in s:
            self._row = (0,)
        elif "BOOL_OR" in s:
            self._row = (None, None)
        elif "AVG(" in s:
            self._row = (None,)
        else:
            # row-returning lookup with no matching row
            self._row = None
        return None

    def fetchone(self):
        return self._row

    def fetchall(self):
        return []

    _row = None


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def close(self):
        return None


_ROOT = Path(__file__).resolve().parent.parent.parent
_EQUITY_TABLE_PRESENT = (_ROOT / "tools" / "equity_table.parquet").exists() or (
    _ROOT / "equity_table.parquet"
).exists()

requires_equity_table = pytest.mark.skipif(
    not _EQUITY_TABLE_PRESENT,
    reason="PC-gated: requires equity_table.parquet for the real Canonicalizer",
)

_SRP_SPOT = {
    "street": "preflop",
    "hero_hole": ["Ah", "Kh"],
    "hero_position": "BTN",
    "action_sequence": ["BTN:open_2.5", "BB:call"],
}


def test_all_hand_classes_is_169():
    from tools.build_preflop_equity_table import ALL_HAND_CLASSES

    assert len(ALL_HAND_CLASSES) == 169


def test_combo_counts_sum_to_1326():
    from tools.build_preflop_equity_table import ALL_HAND_CLASSES, _villain_combos_for_class

    total = sum(len(_villain_combos_for_class(cls, set()) or []) for cls in ALL_HAND_CLASSES)
    assert total == 1326


def test_pair_suited_offsuit_combo_counts():
    from tools.build_preflop_equity_table import _villain_combos_for_class

    pair_combos = _villain_combos_for_class("AA", set())
    assert pair_combos is not None
    assert len(pair_combos) == 6

    suited_combos = _villain_combos_for_class("AKs", set())
    assert suited_combos is not None
    assert len(suited_combos) == 4

    offsuit_combos = _villain_combos_for_class("AKo", set())
    assert offsuit_combos is not None
    assert len(offsuit_combos) == 12


@requires_equity_table
def test_range_batch_returns_169_hands():
    from src.study.probe import probe_range_for_actor

    mock_milvus = MagicMock()
    mock_milvus.search.return_value = []

    result = probe_range_for_actor(
        _SRP_SPOT,
        actor_position="BTN",
        _tsdb_conn=_FakeConn(),
        _milvus=mock_milvus,
    )
    assert len(result["hands"]) == 169
    agg = result["aggregate_range_pct"]
    assert isinstance(agg, float)
    assert 0.0 <= agg <= 1.0


def test_range_batch_forwards_spot_to_probe(monkeypatch):
    """Each per-class probe must carry the built spot, else the no-strategy_nodes
    kNN fallback short-circuits (Mode A) and every cell renders blank."""
    import src.study.probe as probe_mod
    import tools.build_embedding as be

    monkeypatch.setattr(be, "encode_spot_to_cluster_key", lambda s: "street_class=postflop")

    seen: list[dict | None] = []

    def _fake_probe(cluster_key, *, k, spot=None, _tsdb_conn=None, _milvus=None):
        seen.append(spot)
        return {"engine_response": {"action_dist": {"check": 1.0}}}

    monkeypatch.setattr(probe_mod, "probe_by_cluster_key", _fake_probe)

    result = probe_mod.probe_range_for_actor(
        {"street": "flop", "board": ["Qs", "3c", "2d"], "pot_type": "3bet", "n_players": 2},
        actor_position="BB",
        _tsdb_conn=_FakeConn(),
        _milvus=MagicMock(),
    )

    assert len(result["hands"]) == 169
    # one probe per non-blocked class; every call must forward a spot with a pinned hole
    assert seen, "no per-class probes were issued"
    assert all(s is not None and s.get("hero_hole") for s in seen)


def test_range_batch_surfaces_uncovered_combos(monkeypatch):
    """Classes the engine has no strategy for are reported as uncovered, not folded."""
    import src.study.probe as probe_mod
    import tools.build_embedding as be

    monkeypatch.setattr(be, "encode_spot_to_cluster_key", lambda s: "street_class=postflop")

    def _no_strategy(cluster_key, *, k, spot=None, _tsdb_conn=None, _milvus=None):
        return {"engine_response": {"action_dist": None}}

    monkeypatch.setattr(probe_mod, "probe_by_cluster_key", _no_strategy)

    result = probe_mod.probe_range_for_actor(
        {"street": "flop", "board": ["Qs", "3c", "2d"], "pot_type": "3bet", "n_players": 2},
        actor_position="BB",
        _tsdb_conn=_FakeConn(),
        _milvus=MagicMock(),
    )

    # Every unblocked combo is uncovered; none counted as fold.
    unblocked = sum(h["combos"] for h in result["hands"])
    assert unblocked > 0
    assert result["uncovered_combos"] == unblocked
    assert result["aggregate_range_pct"] == 0.0
