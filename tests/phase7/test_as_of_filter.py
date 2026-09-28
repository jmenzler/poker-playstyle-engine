"""Corpus-versioning as-of mode for `_build_filter`.

A pin (`as_of_ts`) drops the live `active == True` prefix and selects the
then-state via the `added_at`/`removed_at` window. No pin = unchanged live mode.
"""


def _hard_filter() -> dict:
    return {
        "hero_pos_rel": "IP",
        "n_players_active": 2,
        "pot_type": "srp",
        "street_class": "postflop",
    }


def test_as_of_adds_window_and_drops_active():
    """as_of_ts set -> window clauses present, active prefix omitted."""
    from src.decision_engine.engine import _build_filter

    expr = _build_filter(_hard_filter(), as_of_ts=1000)
    assert "added_at <= 1000" in expr
    assert "removed_at == 0 or removed_at > 1000" in expr
    assert "active == True" not in expr  # rot-allow


def test_no_pin_keeps_active_and_no_window():
    """as_of_ts None -> live mode: active prefix kept, no window clauses."""
    from src.decision_engine.engine import _build_filter

    expr = _build_filter(_hard_filter())
    assert "active == True" in expr  # rot-allow
    assert "added_at" not in expr
    assert "removed_at" not in expr


def _engine_with_neighbors(tmp_path, *, as_of_ts):
    import json
    from unittest.mock import MagicMock

    from src.decision_engine.engine import KNNDecisionEngine

    out = tmp_path / "manifests"
    out.mkdir()
    for name, dim in (("preflop", 34), ("postflop", 80)):
        manifest = {
            "feature_spec_version": 3,
            "fit_population_n": 100,
            "fit_date": "2026-06-24",
            "collection": f"{name}_decisions",
            "dim_stats": [{"dim": i, "name": f"d{i}", "mean": 0.0, "std": 1.0} for i in range(dim)],
        }
        (out / f"zscore_{name}.json").write_text(json.dumps(manifest))

    client = MagicMock()
    client.search.return_value = [
        [
            {
                "distance": 0.9,
                "entity": {
                    "decision_id": "dp1",
                    "hero_action_type": "call",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            }
        ]
    ]
    engine = KNNDecisionEngine(
        client,
        out / "zscore_preflop.json",
        out / "zscore_postflop.json",
        use_preflop_charts=False,
        as_of_ts=as_of_ts,
    )
    return engine, client


def _preflop_state():
    from src.protocols.game_state import GameState

    return GameState(
        street="preflop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=(),
        pot_size_bb=3.0,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=2.5,
        hero_bet_size_bb=0.0,
        action_sequence=("UTG:fold", "MP:fold", "CO:fold"),
        opponents_remaining=2,
        prior_street_aggressor=None,
    )


def test_engine_pin_threads_as_of_window_into_search(tmp_path):
    """A pinned engine threads the as-of window into client.search's filter."""
    engine, client = _engine_with_neighbors(tmp_path, as_of_ts=1000)
    engine.decide(_preflop_state())
    filter_expr = client.search.call_args.kwargs["filter"]
    assert "added_at <= 1000" in filter_expr
    assert "removed_at == 0 or removed_at > 1000" in filter_expr
    assert "active == True" not in filter_expr  # rot-allow


def test_engine_unpinned_keeps_active_filter(tmp_path):
    """An unpinned engine keeps the live `active == True` prefix, no window."""
    engine, client = _engine_with_neighbors(tmp_path, as_of_ts=None)
    engine.decide(_preflop_state())
    filter_expr = client.search.call_args.kwargs["filter"]
    assert "active == True" in filter_expr  # rot-allow
    assert "added_at" not in filter_expr


def test_engine_from_env_resolves_version_to_cutoff(monkeypatch):
    """engine_from_env(as_of_version=N) resolves the cutoff and pins the engine."""
    from unittest.mock import MagicMock

    import src.decision_engine.engine as engine_mod

    captured: dict = {}

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            captured["as_of_ts"] = kwargs.get("as_of_ts")

    monkeypatch.setattr(engine_mod, "_milvus_client_from_env", lambda: MagicMock())
    monkeypatch.setattr(engine_mod, "KNNDecisionEngine", FakeEngine)
    monkeypatch.setattr(engine_mod.corpus_versions, "resolve_cutoff_ts", lambda version: 4242, raising=False)

    engine_mod.engine_from_env(as_of_version=7, warmup=False)
    assert captured["as_of_ts"] == 4242
