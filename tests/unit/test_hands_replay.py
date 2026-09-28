"""Unit tests for get_replay_by_hand + unified by-hand replay endpoint (RPLY-01/04).

No DB or HM3 SQLite required — all DB and HM path calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SELECT_COLS = [
    "obs_id",
    "session_id",
    "ts",
    "cluster_key",
    "action_taken",
    "source",
    "solver_label",
    "max_neighbor_distance",
    "flagged_sparse",
    "embedding",
    "hand_id",
    "decision_id",
    "felt_snapshot",
]


def _make_cursor(rows: list[tuple], cols: list[str] = _SELECT_COLS) -> MagicMock:
    """Return a MagicMock cursor with description + fetchall wired up."""
    cur = MagicMock()
    cur.description = [(c,) for c in cols]
    cur.fetchall.return_value = rows
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    return cur


def _make_conn(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


def _fake_felt(
    street: str = "preflop",
    board: list[str] | None = None,
    hero_hole_cards: list[str] | None = None,
    pot_size_bb: float = 3.0,
    action_sequence: list[str] | None = None,
) -> dict:
    return {
        "street": street,
        "board_cards": board or [],
        "hero_hole_cards": hero_hole_cards or ["Ah", "Kd"],
        "pot_size_bb": pot_size_bb,
        "action_sequence": action_sequence or ["UTG:fold"],
    }


# ---------------------------------------------------------------------------
# RPLY-01: get_replay_by_hand
# ---------------------------------------------------------------------------


def test_get_replay_by_hand_orders_by_dp_index() -> None:
    """get_replay_by_hand filters by hand_id and orders by the numeric dp index.

    decision_id is f"{hand_id}_dp{idx}"; a lexical ORDER BY decision_id puts _dp10
    before _dp2, scrambling street order. The query must sort by the integer suffix.
    """
    from src.study.hands import get_replay_by_hand

    rows = [
        tuple("obs1 sess1 2026-01-01 ck1 call sim None 0.0 False [] hand1 0 {}".split()),
        tuple("obs2 sess1 2026-01-01 ck1 fold sim None 0.0 False [] hand1 1 {}".split()),
        tuple("obs3 sess1 2026-01-01 ck1 raise sim None 0.0 False [] hand1 2 {}".split()),
    ]
    cur = _make_cursor(rows)
    conn = _make_conn(cur)

    result = get_replay_by_hand("hand1", _tsdb_conn=conn)

    cur.execute.assert_called_once()
    call_sql, call_params = cur.execute.call_args[0]
    assert "WHERE hand_id = %s" in call_sql, f"Expected hand_id filter, got: {call_sql!r}"
    assert "split_part(decision_id, '_dp', 2)" in call_sql, (
        f"Expected numeric dp-index ordering (not lexical decision_id), got: {call_sql!r}"
    )
    assert "decision_id ASC" not in call_sql, (
        f"Lexical 'ORDER BY decision_id ASC' scrambles _dp10 vs _dp2; got: {call_sql!r}"
    )
    assert call_params == ("hand1",), f"Expected params ('hand1',), got {call_params!r}"
    assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
    assert all(isinstance(r, dict) for r in result)


def test_get_replay_by_hand_no_f_string_interpolation() -> None:
    """Confirm the function exists and can be imported without touching DB."""
    import inspect

    from src.study.hands import get_replay_by_hand

    src = inspect.getsource(get_replay_by_hand)
    assert 'f"SELECT' not in src, "f-string SELECT found in get_replay_by_hand"
    assert "f'SELECT" not in src, "f-string SELECT found in get_replay_by_hand"
    assert "% hand_id" not in src, "% hand_id interpolation found"
    assert "+ hand_id" not in src, "+ hand_id concatenation found"


# ---------------------------------------------------------------------------
# RPLY-04: unified step shape
# ---------------------------------------------------------------------------

_UNIFIED_KEYS = {
    "step_idx",
    "street",
    "board",
    "hero_hole",
    "pot",
    "big_blind",
    "action_sequence",
    "action_taken",
    "seat_map",
    "button_seat",
    "hero_seat",
    "actor",
    "bet_amount",
    "street_committed",
    "folded",
    "shown_cards",
    "decision_id",
}


def _make_app():
    """Build a minimal FastAPI app with just the hands router for TestClient tests.

    Overrides the get_tsdb dependency to inject a no-op mock so tests do not require
    a live TimescaleDB connection.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.deps import get_tsdb
    from src.api.hands import router

    mock_conn = MagicMock()

    def _mock_get_tsdb():
        yield mock_conn

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_tsdb] = _mock_get_tsdb
    return TestClient(app)


def _sim_row(dp: int, pos: str, hole: list[str], action: str, street: str, board: list[str]) -> dict:
    return {
        "obs_id": f"obs_{dp}",
        "session_id": "sess1",
        "ts": "2026-01-01",
        "cluster_key": "ck1",
        "action_taken": action,
        "source": "sim",
        "solver_label": None,
        "max_neighbor_distance": 0.0,
        "flagged_sparse": False,
        "embedding": [],
        "hand_id": "sess1_h0",
        "decision_id": f"sess1_h0_dp{dp}",
        "felt_snapshot": {
            "street": street,
            "board_cards": board,
            "pot_size_bb": 1.5,
            "hero_position": pos,
            "hero_hole_cards": hole,
            "action_sequence": [],
            "effective_stack_bb": 99.0,
        },
    }


def test_sim_source_reconstructs_godview_timeline() -> None:
    """SIM rows reconstruct ONE god-view timeline: seats + all hole cards fixed,
    actor advances per action, NOT a different hero per step."""
    sim_rows = [
        _sim_row(0, "UTG", ["Jd", "6c"], "fold", "preflop", []),
        _sim_row(1, "BTN", ["Ah", "Ad"], "call", "preflop", []),
        _sim_row(2, "BB", ["Td", "Ks"], "check", "flop", ["3c", "3s", "7h"]),
    ]

    with (
        patch("src.api.hands.get_replay_by_hand", return_value=sim_rows),
        patch("src.api.hands.parse_hm_replay", return_value=[]) as mock_hm,
    ):
        client = _make_app()
        resp = client.get("/api/hands/by-hand/sess1_h0/replay")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    steps = resp.json()
    assert len(steps) == 3, "one step per action (dp)"

    # SIM is always 6-max: the god-view seats the full ring on every step. Seats with
    # no revealing dp (here MP/CO/SB) appear without hole cards.
    names = {v["name"] for v in steps[0]["seat_map"].values()}
    assert names == {"UTG", "MP", "CO", "BTN", "SB", "BB"}
    for s in steps:
        assert {v["name"] for v in s["seat_map"].values()} == names
        assert s["seat_cards"]["BTN"] == ["Ah", "Ad"], "BTN cards known on every step"
        assert s["seat_cards"]["BB"] == ["Td", "Ks"]

    # Actor advances with the timeline (not a rotating hero anchoring the felt).
    assert [s["actor"] for s in steps] == ["UTG", "BTN", "BB"]
    # Board builds per street.
    assert steps[0]["board"] == [] and steps[2]["board"] == ["3c", "3s", "7h"]
    # Fold accumulates after the folding step.
    assert "UTG" not in steps[0]["folded"] and "UTG" in steps[2]["folded"]
    mock_hm.assert_not_called()


def test_unified_shape_hm_source() -> None:
    """HM source: get_replay_by_hand returns [] -> HM path called -> steps with seat_map."""
    hm_events = [
        {
            "obs_id": "evt1",
            "hand_id": "12345",
            "ts": "",
            "street": "preflop",
            "hero_pos_rel": "IP",
            "cluster_key": "hh_12345_preflop",
            "action_taken": "Hero: call",
            "spot_features": {"board": [], "hero_hole": ["Ah", "Kd"]},
            "seat_map": {"1": {"name": "Hero", "stack": 100.0}, "2": {"name": "Villain", "stack": 100.0}},
            "button_seat": 1,
            "hero_seat": 1,
            "actor": "Hero",
            "bet_amount": None,
            "pot": 1.5,
            "big_blind": 1.0,
            "street_committed": {"Hero": 0.5},
            "shown_cards": {},
        },
        {
            "obs_id": "evt2",
            "hand_id": "12345",
            "ts": "",
            "street": "preflop",
            "hero_pos_rel": "OOP",
            "cluster_key": "hh_12345_preflop",
            "action_taken": "Villain: fold",
            "spot_features": {"board": [], "hero_hole": ["Ah", "Kd"]},
            "seat_map": {"1": {"name": "Hero", "stack": 100.0}, "2": {"name": "Villain", "stack": 100.0}},
            "button_seat": 1,
            "hero_seat": 1,
            "actor": "Villain",
            "bet_amount": None,
            "pot": 2.0,
            "big_blind": 1.0,
            "street_committed": {},
            "shown_cards": {},
        },
    ]

    with (
        patch("src.api.hands.get_replay_by_hand", return_value=[]),
        patch("src.api.hands.parse_hm_replay", return_value=hm_events) as mock_hm,
    ):
        client = _make_app()
        resp = client.get("/api/hands/by-hand/12345/replay")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    steps = resp.json()
    assert len(steps) == 2, f"Expected 2 steps, got {len(steps)}"
    for i, step in enumerate(steps):
        assert set(step.keys()) == _UNIFIED_KEYS, (
            f"Step {i} key mismatch. Extra: {set(step.keys()) - _UNIFIED_KEYS}, "
            f"Missing: {_UNIFIED_KEYS - set(step.keys())}"
        )
        assert step["seat_map"] is not None, f"HM step {i} must have seat_map"
    mock_hm.assert_called_once_with("12345")


def test_both_sources_render_with_inline_replayer_fields() -> None:
    """SIM (god-view) and HM (legacy) shapes differ, but BOTH carry the fields the
    InlineReplayer needs to render a felt: step_idx, street, board, pot, seat_map, actor."""
    sim_rows = [_sim_row(0, "BTN", ["As", "Ks"], "call", "preflop", [])]
    hm_events = [
        {
            "obs_id": "evt1",
            "hand_id": "99999",
            "ts": "",
            "street": "preflop",
            "hero_pos_rel": "IP",
            "cluster_key": "hh_99999_preflop",
            "action_taken": "Hero: call",
            "spot_features": {"board": [], "hero_hole": ["As", "Ks"]},
            "seat_map": {"1": {"name": "Hero", "stack": 50.0}},
            "button_seat": 1,
            "hero_seat": 1,
            "actor": "Hero",
            "bet_amount": None,
            "pot": 1.5,
            "big_blind": 0.5,
            "street_committed": {},
            "shown_cards": {},
        }
    ]

    with (
        patch("src.api.hands.get_replay_by_hand", return_value=sim_rows),
        patch("src.api.hands.parse_hm_replay", return_value=[]),
    ):
        client = _make_app()
        resp_sim = client.get("/api/hands/by-hand/sess1_h0/replay")

    with (
        patch("src.api.hands.get_replay_by_hand", return_value=[]),
        patch("src.api.hands.parse_hm_replay", return_value=hm_events),
    ):
        client = _make_app()
        resp_hm = client.get("/api/hands/by-hand/99999/replay")

    required = {"step_idx", "street", "board", "pot", "seat_map", "actor", "action_taken"}
    sim_step = resp_sim.json()[0]
    hm_step = resp_hm.json()[0]
    assert required <= set(sim_step.keys()), f"SIM missing: {required - set(sim_step.keys())}"
    assert required <= set(hm_step.keys()), f"HM missing: {required - set(hm_step.keys())}"
    # SIM additionally carries the god-view per-seat cards.
    assert "seat_cards" in sim_step and sim_step["seat_cards"]["BTN"] == ["As", "Ks"]


# ---------------------------------------------------------------------------
# T-11-05: invalid hand_id → HTTP 400
# ---------------------------------------------------------------------------


def test_invalid_hand_id_400_path_traversal() -> None:
    """hand_id with path traversal chars returns HTTP 400 (T-11-05)."""
    with (
        patch("src.api.hands.get_replay_by_hand", return_value=[]),
        patch("src.api.hands.parse_hm_replay", return_value=[]),
    ):
        client = _make_app()
        resp = client.get("/api/hands/by-hand/..%2Fetc/replay")

    assert resp.status_code in (400, 404), (
        f"Expected 400 or 404 for path-traversal hand_id, got {resp.status_code}"
    )


def test_invalid_hand_id_400_slash() -> None:
    """hand_id with illegal chars returns HTTP 400 (T-11-05)."""
    with (
        patch("src.api.hands.get_replay_by_hand", return_value=[]),
        patch("src.api.hands.parse_hm_replay", return_value=[]),
    ):
        client = _make_app()
        resp = client.get("/api/hands/by-hand/a!b/replay")

    assert resp.status_code == 400, f"Expected 400 for illegal chars in hand_id, got {resp.status_code}"


def test_invalid_hand_id_400_space() -> None:
    """hand_id with space or special char returns HTTP 400."""
    with (
        patch("src.api.hands.get_replay_by_hand", return_value=[]),
        patch("src.api.hands.parse_hm_replay", return_value=[]),
    ):
        client = _make_app()
        resp = client.get("/api/hands/by-hand/a b/replay")

    assert resp.status_code == 400, f"Expected 400 for space in hand_id, got {resp.status_code}"


def test_uuid_style_hand_id_accepted() -> None:
    """Sim hand_ids are `<uuid>_hNNN` (UUID contains hyphens) — the guard must accept them."""
    with (
        patch("src.api.hands.get_replay_by_hand", return_value=[]),
        patch("src.api.hands.parse_hm_replay", return_value=[]),
    ):
        client = _make_app()
        resp = client.get("/api/hands/by-hand/47674386-11cf-4175-a0bd-6740370913da_h199/replay")

    assert resp.status_code != 400, (
        f"UUID-style hand_id with hyphens must not be rejected by the guard, got {resp.status_code}"
    )
