"""TDD tests for Milvus upsert schema fix (15-06). long-ok

Pins the 19-field Milvus collection schema contract for PatchEngine.apply node_row,
PatchSpec defaults, _inject street+spr_x100 derivation, and canary A5 output_fields.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

_SCHEMA_FIELDS = frozenset(
    {
        "decision_id",
        "embedding",
        "street_class",
        "street",
        "pot_type",
        "hero_pos_rel",
        "n_players_active",
        "spr_x100",
        "hero_action_type",
        "active",
        "confidence",
        "gto_score",
        "feature_spec_version",
        "action_dist",
        "hero_action_size_pot_frac",
        "raise_ratio",
        "hero_action_allin",
        "added_at",
        "removed_at",
    }
)

# PatchSpec field defaults  # rot-allow


def test_patchspec_street_default_empty_string() -> None:
    """PatchSpec.street defaults to empty string sentinel."""
    from src.patch_engine import PatchSpec

    p = PatchSpec(
        cluster_key="ck",
        decision_id="dp_ck",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=None,
    )
    assert p.street == ""


def test_patchspec_spr_x100_default_minus_one() -> None:
    """PatchSpec.spr_x100 defaults to -1 (no SPR filter sentinel)."""
    from src.patch_engine import PatchSpec

    p = PatchSpec(
        cluster_key="ck",
        decision_id="dp_ck",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=None,
    )
    assert p.spr_x100 == -1


def test_patchspec_accepts_street_and_spr_x100() -> None:
    """PatchSpec accepts explicit street + spr_x100 values."""
    from src.patch_engine import PatchSpec

    p = PatchSpec(
        cluster_key="ck",
        decision_id="dp_ck",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=None,
        street="flop",
        spr_x100=350,
    )
    assert p.street == "flop"
    assert p.spr_x100 == 350


# node_row schema correctness  # rot-allow


def _make_mock_tsdb(prev_id: uuid.UUID) -> MagicMock:
    conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (prev_id, "autoloop")
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.transaction.return_value.__enter__ = MagicMock(return_value=conn)
    conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    return conn


def test_node_row_has_exactly_19_schema_fields() -> None:
    """apply() builds node_row with EXACTLY the 19 Milvus collection schema fields."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"bet_50": 0.8, "fold": 0.2},
        embedding=[0.1, 0.2],
        prev_node_id=prev_id,
        street="flop",
        spr_x100=350,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_tsdb = _make_mock_tsdb(prev_id)
    mock_milvus = MagicMock()

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    mock_milvus.upsert.assert_called_once()
    node_row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert set(node_row.keys()) == _SCHEMA_FIELDS, (
        f"node_row keys mismatch.\n"
        f"  Extra  (must remove): {set(node_row.keys()) - _SCHEMA_FIELDS}\n"
        f"  Missing (must add):   {_SCHEMA_FIELDS - set(node_row.keys())}"
    )


def test_node_row_action_dist_is_json_string() -> None:
    """apply() serializes action_dist to a JSON string on the Milvus row (VARCHAR field)."""
    import json

    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"bet_50": 0.8, "fold": 0.2},
        embedding=[0.1, 0.2],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1, ev_loss_delta=0.1, n_hands=100, confidence_interval=(0.01, 0.2), n_cluster_hits=10
    )
    mock_tsdb = _make_mock_tsdb(prev_id)
    mock_milvus = MagicMock()
    PatchEngine().apply(patch, validation=validation, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    node_row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert isinstance(node_row["action_dist"], str)
    assert json.loads(node_row["action_dist"]) == {"bet_50": 0.8, "fold": 0.2}


def test_node_row_uses_embedding_not_vector() -> None:
    """apply() node_row uses 'embedding' key — NOT 'vector'."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"fold": 1.0},
        embedding=[0.5, 0.6],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_tsdb = _make_mock_tsdb(prev_id)
    mock_milvus = MagicMock()

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    node_row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert "embedding" in node_row, "node_row must have 'embedding' key"
    assert "vector" not in node_row, "node_row must NOT have 'vector' key"
    assert node_row["embedding"] == [0.5, 0.6]


def test_node_row_has_no_source_field() -> None:
    """apply() node_row does NOT include 'source' — not a Milvus schema field."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_tsdb = _make_mock_tsdb(prev_id)
    mock_milvus = MagicMock()

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    node_row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert "source" not in node_row, "node_row must NOT contain 'source' (not a Milvus field)"


def test_node_row_feature_spec_version_is_int() -> None:
    """apply() node_row feature_spec_version is an int, NOT the string 'v2'."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_tsdb = _make_mock_tsdb(prev_id)
    mock_milvus = MagicMock()

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    node_row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert isinstance(node_row["feature_spec_version"], int), (
        f"feature_spec_version must be int, got {type(node_row['feature_spec_version'])!r} "
        f"value={node_row['feature_spec_version']!r}"
    )
    assert node_row["feature_spec_version"] != "v2"


def test_node_row_street_and_spr_x100_present() -> None:
    """apply() node_row includes 'street' and 'spr_x100' from PatchSpec."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=prev_id,
        street="turn",
        spr_x100=480,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_tsdb = _make_mock_tsdb(prev_id)
    mock_milvus = MagicMock()

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    node_row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert "street" in node_row, "node_row must include 'street'"
    assert "spr_x100" in node_row, "node_row must include 'spr_x100'"
    assert node_row["street"] == "turn"
    assert node_row["spr_x100"] == 480


def test_node_row_street_default_sentinel() -> None:
    """apply() with PatchSpec.street='' (default) passes empty string into node_row."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_tsdb = _make_mock_tsdb(prev_id)
    mock_milvus = MagicMock()

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    node_row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert node_row["street"] == ""
    assert node_row["spr_x100"] == -1


def test_node_row_stamps_added_at_now_removed_at_zero() -> None:
    """apply() stamps a positive int added_at (live patch) and removed_at=0 (not removed)."""
    from datetime import UTC, datetime

    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_tsdb = _make_mock_tsdb(prev_id)
    mock_milvus = MagicMock()

    before = int(datetime.now(tz=UTC).timestamp())
    PatchEngine().apply(patch, validation=validation, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)
    after = int(datetime.now(tz=UTC).timestamp())

    node_row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert isinstance(node_row["added_at"], int)
    assert before <= node_row["added_at"] <= after
    assert node_row["removed_at"] == 0


# _inject: street + spr_x100 in the Milvus node row  # rot-allow


def _make_full_obs_row(street: str = "flop", pot: int = 1000, effective_stack: int = 9500) -> dict:
    """Build a realistic obs_row whose felt_snapshot matches the real shape."""
    return {
        "obs_id": "obs-schema-fix-001",
        "decision_id": "dp-schema-fix-001",
        "cluster_key": "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop",
        "embedding": [0.1] * 80,
        "max_neighbor_distance": 0.8,
        "felt_snapshot": {
            "action_sequence": ["BTN:raise:3", "BB:3bet:9", "BTN:call"],
            "board_cards": ["Ah", "7c", "2d"],
            "effective_stack_bb": effective_stack / 100,
            "hero_bet_size_bb": 0.0,
            "hero_facing_bet_bb": 0.0,
            "hero_hole_cards": ["Kc", "Qh"],
            "hero_position": "BB",
            "opponents_remaining": 1,
            "pot_size_bb": pot / 100,
            "prior_street_aggressor": "BTN",
            "street": street,
        },
        "cluster_freq": 40,
        "ev_loss": 0.7,
        "priority_score": 40 * 0.7 * 0.8,
    }


def test_inject_passes_street_to_milvus_row() -> None:
    """_inject extracts street from full_context and puts it in the Milvus node row."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_full_obs_row(street="turn")
    spot = SolverSpot(
        pot=1000,
        effective_stack=9500,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"bet_50": 0.8, "check": 0.2}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
        )

    mock_milvus.upsert.assert_called_once()
    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert row["street"] == "turn", f"Expected street='turn' in Milvus row, got {row.get('street')!r}"


def test_inject_derives_spr_x100_correctly() -> None:
    """_inject derives spr_x100 = round(effective_stack / pot * 100) and puts it in Milvus row.

    pot=1000 (x100 cents), effective_stack=9500 -> spr = 9500/1000 * 100 = 950.
    """
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_full_obs_row(street="flop", pot=1000, effective_stack=9500)
    spot = SolverSpot(
        pot=1000,
        effective_stack=9500,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"check": 1.0}
    mock_result.exploitability_pct = 0.3

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
        )

    mock_milvus.upsert.assert_called_once()
    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    expected_spr_x100 = round(9500 / 1000 * 100)  # = 950
    assert row["spr_x100"] == expected_spr_x100, (
        f"Expected spr_x100={expected_spr_x100} in Milvus row, got {row.get('spr_x100')}"
    )


def test_inject_spr_x100_sentinel_when_full_context_pot_zero() -> None:
    """_inject guard: if full_context['pot'] is zero, spr_x100=-1 in the Milvus row."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_full_obs_row(street="flop", pot=1000, effective_stack=9500)
    spot = SolverSpot(
        pot=1000,
        effective_stack=9500,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"check": 1.0}
    mock_result.exploitability_pct = 0.3

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    mock_milvus = MagicMock()

    patched_context = {
        "pot": 0,
        "effective_stack": 9500,
        "street": "flop",
        "action_line_full": ["BTN:raise:3", "BB:3bet:9", "BTN:call"],
        "hero_hole": ["Kc", "Qh"],
        "board": ["Ah", "7c", "2d"],
        "prev_bet": None,
        "hero_pos": "BB",
        "villain_pos": "BTN",
        "pot_type": "3bet",
        "n_players_at_street": 2,
        "preflop_action_seq": ["BTN:raise:3", "BB:3bet:9", "BTN:call"],
        "range_ip": "AA",
        "range_oop": "AA",
        "solver_settings": {},
    }

    with (
        patch("src.solver.queue_driver.persist_solve"),
        patch("src.solver.queue_driver._build_spot", return_value=(spot, patched_context)),
    ):
        _inject(
            obs_row,
            spot,
            mock_result,
            conn=mock_conn,
            milvus=mock_milvus,
            palette_lookup={},
        )

    mock_milvus.upsert.assert_called_once()
    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert row["spr_x100"] == -1, (
        f"Expected spr_x100=-1 (sentinel for zero pot) in Milvus row, got {row.get('spr_x100')}"
    )


def test_inject_node_row_has_exactly_19_schema_fields() -> None:
    """_inject builds the Milvus node row with EXACTLY the 19 schema fields."""
    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_full_obs_row(street="flop")
    spot = SolverSpot(
        pot=1000,
        effective_stack=9500,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"bet_50": 0.8, "check": 0.2}
    mock_result.exploitability_pct = 0.4

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    mock_milvus = MagicMock()

    with patch("src.solver.queue_driver.persist_solve"):
        _inject(obs_row, spot, mock_result, conn=mock_conn, milvus=mock_milvus, palette_lookup={})

    mock_milvus.upsert.assert_called_once()
    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert set(row.keys()) == _SCHEMA_FIELDS, (
        f"node_row keys mismatch.\n"
        f"  Extra  (must remove): {set(row.keys()) - _SCHEMA_FIELDS}\n"
        f"  Missing (must add):   {_SCHEMA_FIELDS - set(row.keys())}"
    )


def test_inject_stamps_added_at_now_removed_at_zero() -> None:
    """_inject stamps a positive int added_at (live solve) and removed_at=0."""
    from datetime import UTC, datetime

    from src.solver.postflop_cli import SolverSpot
    from src.solver.queue_driver import _inject

    obs_row = _make_full_obs_row(street="flop")
    spot = SolverSpot(
        pot=1000,
        effective_stack=9500,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.action_dist = {"check": 1.0}
    mock_result.exploitability_pct = 0.3

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    mock_milvus = MagicMock()

    before = int(datetime.now(tz=UTC).timestamp())
    with patch("src.solver.queue_driver.persist_solve"):
        _inject(obs_row, spot, mock_result, conn=mock_conn, milvus=mock_milvus, palette_lookup={})
    after = int(datetime.now(tz=UTC).timestamp())

    row = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert isinstance(row["added_at"], int)
    assert before <= row["added_at"] <= after
    assert row["removed_at"] == 0


# Canary A5 output_fields  # rot-allow


def test_canary_a5_output_fields_is_decision_id(tmp_path) -> None:
    """Canary A5 search uses output_fields=['decision_id'] — collection has no cluster_key field."""
    import ast
    from pathlib import Path

    canary_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "phase15_canary.py"
    canary_src = canary_path.read_text()
    tree = ast.parse(canary_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "output_fields":
            val = node.value
            if isinstance(val, ast.List):
                fields = [elt.s for elt in val.elts if isinstance(elt, ast.Constant)]
                assert "cluster_key" not in fields, (
                    "output_fields must NOT include 'cluster_key' — field does not exist in "
                    "Milvus schema. Use ['decision_id'] instead."
                )
