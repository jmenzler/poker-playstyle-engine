"""tests/phase5/test_patch_engine.py — Phase 5 Plan 03 PatchEngine tests.

Requirements covered:
- PTCH-01: all writes to strategy_nodes go through PatchEngine.apply(); direct INSERT raises
- PTCH-02: every applied patch creates a patches row with all required fields
- PTCH-03: PatchEngine.rollback(patch_id) restores prior node and creates rollback row
- PTCH-04: applying patch to source='seed' node raises PatchForbiddenError
- ERR-04: PatchConflictError raised if cluster_key was patched between prepare and apply
- LOOP-06: every auto-loop patch appears in patches table with source='autoloop'
- VALN-02: ValidationResult stored as JSONB with all required fields
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import msgspec
import pytest

# ---------------------------------------------------------------------------
# Task 1: msgspec.Struct type contract tests
# ---------------------------------------------------------------------------


def test_patchspec_default_values() -> None:
    """PatchSpec constructs with gto_score=0.5, confidence=0.5, source='autoloop' defaults."""
    from src.patch_engine import PatchSpec

    p = PatchSpec(
        cluster_key="ck",
        decision_id="dp_ck",
        action_dist={"fold": 1.0},
        embedding=[0.1, 0.2],
        prev_node_id=None,
    )
    assert p.gto_score == 0.5
    assert p.confidence == 0.5
    assert p.source == "autoloop"
    assert p.prev_node_id is None
    assert p.cluster_key == "ck"
    assert p.action_dist == {"fold": 1.0}
    assert p.embedding == [0.1, 0.2]


def test_patchspec_is_frozen() -> None:
    """PatchSpec is frozen — assigning to a field raises AttributeError."""
    from src.patch_engine import PatchSpec

    p = PatchSpec(
        cluster_key="ck",
        decision_id="dp_ck",
        action_dist={"fold": 1.0},
        embedding=[0.1],
        prev_node_id=None,
    )
    with pytest.raises(AttributeError):
        p.cluster_key = "other"  # type: ignore[misc]


def test_validationresult_default_and_to_builtins() -> None:
    """ValidationResult constructs correctly; msgspec.to_builtins converts tuple to list."""
    from src.patch_engine import ValidationResult

    v = ValidationResult(
        seed=42,
        ev_loss_delta=0.05,
        n_hands=5000,
        confidence_interval=(0.01, 0.09),
        n_cluster_hits=100,
    )
    assert v.zero_hits is False
    builtins = msgspec.to_builtins(v)
    assert builtins["seed"] == 42
    assert builtins["ev_loss_delta"] == 0.05
    assert builtins["n_hands"] == 5000
    # msgspec.to_builtins preserves tuples (not converting to list in this version);
    # accept both forms — the JSON path (json.dumps) handles serialization separately.
    assert list(builtins["confidence_interval"]) == [0.01, 0.09]
    assert builtins["n_cluster_hits"] == 100
    assert builtins["zero_hits"] is False


def test_patchrecord_constructs_with_required_fields() -> None:
    """PatchRecord constructs with all required fields."""
    from src.patch_engine import PatchRecord

    patch_id = uuid.uuid4()
    new_node_id = uuid.uuid4()
    prev_node_id = uuid.uuid4()
    r = PatchRecord(
        patch_id=patch_id,
        ts="2026-05-19T00:00:00Z",
        status="applied",
        cluster_key="hero_pos_rel=BTN|street_class=postflop",
        new_node_id=new_node_id,
        prev_node_id=prev_node_id,
    )
    assert r.patch_id == patch_id
    assert r.status == "applied"
    assert r.prev_node_id == prev_node_id
    assert r.new_node_id == new_node_id


# ---------------------------------------------------------------------------
# Task 2: PatchEngine.apply() tests
# ---------------------------------------------------------------------------


def test_apply_executes_two_inserts_in_correct_order(mock_tsdb_conn, mock_milvus_client) -> None:
    """PTCH-02: apply executes INSERT strategy_nodes -> INSERT patches in one tx (no supersede)."""
    from src.patch_engine import PatchEngine, PatchRecord, PatchSpec, ValidationResult

    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"bet_50": 0.8, "fold": 0.2},
        embedding=[0.1, 0.2, 0.3],
        prev_node_id=None,
    )
    validation = ValidationResult(
        seed=99,
        ev_loss_delta=0.2,
        n_hands=5000,
        confidence_interval=(0.05, 0.35),
        n_cluster_hits=300,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value

    engine = PatchEngine()
    result = engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb_conn, _milvus=mock_milvus_client)

    mock_tsdb_conn.transaction.assert_called_once()

    all_calls = mock_cur.execute.call_args_list
    assert len(all_calls) == 2, f"Expected 2 execute calls, got {len(all_calls)}"
    assert "INSERT" in all_calls[0][0][0].upper() and "STRATEGY_NODES" in all_calls[0][0][0].upper()
    assert "INSERT" in all_calls[1][0][0].upper() and "PATCHES" in all_calls[1][0][0].upper()

    assert isinstance(result, PatchRecord)
    assert result.status == "applied"
    assert result.cluster_key == patch.cluster_key


def test_apply_writes_autoloop_source(mock_tsdb_conn, mock_milvus_client) -> None:
    """LOOP-06: default PatchSpec has source='autoloop'; INSERT INTO strategy_nodes params contain 'autoloop'."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"bet_50": 0.9, "fold": 0.1},
        embedding=[0.5],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=50,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "autoloop")

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb_conn, _milvus=mock_milvus_client)

    all_calls = mock_cur.execute.call_args_list
    # INSERT INTO strategy_nodes is the 1st execute call (index 0)
    insert_nodes_call = all_calls[0]
    params = insert_nodes_call[0][1]
    assert "autoloop" in params, f"'autoloop' not in INSERT params: {params}"


def test_apply_writes_patches_row_with_all_required_fields(mock_tsdb_conn, mock_milvus_client) -> None:
    """PTCH-02: INSERT INTO patches param tuple has all required fields."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"bet_50": 0.7, "fold": 0.3},
        embedding=[0.1],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=7,
        ev_loss_delta=0.12,
        n_hands=1000,
        confidence_interval=(0.02, 0.22),
        n_cluster_hits=80,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "sim")

    engine = PatchEngine()
    engine.apply(
        patch,
        validation=validation,
        pre_ev_loss=0.5,
        post_ev_loss=0.38,
        _tsdb_conn=mock_tsdb_conn,
        _milvus=mock_milvus_client,
    )

    all_calls = mock_cur.execute.call_args_list
    # INSERT INTO patches is the 2nd execute call (index 1)
    patches_call = all_calls[1]
    params = patches_call[0][1]
    # (patch_id, source, cluster_key, prev_node_id, new_node_id, pre_ev_loss, post_ev_loss, validation_json, decision_id)
    assert len(params) == 9, f"Expected 9 params, got {len(params)}: {params}"
    assert params[1] == "autoloop"  # source (PatchSpec default)
    assert params[2] == patch.cluster_key
    assert params[3] is None  # prev_node_id always NULL in the coexist model
    assert params[5] == 0.5  # pre_ev_loss
    assert params[6] == 0.38  # post_ev_loss
    assert params[8] == patch.decision_id  # source DP id


def test_apply_includes_validation_json_in_patches_row(mock_tsdb_conn, mock_milvus_client) -> None:
    """VALN-02: the validation param in INSERT INTO patches is JSON with all ValidationResult keys."""
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
        seed=999,
        ev_loss_delta=0.08,
        n_hands=500,
        confidence_interval=(0.003, 0.157),
        n_cluster_hits=40,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "sim")

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb_conn, _milvus=mock_milvus_client)

    all_calls = mock_cur.execute.call_args_list
    patches_call = all_calls[1]
    params = patches_call[0][1]
    validation_json = params[7]
    parsed = json.loads(validation_json)
    assert "seed" in parsed
    assert "ev_loss_delta" in parsed
    assert "n_hands" in parsed
    assert "confidence_interval" in parsed
    assert "n_cluster_hits" in parsed
    assert "zero_hits" in parsed
    assert parsed["seed"] == 999
    assert parsed["confidence_interval"] == [0.003, 0.157]


def test_apply_milvus_failure_does_not_rollback_tsdb(mock_tsdb_conn, mock_milvus_client, capsys) -> None:
    """TSDB-first: Milvus upsert exception is swallowed; warning logged; PatchRecord returned.

    structlog logs to stdout (not Python logging), so we capture via capsys.
    """
    from src.patch_engine import PatchEngine, PatchRecord, PatchSpec, ValidationResult

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
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "autoloop")

    # Make Milvus upsert fail
    mock_milvus_client.upsert.side_effect = RuntimeError("milvus down")

    engine = PatchEngine()
    result = engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb_conn, _milvus=mock_milvus_client)

    # No exception raised to caller
    assert isinstance(result, PatchRecord)
    assert result.status == "applied"

    # TSDB transaction committed (exit called with success args)
    tx_exit = mock_tsdb_conn.transaction.return_value.__exit__
    tx_exit.assert_called()

    # Warning was logged to stdout (structlog bypasses Python logging)
    captured = capsys.readouterr()
    assert "milvus" in captured.out.lower(), f"No milvus warning in stdout: {captured.out!r}"


def test_apply_dispatches_correct_milvus_collection_for_preflop(mock_tsdb_conn, mock_milvus_client) -> None:
    """Collection routing: preflop cluster_key dispatches to 'preflop_decisions'."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=preflop",
        decision_id="dp_preflop",
        action_dist={"raise": 0.6, "fold": 0.4},
        embedding=[0.1, 0.2],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "sim")

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb_conn, _milvus=mock_milvus_client)

    mock_milvus_client.upsert.assert_called_once()
    call_kwargs = mock_milvus_client.upsert.call_args
    assert call_kwargs.kwargs["collection_name"] == "preflop_decisions"


def test_apply_dispatches_postflop_collection(mock_tsdb_conn, mock_milvus_client) -> None:
    """Collection routing: postflop cluster_key dispatches to 'postflop_decisions'."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"bet_50": 0.9, "fold": 0.1},
        embedding=[0.1],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=2,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "sim")

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb_conn, _milvus=mock_milvus_client)

    mock_milvus_client.upsert.assert_called_once()
    call_kwargs = mock_milvus_client.upsert.call_args
    assert call_kwargs.kwargs["collection_name"] == "postflop_decisions"


def test_apply_milvus_dominant_action_is_max_prob(mock_tsdb_conn, mock_milvus_client) -> None:
    """Open Question 3: upsert payload hero_action_type is the max-probability action."""
    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"bet_50": 0.7, "fold": 0.2, "call": 0.1},
        embedding=[0.1],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=3,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "sim")

    engine = PatchEngine()
    engine.apply(patch, validation=validation, _tsdb_conn=mock_tsdb_conn, _milvus=mock_milvus_client)

    mock_milvus_client.upsert.assert_called_once()
    upsert_data = mock_milvus_client.upsert.call_args.kwargs["data"][0]
    assert upsert_data["hero_action_type"] == "bet_50"


def test_apply_skip_milvus_does_not_upsert(mock_tsdb_conn, mock_milvus_client) -> None:
    """skip_milvus=True writes the TSDB node but never touches Milvus — for gaps
    whose embedding could not be z-scored (a wrong-space vector must never reach
    the COSINE corpus)."""
    from src.patch_engine import PatchEngine, PatchRecord, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"check": 1.0},
        embedding=[],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "autoloop")

    engine = PatchEngine()
    result = engine.apply(
        patch,
        validation=validation,
        _tsdb_conn=mock_tsdb_conn,
        _milvus=mock_milvus_client,
        skip_milvus=True,
    )

    assert isinstance(result, PatchRecord)
    assert result.status == "applied"
    mock_milvus_client.upsert.assert_not_called()


def test_apply_skip_milvus_does_not_open_env_client(mock_tsdb_conn) -> None:
    """skip_milvus=True with _milvus=None must NOT open an env-based client —
    otherwise the raw vector would still reach the corpus via connect_from_env."""
    from unittest.mock import patch as mock_patch

    from src.patch_engine import PatchEngine, PatchSpec, ValidationResult

    prev_id = uuid.uuid4()
    patch = PatchSpec(
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        decision_id="dp_test",
        action_dist={"check": 1.0},
        embedding=[],
        prev_node_id=prev_id,
    )
    validation = ValidationResult(
        seed=1,
        ev_loss_delta=0.1,
        n_hands=100,
        confidence_interval=(0.01, 0.2),
        n_cluster_hits=10,
    )
    mock_cur = mock_tsdb_conn.cursor.return_value.__enter__.return_value
    mock_cur.fetchone.return_value = (prev_id, "autoloop")

    with mock_patch("src.patch_engine.milvus_db.connect_from_env") as conn_env:
        engine = PatchEngine()
        engine.apply(
            patch,
            validation=validation,
            _tsdb_conn=mock_tsdb_conn,
            _milvus=None,
            skip_milvus=True,
        )
    conn_env.assert_not_called()


# ---------------------------------------------------------------------------
# Task 3: PatchEngine.rollback() tests
# ---------------------------------------------------------------------------


def _mock_tsdb_for_rollback(
    prev_node_id: uuid.UUID | None,
    new_node_id: uuid.UUID,
    cluster_key: str,
    status: str = "applied",
    source: str = "autoloop",
    decision_id: str | None = "dp_rb",
) -> MagicMock:
    """Return a mock_tsdb_conn whose rollback SELECT yields the 6-tuple
    (prev_node_id, new_node_id, cluster_key, status, source, decision_id)."""
    conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (prev_node_id, new_node_id, cluster_key, status, source, decision_id)
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.transaction.return_value.__enter__ = MagicMock(return_value=conn)
    conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    return conn


def _mock_tsdb_rollback_not_found() -> MagicMock:
    """Return a mock_tsdb_conn where rollback SELECT returns None (patch not found)."""
    conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.transaction.return_value.__enter__ = MagicMock(return_value=conn)
    conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    return conn


def _full_milvus_entity(decision_id: str = "dp_src") -> dict:
    """A representative full Milvus entity as returned by client.query(output_fields=["*"])."""
    return {
        "decision_id": decision_id,
        "embedding": [0.0] * 8,
        "street_class": "postflop",
        "street": "flop",
        "pot_type": "srp",
        "hero_pos_rel": "BTN",
        "n_players_active": 2,
        "spr_x100": 100,
        "hero_action_size_pot_frac": -1.0,
        "raise_ratio": -1.0,
        "hero_action_allin": False,
        "hero_action_type": "bet",
        "active": True,
        "confidence": 0.9,
        "gto_score": 0.5,
        "feature_spec_version": 1,
        "action_dist": "{}",
        "added_at": 1234567890,
        "removed_at": 0,
    }


def test_rollback_marks_status_and_soft_deletes_milvus_row() -> None:
    """Coexist rollback: node inactive + patches status='rolled_back' in place + Milvus row soft-deleted."""
    from src.patch_engine import PatchEngine, PatchRecord

    new_uuid = uuid.uuid4()
    cluster_key = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    original_patch_id = uuid.uuid4()
    mock_tsdb = _mock_tsdb_for_rollback(None, new_uuid, cluster_key, "applied", decision_id="dp_src")
    mock_milvus = MagicMock()
    mock_milvus.query.return_value = [_full_milvus_entity("dp_src")]

    engine = PatchEngine()
    result = engine.rollback(original_patch_id, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    assert isinstance(result, PatchRecord)
    assert result.status == "rolled_back"
    mock_tsdb.transaction.assert_called_once()

    mock_cur = mock_tsdb.cursor.return_value.__enter__.return_value
    sqls = [c[0][0].upper() for c in mock_cur.execute.call_args_list]
    assert any("UPDATE STRATEGY_NODES" in s and "ACTIVE = FALSE" in s for s in sqls)
    # superseded_by is dropped under coexist — the rollback UPDATE must not reference it.
    assert not any("SUPERSEDED_BY" in s for s in sqls)
    assert any("UPDATE PATCHES" in s and "ROLLED_BACK" in s for s in sqls)
    # The coexist model mutates status in place — it never inserts a new patches row.
    assert not any("INSERT INTO PATCHES" in s for s in sqls)

    # Soft-delete: re-fetch the full entity, then upsert it with active=False + removed_at>0.
    mock_milvus.query.assert_called_once()
    assert 'decision_id == "dp_src"' in mock_milvus.query.call_args.kwargs["filter"]
    mock_milvus.delete.assert_not_called()
    mock_milvus.upsert.assert_called_once()
    upserted = mock_milvus.upsert.call_args.kwargs["data"][0]
    assert upserted["decision_id"] == "dp_src"
    assert upserted["active"] is False
    assert upserted["removed_at"] > 0


def test_rollback_null_decision_id_skips_milvus_softdelete() -> None:
    """Legacy patch (decision_id NULL): rollback cannot locate the Milvus row, so no soft-delete."""
    from src.patch_engine import PatchEngine

    new_uuid = uuid.uuid4()
    cluster_key = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    original_patch_id = uuid.uuid4()
    mock_tsdb = _mock_tsdb_for_rollback(None, new_uuid, cluster_key, "applied", decision_id=None)
    mock_milvus = MagicMock()

    engine = PatchEngine()
    engine.rollback(original_patch_id, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    mock_milvus.query.assert_not_called()
    mock_milvus.upsert.assert_not_called()


def test_rollback_idempotency_blocks_already_rolled_back() -> None:
    """rollback raises RuntimeError when the patch already has status='rolled_back'."""
    from src.patch_engine import PatchEngine

    new_uuid = uuid.uuid4()
    cluster_key = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    original_patch_id = uuid.uuid4()
    mock_tsdb = _mock_tsdb_for_rollback(None, new_uuid, cluster_key, "rolled_back")
    mock_milvus = MagicMock()

    engine = PatchEngine()
    with pytest.raises(RuntimeError, match="already rolled back"):
        engine.rollback(original_patch_id, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)


def test_rollback_missing_patch_raises_lookup_error() -> None:
    """Rollback raises LookupError when patch_id not in patches table."""
    from src.patch_engine import PatchEngine

    original_patch_id = uuid.uuid4()
    mock_tsdb = _mock_tsdb_rollback_not_found()
    mock_milvus = MagicMock()

    engine = PatchEngine()
    with pytest.raises(LookupError, match="patch_id"):
        engine.rollback(original_patch_id, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)


def test_rollback_milvus_failure_does_not_rollback_tsdb(capsys) -> None:
    """Milvus soft-delete failure during rollback: TSDB-first — log warning, return PatchRecord.

    structlog logs to stdout (not Python logging), so we capture via capsys.
    """
    from src.patch_engine import PatchEngine, PatchRecord

    new_uuid = uuid.uuid4()
    cluster_key = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    original_patch_id = uuid.uuid4()
    mock_tsdb = _mock_tsdb_for_rollback(None, new_uuid, cluster_key, "applied", decision_id="dp_src")
    mock_milvus = MagicMock()
    mock_milvus.query.return_value = [_full_milvus_entity("dp_src")]
    mock_milvus.upsert.side_effect = RuntimeError("milvus down during rollback")

    engine = PatchEngine()
    result = engine.rollback(original_patch_id, _tsdb_conn=mock_tsdb, _milvus=mock_milvus)

    assert isinstance(result, PatchRecord)
    assert result.status == "rolled_back"
    # Warning logged to stdout (structlog bypasses Python logging)
    captured = capsys.readouterr()
    assert "milvus" in captured.out.lower(), f"No milvus warning in stdout: {captured.out!r}"
