"""tests/phase5/test_autoloop_step.py — AutoLoopDriver unit tests (LOOP-01..06).

Requirements covered:
- LOOP-01: AutoLoopDriver satisfies AutoLoop protocol (runtime_checkable)
- LOOP-02: sequential patch application (no concurrency)
- LOOP-03: patch applied only after ci_low > 0
- LOOP-04: config read from config/autoloop.toml
- LOOP-05: enabled=False exits before any DB I/O
- LOOP-06: every patch carries source='autoloop'
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from src._config import AutoLoopConfig
from src._errors import NoStrategyError, PatchConflictError
from src.patch_engine import PatchRecord, ValidationResult
from src.protocols.auto_loop import AutoLoop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    cluster_key: str = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
):
    """Create a minimal ClusterCandidate for tests without importing autoloop.driver."""
    from src.autoloop.leak_detector import ClusterCandidate

    return ClusterCandidate(
        cluster_key=cluster_key,
        ev_loss=0.5,
        n_obs=50,
        prev_node_id=None,
        score=0.5,
    )


def _make_validation_result(ci_low: float = 0.1, ci_high: float = 0.5, zero_hits: bool = False):
    return ValidationResult(
        seed=0xFEEDBEEF,
        ev_loss_delta=0.2,
        n_hands=100,
        confidence_interval=(ci_low, ci_high),
        n_cluster_hits=10,
        zero_hits=zero_hits,
    )


def _make_patch_record(
    cluster_key: str = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
):
    return PatchRecord(
        patch_id=uuid.uuid4(),
        ts="2026-01-01T00:00:00+00:00",
        status="applied",
        cluster_key=cluster_key,
        new_node_id=uuid.uuid4(),
        prev_node_id=None,
    )


def _make_driver_with_mocks(monkeypatch, *, config: AutoLoopConfig | None = None):
    """Create AutoLoopDriver and patch all external collaborators.

    Returns (driver, patches_dict) where patches_dict has:
      - load_toml_config: monkeypatched to return config
      - timescale_connect: monkeypatched
      - milvus_connect: monkeypatched
      - select_patch_candidates: monkeypatched
      - generate_candidate_action_dist: monkeypatched
      - validate: monkeypatched
      - ev_loss: monkeypatched
      - patch_engine_apply: monkeypatched on PatchEngine instance
    """
    from src.autoloop.driver import AutoLoopDriver

    if config is None:
        config = AutoLoopConfig(
            enabled=True,
            tau_leak=0.3,
            min_observations=5,
            max_patches_per_run=5,
            n_hands_validation=10,
            bootstrap_resamples=10,
            ci_alpha=0.05,
        )

    mock_config = MagicMock(return_value=config)
    monkeypatch.setattr("src.autoloop.driver.load_toml_config", mock_config)

    # Patch DSN helpers so tests don't require live env vars
    monkeypatch.setattr("src.autoloop.driver._tsdb_dsn_from_env", MagicMock(return_value="dsn://test"))
    monkeypatch.setattr(
        "src.autoloop.driver._milvus_uri_from_env", MagicMock(return_value="http://test:51530")
    )

    mock_tsdb = MagicMock()
    mock_tsdb_connect = MagicMock(return_value=mock_tsdb)
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("session-123",)
    mock_tsdb.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_tsdb.cursor.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("src.autoloop.driver.timescale.connect", mock_tsdb_connect)

    mock_milvus = MagicMock()
    mock_milvus_connect = MagicMock(return_value=mock_milvus)
    monkeypatch.setattr("src.autoloop.driver.milvus_db.connect", mock_milvus_connect)

    mock_select = MagicMock(return_value=[])
    monkeypatch.setattr("src.autoloop.driver.select_patch_candidates", mock_select)

    mock_gen = MagicMock(return_value=({"fold": 0.3, "call": 0.4, "raise_2x": 0.3}, [0.1] * 80, "dp_test"))
    monkeypatch.setattr("src.autoloop.driver.generate_candidate_action_dist", mock_gen)

    mock_validate = MagicMock(return_value=_make_validation_result(ci_low=0.1))
    monkeypatch.setattr("src.autoloop.driver.validate", mock_validate)

    mock_ev_loss = MagicMock(return_value=0.5)
    monkeypatch.setattr("src.autoloop.driver.ev_loss", mock_ev_loss)

    fake_engine = MagicMock()
    fake_adapter = MagicMock()

    driver = AutoLoopDriver(
        config_path=Path("config/autoloop.toml"),
        baseline_engine=fake_engine,
        sim_adapter=fake_adapter,
    )

    patches = {
        "load_toml_config": mock_config,
        "timescale_connect": mock_tsdb_connect,
        "milvus_connect": mock_milvus_connect,
        "tsdb_conn": mock_tsdb,
        "cursor": mock_cursor,
        "select_patch_candidates": mock_select,
        "generate_candidate_action_dist": mock_gen,
        "validate": mock_validate,
        "ev_loss": mock_ev_loss,
        "milvus_client": mock_milvus,
    }
    return driver, patches


# ---------------------------------------------------------------------------
# Test 1: LOOP-01 — Protocol satisfaction
# ---------------------------------------------------------------------------


def test_driver_satisfies_autoloop_protocol():
    """LOOP-01: AutoLoopDriver is a runtime-checkable AutoLoop instance."""
    from src.autoloop.driver import AutoLoopDriver

    driver = AutoLoopDriver()
    assert isinstance(driver, AutoLoop), "AutoLoopDriver must satisfy the AutoLoop runtime-checkable Protocol"


# ---------------------------------------------------------------------------
# Test 2: LOOP-05 — Disabled config short-circuits before any DB I/O
# ---------------------------------------------------------------------------


def test_disabled_config_returns_zero_without_db_io(monkeypatch):
    """LOOP-05: enabled=False returns 0 without touching any DB connection."""
    from src.autoloop.driver import AutoLoopDriver

    disabled_config = AutoLoopConfig(enabled=False)
    monkeypatch.setattr(
        "src.autoloop.driver.load_toml_config",
        MagicMock(return_value=disabled_config),
    )

    # Any DB I/O attempt must fail the test — patch connects to RAISE
    def _raise(*a, **kw):
        raise AssertionError("DB connection attempted despite enabled=False (LOOP-05 violated)")

    monkeypatch.setattr("src.autoloop.driver.timescale.connect", _raise)
    monkeypatch.setattr("src.autoloop.driver.milvus_db.connect", _raise)

    driver = AutoLoopDriver()
    result = driver.step()

    assert result == 0, "step() must return 0 when enabled=False"


# ---------------------------------------------------------------------------
# Test 3: LOOP-02 — Sequential ordering (validate then apply, per candidate)
# ---------------------------------------------------------------------------


def test_sequential_ordering_no_parallel_apply(monkeypatch):
    """LOOP-02: validate and apply interleave per candidate (not batched)."""
    from src.autoloop.driver import AutoLoopDriver
    from src.patch_engine import PatchEngine

    ck1 = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    ck2 = "hero_pos_rel=CO|n_players_active=3|pot_type=srp|street_class=postflop"
    ck3 = "hero_pos_rel=UTG|n_players_active=4|pot_type=srp|street_class=postflop"
    candidates = [_make_candidate(ck1), _make_candidate(ck2), _make_candidate(ck3)]

    config = AutoLoopConfig(
        enabled=True,
        tau_leak=0.3,
        min_observations=5,
        max_patches_per_run=5,
        n_hands_validation=10,
        bootstrap_resamples=10,
        ci_alpha=0.05,
    )
    monkeypatch.setattr("src.autoloop.driver.load_toml_config", MagicMock(return_value=config))
    monkeypatch.setattr("src.autoloop.driver._tsdb_dsn_from_env", MagicMock(return_value="dsn://test"))
    monkeypatch.setattr(
        "src.autoloop.driver._milvus_uri_from_env", MagicMock(return_value="http://test:51530")
    )

    mock_tsdb = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("session-abc",)
    mock_tsdb.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_tsdb.cursor.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("src.autoloop.driver.timescale.connect", MagicMock(return_value=mock_tsdb))
    monkeypatch.setattr("src.autoloop.driver.milvus_db.connect", MagicMock(return_value=MagicMock()))

    monkeypatch.setattr("src.autoloop.driver.select_patch_candidates", MagicMock(return_value=candidates))
    monkeypatch.setattr("src.autoloop.driver.ev_loss", MagicMock(return_value=0.5))

    call_order: list[str] = []

    def fake_gen(cluster_key, *, _tsdb_conn=None, _milvus=None):
        call_order.append(f"gen-{cluster_key[:15]}")
        return {"fold": 0.5, "call": 0.5}, [0.1] * 80

    def fake_validate(
        patch_spec, config, *, baseline_engine, sim_adapter, seed, _tsdb_conn=None, _milvus=None
    ):
        call_order.append(f"validate-{patch_spec.cluster_key[:15]}")
        return _make_validation_result(ci_low=0.1)

    applied_calls: list[str] = []

    def fake_apply(
        self_pe, patch, *, validation, pre_ev_loss=None, post_ev_loss=None, _tsdb_conn=None, _milvus=None
    ):
        applied_calls.append(patch.cluster_key[:15])
        call_order.append(f"apply-{patch.cluster_key[:15]}")
        return _make_patch_record(patch.cluster_key)

    monkeypatch.setattr("src.autoloop.driver.generate_candidate_action_dist", fake_gen)
    monkeypatch.setattr("src.autoloop.driver.validate", fake_validate)

    # Patch PatchEngine.apply on the class (receives self as first arg)
    monkeypatch.setattr(PatchEngine, "apply", fake_apply)

    driver = AutoLoopDriver(baseline_engine=MagicMock(), sim_adapter=MagicMock())
    result = driver.step()

    assert result == 3, f"Expected 3 applied, got {result}"

    # Verify interleaved ordering: gen-ck1, validate-ck1, apply-ck1, gen-ck2, validate-ck2, apply-ck2, ...
    for i, ck in enumerate([ck1, ck2, ck3]):
        prefix = ck[:15]
        base_idx = i * 3
        assert call_order[base_idx] == f"gen-{prefix}", (
            f"Expected gen at position {base_idx}, got {call_order[base_idx]}"
        )
        assert call_order[base_idx + 1] == f"validate-{prefix}", (
            f"Expected validate at position {base_idx + 1}"
        )
        assert call_order[base_idx + 2] == f"apply-{prefix}", f"Expected apply at position {base_idx + 2}"


# ---------------------------------------------------------------------------
# Test 4: LOOP-03 — Discard on negative ci_low
# ---------------------------------------------------------------------------


def test_discard_on_negative_ci_low(monkeypatch):
    """LOOP-03: patch discarded when ci_low < 0."""
    from src.patch_engine import PatchEngine

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidate = _make_candidate()
    patches["select_patch_candidates"].return_value = [candidate]
    patches["validate"].return_value = _make_validation_result(ci_low=-0.1)

    mock_apply = MagicMock()
    monkeypatch.setattr(PatchEngine, "apply", mock_apply)

    result = driver.step()

    assert result == 0, "driver should return 0 when ci_low < 0"
    mock_apply.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: Boundary — Discard on ci_low == 0.0 (strict > 0 gate)
# ---------------------------------------------------------------------------


def test_discard_on_zero_ci_low(monkeypatch):
    """LOOP-03 boundary: ci_low == 0.0 is rejected (strict > 0 required)."""
    from src.patch_engine import PatchEngine

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidate = _make_candidate()
    patches["select_patch_candidates"].return_value = [candidate]
    patches["validate"].return_value = _make_validation_result(ci_low=0.0)

    mock_apply = MagicMock()
    monkeypatch.setattr(PatchEngine, "apply", mock_apply)

    result = driver.step()

    assert result == 0, "driver should return 0 when ci_low == 0.0 (strict > 0)"
    mock_apply.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Apply called on positive ci_low
# ---------------------------------------------------------------------------


def test_apply_called_on_positive_ci_low(monkeypatch):
    """LOOP-03: PatchEngine.apply is called when ci_low > 0."""
    from src.patch_engine import PatchEngine

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidate = _make_candidate()
    patches["select_patch_candidates"].return_value = [candidate]
    patches["validate"].return_value = _make_validation_result(ci_low=0.01)

    mock_apply = MagicMock(return_value=_make_patch_record())
    monkeypatch.setattr(PatchEngine, "apply", mock_apply)

    result = driver.step()

    assert result == 1
    mock_apply.assert_called_once()


# ---------------------------------------------------------------------------
# Test 7: zero_hits validation discards patch
# ---------------------------------------------------------------------------


def test_zero_hits_validation_discards_patch(monkeypatch):
    """Driver discards patch when validate returns zero_hits=True."""
    from src.patch_engine import PatchEngine

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidate = _make_candidate()
    patches["select_patch_candidates"].return_value = [candidate]
    patches["validate"].return_value = _make_validation_result(ci_low=0.5, zero_hits=True)

    mock_apply = MagicMock()
    monkeypatch.setattr(PatchEngine, "apply", mock_apply)

    result = driver.step()

    assert result == 0
    mock_apply.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: LOOP-06 — source='autoloop' in PatchSpec
# ---------------------------------------------------------------------------


def test_loop_06_source_autoloop_in_patchspec(monkeypatch):
    """LOOP-06: PatchSpec passed to PatchEngine.apply has source='autoloop'."""
    from src.patch_engine import PatchEngine

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidate = _make_candidate()
    patches["select_patch_candidates"].return_value = [candidate]
    patches["validate"].return_value = _make_validation_result(ci_low=0.1)

    captured_specs: list[Any] = []

    def capture_apply(
        self_pe, patch, *, validation, pre_ev_loss=None, post_ev_loss=None, _tsdb_conn=None, _milvus=None
    ):
        captured_specs.append(patch)
        return _make_patch_record()

    monkeypatch.setattr(PatchEngine, "apply", capture_apply)

    driver.step()

    assert len(captured_specs) == 1
    assert captured_specs[0].source == "autoloop", (
        f"PatchSpec.source must be 'autoloop', got {captured_specs[0].source!r}"
    )


# ---------------------------------------------------------------------------
# Test 9: Apply failure continues to next candidate
# ---------------------------------------------------------------------------


def test_apply_failure_continues_to_next_candidate(monkeypatch):
    """When PatchEngine.apply raises PatchConflictError for candidate 2, candidates 1 and 3 succeed."""
    from src.patch_engine import PatchEngine

    ck1 = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    ck2 = "hero_pos_rel=CO|n_players_active=3|pot_type=srp|street_class=postflop"
    ck3 = "hero_pos_rel=UTG|n_players_active=4|pot_type=srp|street_class=postflop"

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidates = [_make_candidate(ck1), _make_candidate(ck2), _make_candidate(ck3)]
    patches["select_patch_candidates"].return_value = candidates
    patches["validate"].return_value = _make_validation_result(ci_low=0.1)

    call_count = [0]

    def apply_with_failure(
        self_pe, patch, *, validation, pre_ev_loss=None, post_ev_loss=None, _tsdb_conn=None, _milvus=None
    ):
        call_count[0] += 1
        if patch.cluster_key == ck2:
            raise PatchConflictError(f"ERR-04: conflict for {ck2}")
        return _make_patch_record(patch.cluster_key)

    monkeypatch.setattr(PatchEngine, "apply", apply_with_failure)

    result = driver.step()

    assert result == 2, f"Expected 2 applied (1 and 3), got {result}"
    assert call_count[0] == 3, "apply must be attempted for all 3 candidates"


# ---------------------------------------------------------------------------
# Test 10: Rebalance NoStrategyError skips candidate
# ---------------------------------------------------------------------------


def test_rebalance_no_strategy_skips_candidate(monkeypatch):
    """generate_candidate_action_dist raises NoStrategyError -> candidate skipped."""
    from src.patch_engine import PatchEngine

    ck1 = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    ck2 = "hero_pos_rel=CO|n_players_active=3|pot_type=srp|street_class=postflop"

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidates = [_make_candidate(ck1), _make_candidate(ck2)]
    patches["select_patch_candidates"].return_value = candidates
    patches["validate"].return_value = _make_validation_result(ci_low=0.1)

    def gen_with_error(cluster_key, *, _tsdb_conn=None, _milvus=None):
        if cluster_key == ck1:
            raise NoStrategyError(f"no reps for {ck1}")
        return {"fold": 0.5, "call": 0.5}, [0.1] * 80

    patches["generate_candidate_action_dist"].side_effect = gen_with_error

    mock_apply = MagicMock(return_value=_make_patch_record(ck2))
    monkeypatch.setattr(PatchEngine, "apply", mock_apply)

    result = driver.step()

    assert result == 1, f"Expected 1 (ck2 only), got {result}"
    assert mock_apply.call_count == 1


# ---------------------------------------------------------------------------
# Test 11: VALN-03 — Validation exception does not apply patch
# ---------------------------------------------------------------------------


def test_validation_exception_does_not_apply(monkeypatch):
    """VALN-03: validate raising RuntimeError prevents apply; subsequent candidates succeed."""
    from src.patch_engine import PatchEngine

    ck1 = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    ck2 = "hero_pos_rel=CO|n_players_active=3|pot_type=srp|street_class=postflop"

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidates = [_make_candidate(ck1), _make_candidate(ck2)]
    patches["select_patch_candidates"].return_value = candidates

    def validate_with_error(
        patch_spec, config, *, baseline_engine, sim_adapter, seed, _tsdb_conn=None, _milvus=None
    ):
        if patch_spec.cluster_key == ck1:
            raise RuntimeError("sim error for ck1")
        return _make_validation_result(ci_low=0.1)

    patches["validate"].side_effect = validate_with_error

    mock_apply = MagicMock(return_value=_make_patch_record(ck2))
    monkeypatch.setattr(PatchEngine, "apply", mock_apply)

    result = driver.step()

    assert result == 1, f"Expected 1 (ck2 only), got {result}"
    mock_apply.assert_called_once()


# ---------------------------------------------------------------------------
# Test 12: No recent session returns 0
# ---------------------------------------------------------------------------


def test_no_recent_session_returns_zero(monkeypatch):
    """When _last_record_session_id returns None, driver returns 0 immediately."""
    from src.autoloop.driver import AutoLoopDriver

    config = AutoLoopConfig(enabled=True)
    monkeypatch.setattr("src.autoloop.driver.load_toml_config", MagicMock(return_value=config))
    monkeypatch.setattr("src.autoloop.driver._tsdb_dsn_from_env", MagicMock(return_value="dsn://test"))
    monkeypatch.setattr(
        "src.autoloop.driver._milvus_uri_from_env", MagicMock(return_value="http://test:51530")
    )

    mock_tsdb = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None  # No session
    mock_tsdb.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_tsdb.cursor.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("src.autoloop.driver.timescale.connect", MagicMock(return_value=mock_tsdb))
    monkeypatch.setattr("src.autoloop.driver.milvus_db.connect", MagicMock(return_value=MagicMock()))

    mock_select = MagicMock()
    monkeypatch.setattr("src.autoloop.driver.select_patch_candidates", mock_select)

    driver = AutoLoopDriver(baseline_engine=MagicMock(), sim_adapter=MagicMock())
    result = driver.step()

    assert result == 0
    mock_select.assert_not_called()


# ---------------------------------------------------------------------------
# Test 13: LOOP-04 — Config loaded from default path
# ---------------------------------------------------------------------------


def test_config_loaded_from_default_path(monkeypatch):
    """LOOP-04: load_toml_config called with Path('config/autoloop.toml')."""
    from src.autoloop.driver import AutoLoopDriver

    config = AutoLoopConfig(enabled=False)
    mock_load = MagicMock(return_value=config)
    monkeypatch.setattr("src.autoloop.driver.load_toml_config", mock_load)

    # enabled=False — no DB connections needed, but patch helpers defensively
    monkeypatch.setattr("src.autoloop.driver._tsdb_dsn_from_env", MagicMock(return_value="dsn://test"))
    monkeypatch.setattr(
        "src.autoloop.driver._milvus_uri_from_env", MagicMock(return_value="http://test:51530")
    )
    monkeypatch.setattr("src.autoloop.driver.timescale.connect", MagicMock())
    monkeypatch.setattr("src.autoloop.driver.milvus_db.connect", MagicMock())

    driver = AutoLoopDriver()
    driver.step()

    mock_load.assert_called_once()
    call_args = mock_load.call_args
    path_arg = call_args[0][0] if call_args[0] else call_args[1].get("path")
    assert path_arg == Path("config/autoloop.toml"), (
        f"Expected Path('config/autoloop.toml'), got {path_arg!r}"
    )


# ---------------------------------------------------------------------------
# Test 14: max_patches_per_run caps candidates
# ---------------------------------------------------------------------------


def test_loop_04_max_patches_per_run_caps_candidates(monkeypatch):
    """select_patch_candidates is called with the config that caps to max_patches_per_run."""
    from src.patch_engine import PatchEngine

    config = AutoLoopConfig(
        enabled=True,
        tau_leak=0.3,
        min_observations=5,
        max_patches_per_run=3,
        n_hands_validation=10,
        bootstrap_resamples=10,
        ci_alpha=0.05,
    )

    driver, patches = _make_driver_with_mocks(monkeypatch, config=config)

    ck1 = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
    ck2 = "hero_pos_rel=CO|n_players_active=3|pot_type=srp|street_class=postflop"
    ck3 = "hero_pos_rel=UTG|n_players_active=4|pot_type=srp|street_class=postflop"
    # The leak_detector already caps to max_patches_per_run=3 — mock returns exactly 3
    candidates = [_make_candidate(ck1), _make_candidate(ck2), _make_candidate(ck3)]
    patches["select_patch_candidates"].return_value = candidates
    patches["validate"].return_value = _make_validation_result(ci_low=0.1)

    mock_apply = MagicMock(
        side_effect=lambda p, *, validation, pre_ev_loss=None, post_ev_loss=None, _tsdb_conn=None, _milvus=None: (
            _make_patch_record(p.cluster_key)
        )
    )
    monkeypatch.setattr(PatchEngine, "apply", mock_apply)

    result = driver.step()

    assert result == 3
    assert mock_apply.call_count == 3


# ---------------------------------------------------------------------------
# Test 15: PatchRecord returned from apply carries patch_id in log
# ---------------------------------------------------------------------------


def test_record_returned_from_apply_carries_patch_id(monkeypatch):
    """PatchEngine.apply returns a PatchRecord; driver processes and completes without error."""
    from src.patch_engine import PatchEngine

    patch_id = uuid.uuid4()
    expected_record = PatchRecord(
        patch_id=patch_id,
        ts="2026-01-01T00:00:00+00:00",
        status="applied",
        cluster_key="hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop",
        new_node_id=uuid.uuid4(),
        prev_node_id=None,
    )

    driver, patches = _make_driver_with_mocks(monkeypatch)
    candidate = _make_candidate()
    patches["select_patch_candidates"].return_value = [candidate]
    patches["validate"].return_value = _make_validation_result(ci_low=0.1)

    mock_apply = MagicMock(return_value=expected_record)
    monkeypatch.setattr(PatchEngine, "apply", mock_apply)

    result = driver.step()

    assert result == 1
    mock_apply.assert_called_once()
    # The returned record should have the patch_id we set
    actual_record = mock_apply.return_value
    assert actual_record.patch_id == patch_id


# Version bump fires only when applied > 0


def _make_step_driver_with_mocks(monkeypatch, *, candidates):
    """Self-contained mock harness for the version-bump tests.

    Returns the driver plus the mock tsdb_conn so the bump test can assert
    the snapshot received the injected conn.
    """
    from src.autoloop.driver import AutoLoopDriver

    config = AutoLoopConfig(
        enabled=True,
        tau_leak=0.3,
        min_observations=5,
        max_patches_per_run=5,
        n_hands_validation=10,
        bootstrap_resamples=10,
        ci_alpha=0.05,
    )
    monkeypatch.setattr("src.autoloop.driver.load_toml_config", MagicMock(return_value=config))
    monkeypatch.setattr("src.autoloop.driver._tsdb_dsn_from_env", MagicMock(return_value="dsn://test"))

    mock_tsdb = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("session-bump",)
    mock_tsdb.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_tsdb.cursor.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("src.autoloop.driver.timescale.connect", MagicMock(return_value=mock_tsdb))
    monkeypatch.setattr("src.autoloop.driver.milvus_db.connect_from_env", MagicMock(return_value=MagicMock()))

    monkeypatch.setattr("src.autoloop.driver.select_patch_candidates", MagicMock(return_value=candidates))
    monkeypatch.setattr(
        "src.autoloop.driver.generate_candidate_action_dist",
        MagicMock(return_value=({"fold": 0.5, "call": 0.5}, [0.1] * 80, "dp_test")),
    )
    monkeypatch.setattr("src.autoloop.driver.ev_loss", MagicMock(return_value=0.5))

    driver = AutoLoopDriver(baseline_engine=MagicMock(), sim_adapter=MagicMock())
    return driver, mock_tsdb


def test_version_bump_fires_when_applied(monkeypatch):
    """A step that applies >=1 patch snapshots a corpus version (source='autoloop')."""
    from src.autoloop.driver import AutoLoopDriver  # noqa: F401
    from src.patch_engine import PatchEngine

    driver, mock_tsdb = _make_step_driver_with_mocks(monkeypatch, candidates=[_make_candidate()])
    monkeypatch.setattr(
        "src.autoloop.driver.validate", MagicMock(return_value=_make_validation_result(ci_low=0.1))
    )
    monkeypatch.setattr(PatchEngine, "apply", MagicMock(return_value=_make_patch_record()))

    mock_snapshot = MagicMock(return_value={"version": 7, "cutoff_ts": 123})
    monkeypatch.setattr("src.autoloop.driver.corpus_versions.snapshot", mock_snapshot)

    applied = driver.step()

    assert applied == 1
    mock_snapshot.assert_called_once()
    _, kwargs = mock_snapshot.call_args
    assert kwargs["source"] == "autoloop"
    assert kwargs["_tsdb_conn"] is mock_tsdb


def test_version_bump_commits_via_transaction(monkeypatch):
    """End-to-end with the REAL corpus_versions.snapshot: the autoloop step's
    snapshot wraps its INSERT in conn.transaction() so the injected non-autocommit
    conn commits instead of silently rolling back on close."""
    from src.patch_engine import PatchEngine

    driver, mock_tsdb = _make_step_driver_with_mocks(monkeypatch, candidates=[_make_candidate()])

    tx_entered = {"value": False}
    tx_cm = MagicMock()
    tx_cm.__enter__ = lambda _self: (tx_entered.__setitem__("value", True), tx_cm)[1]
    tx_cm.__exit__ = lambda *a: False
    mock_tsdb.transaction.return_value = tx_cm

    insert_cursor = MagicMock()
    insert_cursor.description = [("version",), ("cutoff_ts",)]
    insert_cursor.fetchone.return_value = (7, 123)
    mock_tsdb.cursor.return_value.__enter__ = MagicMock(return_value=insert_cursor)

    monkeypatch.setattr(
        "src.autoloop.driver.validate", MagicMock(return_value=_make_validation_result(ci_low=0.1))
    )
    monkeypatch.setattr(PatchEngine, "apply", MagicMock(return_value=_make_patch_record()))

    applied = driver.step()

    assert applied == 1
    assert tx_entered["value"], "snapshot must commit its INSERT inside conn.transaction()"


def test_version_bump_skipped_when_zero_applied(monkeypatch):
    """A step that applies no patches does NOT snapshot a corpus version."""
    from src.patch_engine import PatchEngine

    driver, _ = _make_step_driver_with_mocks(monkeypatch, candidates=[_make_candidate()])
    # ci_low <= 0 → discard → applied == 0
    monkeypatch.setattr(
        "src.autoloop.driver.validate", MagicMock(return_value=_make_validation_result(ci_low=-0.1))
    )
    monkeypatch.setattr(PatchEngine, "apply", MagicMock())

    mock_snapshot = MagicMock()
    monkeypatch.setattr("src.autoloop.driver.corpus_versions.snapshot", mock_snapshot)

    applied = driver.step()

    assert applied == 0
    mock_snapshot.assert_not_called()
