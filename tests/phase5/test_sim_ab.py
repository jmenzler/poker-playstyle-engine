"""tests/phase5/test_sim_ab.py — Tests for sim A/B validation (Plan 04).

Requirements covered:
- VALN-01: A/B runs two strategies over same RNG seed; identical game sequences
- VALN-02: ValidationResult includes seed, ev_loss_delta, n_hands, confidence_interval,
           n_cluster_hits, zero_hits; stored with patch record as JSONB
- VALN-03: auto-loop apply path calls A/B before writing; validation errors → patch not applied

Test groups:
  Task 2 — DryRunObservationWriter + bootstrap_ci (6 tests)
  Task 3 — validate() orchestrator (5 tests)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

TARGET_KEY = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"


# ===========================================================================
# Task 2: DryRunObservationWriter + bootstrap_ci
# ===========================================================================


class TestDryRunObservationWriter:
    def test_accepts_same_kwargs_as_observation_writer(self):
        """DryRunObservationWriter.record() accepts all kwargs ObservationWriter does."""
        from src.validation.sim_ab import DryRunObservationWriter

        w = DryRunObservationWriter("session-1")
        # Should not raise with all supported kwargs — these are exactly the kwargs
        # run_record_session always passes (src/sim/harness.py).
        w.record(
            cluster_key="k=v",
            embedding=[0.1, 0.2],
            action_taken="call",
            flagged_sparse=False,
            max_neighbor_distance=None,
            hand_id="sess_h0",
            decision_id="sess_h0_dp0",
            felt_snapshot={"pot": 1.5},
        )
        # Also works with defaults (no kwargs)
        w.record(cluster_key="k=v", embedding=[0.3], action_taken="fold")
        assert len(w.rows) == 2

    def test_signature_matches_observation_writer_record(self):
        """DryRunObservationWriter.record() accepts every param ObservationWriter.record
        does, so the harness can swap writers without a TypeError on any call site.
        """
        import inspect

        from src.sim.observation_writer import ObservationWriter
        from src.validation.sim_ab import DryRunObservationWriter

        prod = inspect.signature(ObservationWriter.record).parameters
        dry = inspect.signature(DryRunObservationWriter.record).parameters
        missing = set(prod) - set(dry)
        assert not missing, f"DryRunObservationWriter.record missing params: {missing}"

    def test_close_returns_row_count_and_rows_accessible(self):
        """close() returns the count of buffered rows; rows available via .rows property."""
        from src.validation.sim_ab import DryRunObservationWriter

        w = DryRunObservationWriter("sess-2")
        w.record("ck", [1.0], "call")
        w.record("ck", [2.0], "bet_50")
        w.record("other", [3.0], "fold")

        count = w.close()
        assert count == 3
        assert len(w.rows) == 3

    def test_never_opens_db_connection(self, monkeypatch: pytest.MonkeyPatch):
        """DryRunObservationWriter never calls timescale.connect or milvus.connect."""
        from src.validation.sim_ab import DryRunObservationWriter

        timescale_calls: list = []
        milvus_calls: list = []
        monkeypatch.setattr("src.db.timescale.connect", lambda *a, **kw: timescale_calls.append(1))
        monkeypatch.setattr("src.db.milvus.connect", lambda *a, **kw: milvus_calls.append(1))

        w = DryRunObservationWriter("sess-3")
        w.record("ck", [0.1], "check")
        w.close()

        assert not timescale_calls, "timescale.connect called unexpectedly"
        assert not milvus_calls, "milvus.connect called unexpectedly"

    def test_context_manager_protocol(self):
        """DryRunObservationWriter works as a context manager (__enter__ / __exit__)."""
        from src.validation.sim_ab import DryRunObservationWriter

        with DryRunObservationWriter("sess-4") as w:
            w.record("ck", [1.0], "fold")
            assert len(w.rows) == 1
        # After exit, rows are still accessible
        assert len(w.rows) == 1


class TestBootstrapCI:
    def test_empty_input_returns_zero_zero(self):
        """bootstrap_ci on empty list returns (0.0, 0.0) — Pitfall 5 guard."""
        from src.validation.sim_ab import bootstrap_ci

        result = bootstrap_ci([], n_resamples=100, alpha=0.05)
        assert result == (0.0, 0.0)

    def test_seeded_rng_is_deterministic(self):
        """Same seed produces identical (ci_low, ci_high) across two calls."""
        from src.validation.sim_ab import bootstrap_ci

        deltas = [0.1, 0.2, 0.3, -0.1, 0.15]
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        r1 = bootstrap_ci(deltas, n_resamples=50, alpha=0.05, rng=rng1)
        r2 = bootstrap_ci(deltas, n_resamples=50, alpha=0.05, rng=rng2)
        assert r1 == r2, f"Expected deterministic output, got {r1} vs {r2}"

    def test_all_positive_deltas_ci_low_positive(self):
        """bootstrap_ci on all-positive deltas returns ci_low > 0."""
        from src.validation.sim_ab import bootstrap_ci

        deltas = [1.0] * 50  # all same positive value
        ci_low, ci_high = bootstrap_ci(deltas, n_resamples=100, alpha=0.05, rng=np.random.default_rng(0))
        assert ci_low > 0.0, f"Expected ci_low > 0 for all-positive deltas, got {ci_low}"
        assert ci_high > 0.0

    def test_all_zero_deltas_returns_zeros(self):
        """bootstrap_ci on all-zero deltas returns (0.0, 0.0)."""
        from src.validation.sim_ab import bootstrap_ci

        deltas = [0.0] * 20
        ci_low, ci_high = bootstrap_ci(deltas, n_resamples=100, alpha=0.05, rng=np.random.default_rng(1))
        assert ci_low == 0.0
        assert ci_high == 0.0

    def test_returns_tuple_of_two_floats(self):
        """bootstrap_ci returns a 2-tuple of floats."""
        from src.validation.sim_ab import bootstrap_ci

        result = bootstrap_ci([0.5, -0.1, 0.3], n_resamples=10, alpha=0.05)
        assert len(result) == 2
        ci_low, ci_high = result
        assert isinstance(ci_low, float)
        assert isinstance(ci_high, float)


# ===========================================================================
# Task 3: validate() orchestrator
# ===========================================================================


def _make_patch_spec(cluster_key: str = TARGET_KEY) -> object:
    """Return a PatchSpec-shaped object for testing.

    Uses a SimpleNamespace to avoid depending on patch_engine.py being
    available (parallel wave-2 plan). All fields required by validate().
    """
    import types

    return types.SimpleNamespace(
        cluster_key=cluster_key,
        action_dist={"call": 0.9, "fold": 0.1},
    )


def _make_autoloop_config(n_hands: int = 4, resamples: int = 10) -> object:
    """Return an AutoLoopConfig-shaped object for testing."""
    from src._config import AutoLoopConfig

    return AutoLoopConfig(
        enabled=True,
        tau_leak=0.3,
        min_observations=5,
        max_patches_per_run=1,
        n_hands_validation=n_hands,
        bootstrap_resamples=resamples,
        ci_alpha=0.05,
    )


def _make_mock_run_record_session(writer_rows_by_call: list[list[dict]]):
    """Return a mock run_record_session that populates writer._rows for each call.

    writer_rows_by_call: list of row lists, one per call.
    First call gets writer_rows_by_call[0], second gets [1].
    """
    call_count = [0]

    def mock_run(*, adapter, engine, writer, session_seed, n_hands, session_id=None, **kwargs):
        idx = call_count[0]
        if idx < len(writer_rows_by_call):
            writer._rows.extend(writer_rows_by_call[idx])
        call_count[0] += 1
        return {"total_hands": n_hands, "total_decisions": n_hands, "session_id": session_id or "mock"}

    return mock_run


class TestValidate:
    """Task 3: validate() orchestrator tests."""

    def _make_milvus_client_with_q(self, q_dist: dict[str, float]) -> MagicMock:
        """Return a mock milvus client whose search returns neighbors matching q_dist.

        We fabricate search results that produce the desired blend by returning a
        single neighbor for each action with the right probability.
        """
        client = MagicMock()
        # We'll have _expected_action_dist return q_dist via a side-effect.
        # Easier: patch _expected_action_dist directly in the test methods.
        return client

    def test_validate_returns_validation_result_with_all_required_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """VALN-02: returned object has seed, ev_loss_delta, n_hands, confidence_interval,
        n_cluster_hits, zero_hits.
        """
        from src.validation.sim_ab import validate

        patch_spec = _make_patch_spec()
        config = _make_autoloop_config()

        baseline_rows = [
            {"cluster_key": TARGET_KEY, "embedding": [0.1], "action_taken": "fold"},
        ]
        candidate_rows = [
            {"cluster_key": TARGET_KEY, "embedding": [0.1], "action_taken": "call"},
        ]

        mock_session = _make_mock_run_record_session([baseline_rows, candidate_rows])
        monkeypatch.setattr("src.validation.sim_ab.run_record_session", mock_session)

        # Patch _expected_action_dist so we don't need a real Milvus client
        q_dist = {"call": 0.9, "fold": 0.1}
        monkeypatch.setattr(
            "src.validation.sim_ab._expected_action_dist",
            lambda client, embedding, collection: q_dist,
        )

        mock_milvus = MagicMock()
        result = validate(
            patch_spec,
            config,
            baseline_engine=MagicMock(),
            sim_adapter=MagicMock(),
            seed=42,
            _milvus=mock_milvus,
        )

        # Check all required VALN-02 fields exist
        assert hasattr(result, "seed")
        assert hasattr(result, "ev_loss_delta")
        assert hasattr(result, "n_hands")
        assert hasattr(result, "confidence_interval")
        assert hasattr(result, "n_cluster_hits")
        assert hasattr(result, "zero_hits")
        assert result.seed == 42
        assert result.n_hands == config.n_hands_validation
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2

    def test_validate_paired_seed_both_runs_use_same_seed(self, monkeypatch: pytest.MonkeyPatch):
        """VALN-01: both baseline and candidate runs use the same session_seed."""
        from src.validation.sim_ab import validate

        seen_seeds: list[int] = []

        def capturing_session(*, adapter, engine, writer, session_seed, n_hands, **kwargs):
            seen_seeds.append(session_seed)
            # Populate writer with target-key rows so validate doesn't hit zero_hits
            writer._rows.extend(
                [
                    {"cluster_key": TARGET_KEY, "embedding": [0.1], "action_taken": "call"},
                ]
            )
            return {"total_hands": n_hands, "total_decisions": n_hands}

        monkeypatch.setattr("src.validation.sim_ab.run_record_session", capturing_session)
        monkeypatch.setattr(
            "src.validation.sim_ab._expected_action_dist",
            lambda *a, **kw: {"call": 1.0},
        )

        patch_spec = _make_patch_spec()
        config = _make_autoloop_config()
        validate(
            patch_spec,
            config,
            baseline_engine=MagicMock(),
            sim_adapter=MagicMock(),
            seed=777,
            _milvus=MagicMock(),
        )

        assert len(seen_seeds) == 2, f"Expected 2 run_record_session calls, got {len(seen_seeds)}"
        assert seen_seeds[0] == 777, f"Baseline seed should be 777, got {seen_seeds[0]}"
        assert seen_seeds[1] == 777, f"Candidate seed should be 777, got {seen_seeds[1]}"

    def test_validate_uses_dry_run_writer_no_tsdb_writes(self, monkeypatch: pytest.MonkeyPatch):
        """VALN-01 / Pitfall 1: both writers are DryRunObservationWriter, no cursor() calls."""
        from src.validation.sim_ab import DryRunObservationWriter, validate

        captured_writers: list = []

        def capturing_session(*, adapter, engine, writer, session_seed, n_hands, **kwargs):
            captured_writers.append(writer)
            writer._rows.extend(
                [
                    {"cluster_key": TARGET_KEY, "embedding": [0.1], "action_taken": "fold"},
                ]
            )
            return {}

        monkeypatch.setattr("src.validation.sim_ab.run_record_session", capturing_session)
        monkeypatch.setattr(
            "src.validation.sim_ab._expected_action_dist",
            lambda *a, **kw: {"fold": 1.0},
        )

        mock_tsdb = MagicMock()
        patch_spec = _make_patch_spec()
        config = _make_autoloop_config()

        validate(
            patch_spec,
            config,
            baseline_engine=MagicMock(),
            sim_adapter=MagicMock(),
            seed=1,
            _milvus=MagicMock(),
        )

        assert len(captured_writers) == 2
        for w in captured_writers:
            assert isinstance(w, DryRunObservationWriter), f"Expected DryRunObservationWriter, got {type(w)}"
        # mock_tsdb.cursor should never be called
        mock_tsdb.cursor.assert_not_called()

    def test_validate_zero_hits_returns_zero_hits_true(self, monkeypatch: pytest.MonkeyPatch):
        """Pitfall 5: zero target-cluster hits → ValidationResult(zero_hits=True,
        ev_loss_delta=0.0, confidence_interval=(0.0, 0.0)).
        """
        from src.validation.sim_ab import validate

        def no_hits_session(*, adapter, engine, writer, session_seed, n_hands, **kwargs):
            # Rows with a DIFFERENT cluster_key — no hits on target
            writer._rows.extend(
                [
                    {"cluster_key": "other_key", "embedding": [0.1], "action_taken": "call"},
                ]
            )
            return {}

        monkeypatch.setattr("src.validation.sim_ab.run_record_session", no_hits_session)

        patch_spec = _make_patch_spec()
        config = _make_autoloop_config()

        result = validate(
            patch_spec,
            config,
            baseline_engine=MagicMock(),
            sim_adapter=MagicMock(),
            seed=0,
        )

        assert result.zero_hits is True
        assert result.ev_loss_delta == 0.0
        assert result.confidence_interval == (0.0, 0.0)
        assert result.n_cluster_hits == 0

    def test_validate_propagates_exception_from_sim_run(self, monkeypatch: pytest.MonkeyPatch):
        """VALN-03: exceptions from run_record_session are NOT swallowed."""
        from src.validation.sim_ab import validate

        def exploding_session(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("src.validation.sim_ab.run_record_session", exploding_session)

        patch_spec = _make_patch_spec()
        config = _make_autoloop_config()

        with pytest.raises(RuntimeError, match="boom"):
            validate(
                patch_spec,
                config,
                baseline_engine=MagicMock(),
                sim_adapter=MagicMock(),
                seed=0,
            )

    def test_validate_positive_delta_when_candidate_aligned_to_q(self, monkeypatch: pytest.MonkeyPatch):
        """Algorithmic sanity: candidate aligned to Q → ev_loss_delta > 0 and ci_low > 0."""
        from src.validation.sim_ab import validate

        # Q distribution strongly favors 'call'
        q_dist = {"call": 0.9, "fold": 0.05, "check": 0.05}

        def aligning_session(*, adapter, engine, writer, session_seed, n_hands, **kwargs):
            # Baseline takes 'fold' (low Q probability)
            # Candidate takes 'call' (high Q probability)
            # Distinguish by checking writer session_id prefix
            if "baseline" in getattr(writer, "_session_id", ""):
                action = "fold"
            else:
                action = "call"
            writer._rows.extend(
                [
                    {"cluster_key": TARGET_KEY, "embedding": [0.1, 0.2], "action_taken": action}
                    for _ in range(5)
                ]
            )
            return {}

        monkeypatch.setattr("src.validation.sim_ab.run_record_session", aligning_session)
        monkeypatch.setattr(
            "src.validation.sim_ab._expected_action_dist",
            lambda *a, **kw: q_dist,
        )

        patch_spec = _make_patch_spec()
        # Use many resamples for stable CI
        config = _make_autoloop_config(n_hands=5, resamples=200)

        result = validate(
            patch_spec,
            config,
            baseline_engine=MagicMock(),
            sim_adapter=MagicMock(),
            seed=42,
            _milvus=MagicMock(),
        )

        assert result.ev_loss_delta > 0, f"Expected positive delta, got {result.ev_loss_delta}"
        ci_low, ci_high = result.confidence_interval
        assert ci_low > 0, f"Expected ci_low > 0, got {ci_low}"
        assert ci_high >= ci_low  # degenerate case (constant deltas) → ci_low == ci_high; valid

    def test_validate_does_not_close_injected_milvus(self, monkeypatch: pytest.MonkeyPatch):
        """An injected _milvus client is owned by the caller and must never be closed by validate()."""
        from src.validation.sim_ab import validate

        def hit_session(*, adapter, engine, writer, session_seed, n_hands, **kwargs):
            writer._rows.extend([{"cluster_key": TARGET_KEY, "embedding": [0.1], "action_taken": "fold"}])
            return {}

        monkeypatch.setattr("src.validation.sim_ab.run_record_session", hit_session)
        monkeypatch.setattr("src.validation.sim_ab._expected_action_dist", lambda *a, **kw: {"fold": 1.0})

        injected = MagicMock()
        validate(
            _make_patch_spec(),
            _make_autoloop_config(),
            baseline_engine=MagicMock(),
            sim_adapter=MagicMock(),
            seed=0,
            _milvus=injected,
        )
        injected.close.assert_not_called()

    def test_validate_closes_owned_milvus_client(self, monkeypatch: pytest.MonkeyPatch):
        """When validate() opens its own client (no _milvus), it must close it (no leak)."""
        from src.validation.sim_ab import validate

        def hit_session(*, adapter, engine, writer, session_seed, n_hands, **kwargs):
            writer._rows.extend([{"cluster_key": TARGET_KEY, "embedding": [0.1], "action_taken": "fold"}])
            return {}

        monkeypatch.setattr("src.validation.sim_ab.run_record_session", hit_session)
        monkeypatch.setattr("src.validation.sim_ab._expected_action_dist", lambda *a, **kw: {"fold": 1.0})

        owned = MagicMock()
        monkeypatch.setattr("src.db.milvus.connect_from_env", lambda: owned)
        validate(
            _make_patch_spec(),
            _make_autoloop_config(),
            baseline_engine=MagicMock(),
            sim_adapter=MagicMock(),
            seed=0,
        )
        owned.close.assert_called_once()
