# long-ok-file
# rot-allow-file
"""Real felt_snapshot shape tests for BUGs 1-3, 5 (no synthetic keys).  long-ok

Real keys: street, board_cards, pot_size_bb, hero_position, action_sequence,
hero_hole_cards, hero_bet_size_bb, effective_stack_bb, hero_facing_bet_bb,
opponents_remaining, prior_street_aggressor.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

_REAL_3BET_FELT: dict = {
    "street": "flop",
    "board_cards": ["Jc", "7s", "2h"],
    "pot_size_bb": 27.0,
    "hero_position": "BB",
    "action_sequence": [
        "UTG:fold",
        "MP:fold",
        "CO:fold",
        "BTN:open_2.5",
        "SB:fold",
        "BB:3bet_9",
        "BTN:call",
        "BB:check",
        "BTN:bet_5",
    ],
    "hero_hole_cards": ["Ah", "Kd"],
    "hero_bet_size_bb": 0.0,
    "effective_stack_bb": 91.0,
    "hero_facing_bet_bb": 5.0,
    "opponents_remaining": 1,
    "prior_street_aggressor": "BB",
}

_REAL_4BET_FELT: dict = {
    "street": "flop",
    "board_cards": ["Ac", "7d", "3h"],
    "pot_size_bb": 50.0,
    "hero_position": "CO",
    "action_sequence": [
        "UTG:fold",
        "MP:fold",
        "CO:open_2.5",
        "BTN:fold",
        "SB:fold",
        "BB:3bet_9",
        "CO:4bet_22",
        "BB:call",
        "CO:bet_15",
    ],
    "hero_hole_cards": ["As", "Ks"],
    "hero_bet_size_bb": 15.0,
    "effective_stack_bb": 78.0,
    "hero_facing_bet_bb": 0.0,
    "opponents_remaining": 1,
    "prior_street_aggressor": "CO",
}

_REAL_SRP_FELT: dict = {
    "street": "flop",
    "board_cards": ["9h", "6d", "2c"],
    "pot_size_bb": 6.0,
    "hero_position": "BTN",
    "action_sequence": [
        "UTG:fold",
        "MP:fold",
        "CO:fold",
        "BTN:open_2.5",
        "SB:fold",
        "BB:call",
        "BB:check",
    ],
    "hero_hole_cards": ["Kh", "Qh"],
    "hero_bet_size_bb": 0.0,
    "effective_stack_bb": 97.5,
    "hero_facing_bet_bb": 0.0,
    "opponents_remaining": 1,
    "prior_street_aggressor": "BTN",
}


def _make_obs_row(felt: dict, cluster_key: str = "") -> dict:
    return {
        "obs_id": "obs-real-001",
        "cluster_key": cluster_key,
        "felt_snapshot": felt,
        "max_neighbor_distance": 0.75,
        "cluster_freq": 10,
        "ev_loss": 0.5,
        "priority_score": 3.75,
    }


# ---------------------------------------------------------------------------
# BUG 1: _build_spot derives pot_type / villain_pos / preflop_action_seq
# ---------------------------------------------------------------------------


class TestBuildSpotRealFelt:
    """_build_spot must derive pot_type, villain_pos, preflop_action_seq from real shape."""

    def _get_cfg(self):
        from pathlib import Path

        from src._config import AutoLoopConfig, load_toml_config

        return load_toml_config(Path("config/autoloop.toml"), AutoLoopConfig).solver_queue

    def test_3bet_pot_type_from_cluster_key(self):
        """pot_type='3bet' derived from cluster_key, not felt_snapshot."""
        from src.solver.queue_driver import _build_spot

        cluster_key = "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
        obs = _make_obs_row(_REAL_3BET_FELT, cluster_key)
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        assert ctx["pot_type"] == "3bet", (
            f"Expected pot_type='3bet' from cluster_key, got {ctx['pot_type']!r}. "
            "felt_snapshot has NO pot_type key — must parse cluster_key."
        )

    def test_3bet_pot_type_fallback_from_action_sequence(self):
        """When cluster_key lacks pot_type token, derive from action_sequence."""
        from src.solver.queue_driver import _build_spot

        obs = _make_obs_row(_REAL_3BET_FELT, cluster_key="hero_pos_rel=OOP|n_players_active=2")
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        assert ctx["pot_type"] == "3bet", (
            f"Expected pot_type='3bet' from action_sequence, got {ctx['pot_type']!r}."
        )

    def test_4bet_pot_type_from_cluster_key(self):
        """pot_type='4bet' derived from cluster_key."""
        from src.solver.queue_driver import _build_spot

        cluster_key = "hero_pos_rel=IP|n_players_active=2|pot_type=4bet|street_class=postflop"
        obs = _make_obs_row(_REAL_4BET_FELT, cluster_key)
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        assert ctx["pot_type"] == "4bet", (
            f"Expected pot_type='4bet' from cluster_key, got {ctx['pot_type']!r}."
        )

    def test_srp_pot_type_from_action_sequence_fallback(self):
        """SRP pot derives correctly from action_sequence when no cluster_key pot_type."""
        from src.solver.queue_driver import _build_spot

        obs = _make_obs_row(_REAL_SRP_FELT, cluster_key="hero_pos_rel=IP")
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        assert ctx["pot_type"] == "srp", f"Expected 'srp' for single-raise pot, got {ctx['pot_type']!r}."

    def test_villain_pos_not_equal_hero_pos(self):
        """villain_pos must not equal hero_position."""
        from src.solver.queue_driver import _build_spot

        cluster_key = "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
        obs = _make_obs_row(_REAL_3BET_FELT, cluster_key)
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        hero = ctx["hero_pos"]
        villain = ctx["villain_pos"]
        assert hero != villain, (
            f"villain_pos={villain!r} must differ from hero_pos={hero!r}. "
            "Old code set villain==hero when villain_pos key absent."
        )

    def test_villain_pos_is_known_seat(self):
        """villain_pos is a real poker seat (SB/BB/UTG/MP/CO/BTN)."""
        from src.solver.queue_driver import _build_spot

        cluster_key = "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
        obs = _make_obs_row(_REAL_3BET_FELT, cluster_key)
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        known_seats = {"SB", "BB", "UTG", "MP", "CO", "BTN"}
        assert ctx["villain_pos"] in known_seats, (
            f"villain_pos={ctx['villain_pos']!r} is not a recognized seat."
        )

    def test_3bet_villain_pos_is_btn(self):
        """In BB-vs-BTN 3bet, hero=BB → villain_pos=BTN (the preflop opener)."""
        from src.solver.queue_driver import _build_spot

        cluster_key = "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
        obs = _make_obs_row(_REAL_3BET_FELT, cluster_key)
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        assert ctx["villain_pos"] == "BTN", (
            f"In BB-vs-BTN 3bet (hero=BB), villain_pos must be BTN, got {ctx['villain_pos']!r}."
        )

    def test_preflop_action_seq_non_empty_for_raised_pot(self):
        """preflop_action_seq must be non-empty for a raised pot."""
        from src.solver.queue_driver import _build_spot

        cluster_key = "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
        obs = _make_obs_row(_REAL_3BET_FELT, cluster_key)
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        pfa = ctx.get("preflop_action_seq")
        assert pfa, (
            f"preflop_action_seq is empty/None: {pfa!r}. "
            "Must extract preflop prefix from action_sequence for raised pots."
        )

    def test_preflop_action_seq_contains_only_preflop_tokens(self):
        """preflop_action_seq contains only preflop tokens (before flop actions)."""
        from src.solver.queue_driver import _build_spot

        cluster_key = "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
        obs = _make_obs_row(_REAL_3BET_FELT, cluster_key)
        _, ctx = _build_spot(obs, {}, self._get_cfg())
        pfa = ctx.get("preflop_action_seq", [])
        postflop_only = {"BB:check", "BTN:bet_5"}
        overlap = set(pfa) & postflop_only
        assert not overlap, (
            f"preflop_action_seq contains postflop tokens: {overlap}. Full preflop_action_seq: {pfa}"
        )


# ---------------------------------------------------------------------------
# BUG 2: IP/OOP range assignment by postflop button order
# ---------------------------------------------------------------------------


class TestIPOOPRangeAssignment:
    """range_ip is the IP player's range, range_oop the OOP player's — regardless of hero."""

    from typing import ClassVar

    _POSTFLOP_ORDER: ClassVar[dict] = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}

    def _palette_for_3bet_bb_vs_btn(self) -> dict:
        """Minimal fake palette: BB 3bets BTN, BTN opens, both call."""
        return {
            "BB/3bet_vs_BTN": {"AA": 10.0, "KK": 8.0, "QQ": 6.0},
            "BTN/defend_3bet_vs_BB": {"AKs": 5.0, "AQs": 4.0, "KQs": 3.0},
        }

    def test_hero_oop_range_oop_is_heroes_range(self):
        """When hero=BB (OOP in BTN vs BB 3bet), range_oop is the BB's range, not BTN's."""
        from src.solver.range_resolver import _resolve_ranges

        lookup = self._palette_for_3bet_bb_vs_btn()
        range_ip, range_oop = _resolve_ranges(
            obs_spot_features={},
            hero_pos="BB",
            villain_pos="BTN",
            pot_type="3bet",
            opener_pos="BTN",
            bettor_pos="BB",
            palette_lookup=lookup,
        )
        assert range_ip != range_oop, "IP and OOP ranges must differ."
        assert "AA" not in range_ip, (
            f"range_ip (BTN defending) should NOT contain AA (BB's 3bet range). range_ip={range_ip!r}"
        )
        assert "AA" in range_oop, f"range_oop (BB, the 3bettor) MUST contain AA. range_oop={range_oop!r}"

    def test_hero_ip_range_ip_is_heroes_range(self):
        """When hero=BTN (IP in BTN vs BB 3bet), range_ip is BTN's range."""
        from src.solver.range_resolver import _resolve_ranges

        lookup = self._palette_for_3bet_bb_vs_btn()
        range_ip, range_oop = _resolve_ranges(
            obs_spot_features={},
            hero_pos="BTN",
            villain_pos="BB",
            pot_type="3bet",
            opener_pos="BTN",
            bettor_pos="BB",
            palette_lookup=lookup,
        )
        assert "AA" not in range_ip, f"range_ip (BTN/defend) should NOT contain AA. range_ip={range_ip!r}"
        assert "AA" in range_oop, f"range_oop (BB/3bet aggressor) MUST contain AA. range_oop={range_oop!r}"

    def test_build_spot_oop_hero_ranges_not_inverted(self):
        """_build_spot with hero=BB 3bet spot: range_oop uses BB key, range_ip uses BTN key."""
        from pathlib import Path

        from src._config import AutoLoopConfig, load_toml_config
        from src.solver.queue_driver import _build_spot

        cfg = load_toml_config(Path("config/autoloop.toml"), AutoLoopConfig).solver_queue
        cluster_key = "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
        obs = _make_obs_row(_REAL_3BET_FELT, cluster_key)

        lookup = {
            "BB/3bet_vs_BTN": {"AA": 10.0, "KK": 8.0},
            "BTN/defend_3bet_vs_BB": {"AKs": 5.0, "AQs": 4.0},
        }
        _, ctx = _build_spot(obs, lookup, cfg)
        range_ip = ctx["range_ip"]
        range_oop = ctx["range_oop"]

        assert "AA" not in range_ip, (
            f"range_ip (BTN, IP player defending) should NOT contain AA. range_ip={range_ip!r}"
        )
        assert "AA" in range_oop, f"range_oop (BB, OOP 3bettor) MUST contain AA. range_oop={range_oop!r}"


# ---------------------------------------------------------------------------
# BUG 3: _inject writes per-DP Milvus node (NOT via PatchEngine)
# ---------------------------------------------------------------------------


class TestInjectCreateFirstNode:
    """_inject writes per-DP Milvus node using obs decision_id + obs embedding."""

    def _make_obs_row_with_cluster(
        self, decision_id: str = "dp-3bet-001", embedding: list | None = None
    ) -> dict:
        cluster_key = "hero_pos_rel=OOP|n_players_active=2|pot_type=3bet|street_class=postflop"
        row = _make_obs_row(_REAL_3BET_FELT, cluster_key)
        row["decision_id"] = decision_id
        row["embedding"] = embedding or [0.42] * 80
        return row

    def _make_solver_result(self):
        r = MagicMock()
        r.action_dist = {"check": 0.6, "bet_33": 0.4}
        r.exploitability_pct = 0.45
        return r

    def _make_spot(self):
        from src.solver.postflop_cli import SolverSpot

        return SolverSpot(
            pot=2700,
            effective_stack=9100,
            board=["Jc", "7s", "2h"],
            range_ip="AA,KK",
            range_oop="AA,KK",
        )

    def test_no_active_node_calls_apply_with_prev_none(self):
        """_inject calls milvus.upsert (not PatchEngine) with decision_id from obs_row."""
        from src.solver.queue_driver import _inject

        obs_embedding = [0.77] * 80
        obs_row = self._make_obs_row_with_cluster(decision_id="dp-no-active-001", embedding=obs_embedding)
        solver_result = self._make_solver_result()
        spot = self._make_spot()

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
                solver_result,
                conn=mock_conn,
                milvus=mock_milvus,
                palette_lookup={},
            )

        mock_milvus.upsert.assert_called_once()
        row = mock_milvus.upsert.call_args.kwargs["data"][0]
        assert row["decision_id"] == "dp-no-active-001", (
            f"Milvus row decision_id must be obs decision_id, got {row.get('decision_id')!r}"
        )

    def test_no_active_node_uses_embedding_from_obs_row(self):
        """Embedding for the Milvus node is the obs's own vector run through the shared
        z-score+weights transform — same space the corpus + live query use."""
        import numpy as np

        from src.solver.queue_driver import _inject, _load_zscore_transform, normalize

        raw_embedding = [0.42] * 80
        mean, std, weights = _load_zscore_transform("postflop_decisions")
        expected_embedding = normalize(
            np.asarray(raw_embedding, dtype=np.float64), mean, std, weights
        ).tolist()
        obs_row = self._make_obs_row_with_cluster(embedding=raw_embedding)
        solver_result = self._make_solver_result()
        spot = self._make_spot()

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
                solver_result,
                conn=mock_conn,
                milvus=mock_milvus,
                palette_lookup={},
            )

        mock_milvus.upsert.assert_called_once()
        row = mock_milvus.upsert.call_args.kwargs["data"][0]
        assert row["embedding"] == pytest.approx(expected_embedding), (
            f"Milvus row embedding must match normalized obs vector. Got {str(row.get('embedding', []))[:60]}..."
        )

    def test_active_node_existing_path_unchanged(self):
        """With or without an existing active node, _inject still writes per-DP Milvus row."""
        from src.solver.queue_driver import _inject

        obs_row = self._make_obs_row_with_cluster(decision_id="dp-existing-001")
        solver_result = self._make_solver_result()
        spot = self._make_spot()

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
                solver_result,
                conn=mock_conn,
                milvus=mock_milvus,
                palette_lookup={},
            )

        mock_milvus.upsert.assert_called_once()
        row = mock_milvus.upsert.call_args.kwargs["data"][0]
        assert row["decision_id"] == "dp-existing-001"
        assert row["active"] is True


# ---------------------------------------------------------------------------
# BUG 5: SolverQueueCheckpoint round-trip (no last_cycle confusion)
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """SolverQueueCheckpoint written by queue_driver round-trips correctly."""

    def test_solver_queue_checkpoint_has_no_last_cycle(self):
        """SolverQueueCheckpoint must NOT have a last_cycle field."""

        from src.solver.queue_driver import SolverQueueCheckpoint

        fields = set(SolverQueueCheckpoint.__struct_fields__)
        assert "last_cycle" not in fields, (
            "SolverQueueCheckpoint has 'last_cycle' — this is from RunCheckpoint. "
            "The two structs must remain separate."
        )
        assert "spots_completed" in fields, "SolverQueueCheckpoint must have 'spots_completed'."

    def test_queue_driver_read_checkpoint_decodes_solver_struct(self):
        """read_checkpoint used in queue_driver decodes as SolverQueueCheckpoint, not RunCheckpoint."""
        import tempfile

        import msgspec

        from src.autoloop.checkpoint import checkpoint_path_for, write_checkpoint
        from src.solver.queue_driver import SolverQueueCheckpoint

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = SolverQueueCheckpoint(
                run_id="test-roundtrip",
                run_epoch="2026-05-30T00:00:00Z",
                spots_submitted=10,
                spots_completed=8,
                spots_failed=1,
                last_obs_id="obs-8",
                last_eval_suite_at=0,
            )
            path = checkpoint_path_for("test-roundtrip", tmp)
            write_checkpoint(path, ckpt)

            loaded = msgspec.json.decode(path.read_bytes(), type=SolverQueueCheckpoint)
            assert loaded.spots_completed == 8
            assert loaded.last_obs_id == "obs-8"
            assert loaded.run_id == "test-roundtrip"

    def test_run_checkpoint_read_raises_on_solver_checkpoint_bytes(self):
        """read_checkpoint (RunCheckpoint) raises on SolverQueueCheckpoint bytes — structs are distinct."""
        import tempfile

        import msgspec

        from src.autoloop.checkpoint import checkpoint_path_for, read_checkpoint, write_checkpoint
        from src.solver.queue_driver import SolverQueueCheckpoint

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = SolverQueueCheckpoint(
                run_id="test-mismatch",
                run_epoch="2026-05-30T00:00:00Z",
                spots_submitted=10,
                spots_completed=8,
                spots_failed=1,
                last_obs_id="obs-8",
                last_eval_suite_at=0,
            )
            path = checkpoint_path_for("test-mismatch", tmp)
            write_checkpoint(path, ckpt)

            with pytest.raises(msgspec.ValidationError):
                read_checkpoint(path)

    def test_queue_driver_resume_reads_solver_checkpoint(self):
        """QueueDriver.run() resumes correctly by reading SolverQueueCheckpoint, not RunCheckpoint."""
        import tempfile
        from pathlib import Path

        from src.autoloop.checkpoint import checkpoint_path_for, write_checkpoint
        from src.solver.queue_driver import QueueDriver, SolverQueueCheckpoint

        with tempfile.TemporaryDirectory() as tmp:
            run_id = "resume-test-real"
            ckpt = SolverQueueCheckpoint(
                run_id=run_id,
                run_epoch="2026-05-30T00:00:00Z",
                spots_submitted=5,
                spots_completed=5,
                spots_failed=0,
                last_obs_id="obs-5",
                last_eval_suite_at=0,
            )
            path = checkpoint_path_for(run_id, tmp)
            write_checkpoint(path, ckpt)

            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_cursor.fetchall.return_value = []
            mock_cursor.fetchone.return_value = (0,)
            mock_conn.cursor.return_value = mock_cursor

            driver = QueueDriver(run_id=run_id, config_path=Path("config/autoloop.toml"))
            driver._cfg = driver._cfg.__class__(
                **{
                    **{f: getattr(driver._cfg, f) for f in driver._cfg.__struct_fields__},
                    "checkpoint_path": tmp,
                }
            )

            result = driver.run(
                _tsdb_conn=mock_conn,
                _milvus=MagicMock(),
                _solver=MagicMock(),
                _patch_engine=MagicMock(),
                install_signals=False,
            )

            assert result["spots_completed"] >= 5, (
                f"Driver did not resume from checkpoint. spots_completed={result['spots_completed']!r}, "
                "expected >= 5 (from checkpoint)."
            )
