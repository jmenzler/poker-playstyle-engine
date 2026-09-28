"""Unit tests for feature extractors (kNN embedding pipeline).

Tests match plan 01-19R §Test Strategy:
  - test_preflop_dim_count
  - test_postflop_dim_count
  - test_all_values_normalized_to_unit_range
  - test_determinism
  - test_group_weight_application
  - test_equity_lookup_exact_hit
  - test_equity_lookup_scenario_fallback
  - test_equity_lookup_miss_returns_none
  - test_dp_with_missing_villain_handled
  - test_river_dp_all_deciles_equal
  - test_snapshot_5_reference_vectors
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Helpers: build a minimal in-memory EquityLookup and sample DPs
# ---------------------------------------------------------------------------

_EQUITY_TABLE_PATH = Path("tools/equity_table.parquet")

# Canonical hole/board that exists in the real equity table (discovered from data)
_REAL_HOLE = None
_REAL_BOARD = None
_REAL_SCEN = None


def _load_real_lookup():
    """Load actual equity table if available, else None."""
    try:
        from tools.equity_lookup import EquityLookup

        if _EQUITY_TABLE_PATH.exists():
            return EquityLookup(_EQUITY_TABLE_PATH)
    except Exception:
        pass
    return None


def _make_mock_lookup(
    exact_hit: bool = True,
    fallback_hit: bool = False,
) -> Any:
    """Return a mock EquityLookup with controllable behaviour."""
    from tools.equity_lookup import EquityLookup  # noqa: F401

    mock = MagicMock()
    sample_eq = {
        "p10": 0.1,
        "p20": 0.2,
        "p30": 0.3,
        "p40": 0.4,
        "p50": 0.5,
        "p60": 0.6,
        "p70": 0.7,
        "p80": 0.8,
        "p90": 0.9,
        "mean": 0.5,
        "variance": 0.05,
    }
    pop = dict(sample_eq)

    if exact_hit:
        mock.get.return_value = sample_eq
    elif fallback_hit:
        # First call miss, but prefix-fallback returns something
        mock.get.return_value = sample_eq
    else:
        mock.get.return_value = None

    mock.population_mean_equity.return_value = pop
    mock.miss_rate.return_value = 0.0
    return mock


def _preflop_dp(**overrides) -> dict:
    """Build a minimal valid preflop DP."""
    base = {
        "id": "100_dp0",
        "hand_id": "100",
        "decision_idx": 0,
        "street": "preflop",
        "hero_pos": "BTN",
        "hero_pos_rel": "IP",
        "n_players_at_street": 2,
        "pot_type": "srp",
        "pot_bb": 3.0,
        "pot_cents": 30,
        "bb_cents": 10,
        "effective_stack_cents": 1000,
        "hero_stack_cents": 1000,
        "hero_stack_bb": 100.0,
        "spr": 33.3,
        "hero_hole": ["Ah", "Kh"],
        "hero_hole_class": "AKs",
        "board": None,
        "preflop_aggressor": "BTN",
        "preflop_action_seq": [
            {"pos": "BTN", "action": "raise", "size_cents": 25},
            {"pos": "BB", "action": "call", "size_cents": 25},
        ],
        "action_so_far_street": [],
        "facing": "cold",
        "facing_pos": None,
        "facing_size_pot_frac": None,
        "facing_size_cents": None,
        "hero_action_type": "raise",
        "hero_action_size_pot_frac": 2.5,
        "hero_action_to_bb": 2.5,
        "_schema_version": 2,
        "scenario_key": "srp|IP|n2|BB:caller:BTN",
    }
    base.update(overrides)
    return base


def _flop_dp(**overrides) -> dict:
    """Build a minimal valid flop DP."""
    base = {
        "id": "200_dp1",
        "hand_id": "200",
        "decision_idx": 1,
        "street": "flop",
        "hero_pos": "BTN",
        "hero_pos_rel": "IP",
        "n_players_at_street": 2,
        "pot_type": "srp",
        "pot_bb": 6.5,
        "pot_cents": 65,
        "bb_cents": 10,
        "effective_stack_cents": 975,
        "hero_stack_cents": 975,
        "hero_stack_bb": 97.5,
        "spr": 15.0,
        "hero_hole": ["Jh", "Th"],
        "hero_hole_class": "JTs",
        "board": ["9c", "7d", "2h"],
        "preflop_aggressor": "BTN",
        "preflop_action_seq": [
            {"pos": "BTN", "action": "raise", "size_cents": 25},
            {"pos": "BB", "action": "call", "size_cents": 25},
        ],
        "action_so_far_street": [
            {"pos": "BB", "action": "check"},
        ],
        "facing": "check_to",
        "facing_pos": "BB",
        "facing_size_pot_frac": None,
        "facing_size_cents": None,
        "hero_action_type": "bet",
        "hero_action_size_pot_frac": 0.5,
        "hero_action_to_bb": 3.25,
        "_schema_version": 2,
        "scenario_key": "srp|IP|n2|BB:caller:BTN",
    }
    base.update(overrides)
    return base


def _river_dp(**overrides) -> dict:
    """Build a minimal valid river DP."""
    base = _flop_dp()
    base.update(
        {
            "id": "200_dp3",
            "decision_idx": 3,
            "street": "river",
            "board": ["9c", "7d", "2h", "Ks", "3c"],
            "pot_bb": 20.0,
            "spr": 3.0,
        }
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: EquityLookup
# ---------------------------------------------------------------------------


class TestEquityLookup:
    def test_equity_lookup_exact_hit(self):
        """EquityLookup returns correct row for exact key."""
        from tools.equity_lookup import EquityLookup

        # Build a tiny in-memory parquet
        table = pa.table(
            {
                "id": ["flop|2c2d|4c3c3h|srp|IP|n2|MP:pfr"],
                "street": ["flop"],
                "hole_canonical": ["2c2d"],
                "board_canonical": ["4c3c3h"],
                "scenario": ["srp|IP|n2|MP:pfr"],
                "mean_equity": [0.5],
                "p10": [0.1],
                "p20": [0.2],
                "p30": [0.3],
                "p40": [0.4],
                "p50": [0.5],
                "p60": [0.6],
                "p70": [0.7],
                "p80": [0.8],
                "p90": [0.9],
                "n_runouts": [1000],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            pq.write_table(table, tf.name)
            eq = EquityLookup(tf.name)

        result = eq.get("2c2d", "4c3c3h", "srp|IP|n2|MP:pfr")
        assert result is not None
        assert abs(result["mean"] - 0.5) < 1e-6
        assert abs(result["p10"] - 0.1) < 1e-6
        assert abs(result["p90"] - 0.9) < 1e-6

    def test_equity_lookup_scenario_fallback(self):
        """EquityLookup falls back to same prefix when exact scenario missing."""
        from tools.equity_lookup import EquityLookup

        # Two rows: exact different scenario for same hole/board prefix
        table = pa.table(
            {
                "id": ["flop|2c2d|4c3c3h|srp|IP|n2|MP:pfr"],
                "street": ["flop"],
                "hole_canonical": ["2c2d"],
                "board_canonical": ["4c3c3h"],
                "scenario": ["srp|IP|n2|MP:pfr"],
                "mean_equity": [0.42],
                "p10": [0.1],
                "p20": [0.2],
                "p30": [0.3],
                "p40": [0.4],
                "p50": [0.42],
                "p60": [0.55],
                "p70": [0.65],
                "p80": [0.75],
                "p90": [0.85],
                "n_runouts": [500],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            pq.write_table(table, tf.name)
            eq = EquityLookup(tf.name)

        # Query with different villain spec but same pot/rel/n prefix
        result = eq.get("2c2d", "4c3c3h", "srp|IP|n2|BTN:pfr")
        assert result is not None, "Expected fallback to succeed"
        assert abs(result["mean"] - 0.42) < 1e-6

    def test_equity_lookup_miss_returns_none(self):
        """EquityLookup returns None when no row matches even the prefix."""
        from tools.equity_lookup import EquityLookup

        table = pa.table(
            {
                "id": ["flop|2c2d|4c3c3h|srp|IP|n2|MP:pfr"],
                "street": ["flop"],
                "hole_canonical": ["2c2d"],
                "board_canonical": ["4c3c3h"],
                "scenario": ["srp|IP|n2|MP:pfr"],
                "mean_equity": [0.5],
                "p10": [0.1],
                "p20": [0.2],
                "p30": [0.3],
                "p40": [0.4],
                "p50": [0.5],
                "p60": [0.6],
                "p70": [0.7],
                "p80": [0.8],
                "p90": [0.9],
                "n_runouts": [1000],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            pq.write_table(table, tf.name)
            eq = EquityLookup(tf.name)

        # Completely different hole/board — guaranteed miss
        result = eq.get("AhKh", "XxYyZz", "nonexistent|scenario")
        assert result is None

    def test_equity_lookup_population_mean_not_nan(self):
        """population_mean_equity() has no NaN values."""
        from tools.equity_lookup import EquityLookup

        table = pa.table(
            {
                "id": ["flop|2c2d|4c3c3h|srp|IP|n2|MP:pfr"],
                "street": ["flop"],
                "hole_canonical": ["2c2d"],
                "board_canonical": ["4c3c3h"],
                "scenario": ["srp|IP|n2|MP:pfr"],
                "mean_equity": [0.5],
                "p10": [0.1],
                "p20": [0.2],
                "p30": [0.3],
                "p40": [0.4],
                "p50": [0.5],
                "p60": [0.6],
                "p70": [0.7],
                "p80": [0.8],
                "p90": [0.9],
                "n_runouts": [1000],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            pq.write_table(table, tf.name)
            eq = EquityLookup(tf.name)

        pop = eq.population_mean_equity()
        for k, v in pop.items():
            assert not np.isnan(v), f"population_mean_equity[{k!r}] is NaN"

    def test_equity_lookup_miss_rate(self):
        """miss_rate is tracked correctly."""
        from tools.equity_lookup import EquityLookup

        table = pa.table(
            {
                "id": ["flop|2c2d|4c3c3h|srp|IP|n2|MP:pfr"],
                "street": ["flop"],
                "hole_canonical": ["2c2d"],
                "board_canonical": ["4c3c3h"],
                "scenario": ["srp|IP|n2|MP:pfr"],
                "mean_equity": [0.5],
                "p10": [0.1],
                "p20": [0.2],
                "p30": [0.3],
                "p40": [0.4],
                "p50": [0.5],
                "p60": [0.6],
                "p70": [0.7],
                "p80": [0.8],
                "p90": [0.9],
                "n_runouts": [1000],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            pq.write_table(table, tf.name)
            eq = EquityLookup(tf.name)

        eq.get("2c2d", "4c3c3h", "srp|IP|n2|MP:pfr")  # hit
        eq.get("AhKh", "XxYyZz", "none|n")  # miss
        eq.get("2c2d", "4c3c3h", "srp|IP|n2|MP:pfr")  # hit

        # 1 miss out of 3 calls = miss_rate includes fallback misses
        assert eq.miss_rate() > 0.0


# ---------------------------------------------------------------------------
# Tests: Preflop extractor
# ---------------------------------------------------------------------------


class TestPreflopExtractor:
    def test_preflop_dim_count(self):
        """Preflop extractor produces exactly 34 dims (post fix #5 blocker expansion)."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp()
        vec, _ = extract_preflop(dp, eq)
        assert vec.shape == (34,), f"Expected 34 dims, got {vec.shape}"

    def test_preflop_dtype_float32(self):
        """Preflop vector dtype is float32."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp()
        vec, _ = extract_preflop(dp, eq)
        assert vec.dtype == np.float32

    def test_preflop_all_values_in_unit_range(self):
        """All preflop dims are in [0, 1] after weight application."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp()
        vec, _ = extract_preflop(dp, eq)
        assert np.all(vec >= 0.0), f"Negative values: {vec[vec < 0]}"
        assert np.all(vec <= 1.0), f"Values > 1: {vec[vec > 1]}"

    def test_preflop_determinism(self):
        """Same DP produces bitwise identical vectors on two calls."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp()
        v1, _ = extract_preflop(dp, eq)
        v2, _ = extract_preflop(dp, eq)
        np.testing.assert_array_equal(v1, v2)

    def test_preflop_filter_dict_fields(self):
        """extract_preflop returns correct hard-filter fields."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp()
        _, filt = extract_preflop(dp, eq)
        assert filt["street_class"] == "preflop"
        assert filt["pot_type"] == "srp"
        assert filt["hero_pos_rel"] == "IP"
        assert filt["n_players_active"] == 2

    def test_preflop_group_a_uses_equity(self):
        """Group A dims (0-5) reflect equity table values."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        # Mock returns p10=0.1, p30=0.3, p50=0.5, p70=0.7, p90=0.9, mean=0.5
        # With group weight 1.5, each is * 1.5, then clipped to 1.0
        # p10*1.5 = 0.15, p30*1.5 = 0.45, p50*1.5 = 0.75, ...
        dp = _preflop_dp()
        vec, _ = extract_preflop(dp, eq)
        # Just verify dims 0-5 are not all zero
        assert np.any(vec[:6] > 0), "Group A (equity) dims are all zero"

    def test_preflop_equity_miss_uses_pop_mean(self):
        """When equity lookup returns None, extractor uses population mean (no crash, no NaN)."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup(exact_hit=False, fallback_hit=False)
        dp = _preflop_dp()
        vec, _ = extract_preflop(dp, eq)
        assert not np.any(np.isnan(vec)), "NaN in preflop vector after equity miss"
        assert vec.shape == (34,)

    def test_preflop_missing_villain_handled(self):
        """DP with null facing_pos (no facing villain) does not raise."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp(facing_pos=None, preflop_aggressor=None)
        vec, _ = extract_preflop(dp, eq)
        assert vec.shape == (34,)
        assert np.all(vec >= 0.0)

    @pytest.mark.parametrize(
        "hc,expected_suited",
        [
            ("AKs", 1.0),
            ("AKo", 0.0),
            ("AA", 0.0),
            ("72o", 0.0),
        ],
    )
    def test_preflop_suited_bit(self, hc, expected_suited):
        """Group I suited bit is correctly set."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp(hero_hole_class=hc)
        vec, _ = extract_preflop(dp, eq)
        # Group I starts at dim 6; suited is dim 7 (index 7 in 0-based)
        # After weight 1.2 and clip: suited = 1.0 * 1.2 → clipped to 1.0
        # or 0.0 for unsuited
        # Group I dims 6-10; suited is index 7
        suited_val = float(vec[7])
        if expected_suited == 1.0:
            assert suited_val > 0.0, f"Expected suited>0 for {hc}, got {suited_val}"
        else:
            assert suited_val == 0.0, f"Expected suited=0 for {hc}, got {suited_val}"

    def test_group_i_derived_from_raw_hole_when_class_absent(self):
        """Group I (dims 6-10) must be nonzero when hero_hole_class is absent or
        raw canonical cards — real DPs carry only hero_hole, not 169-class notation."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        # Build-time DP: only raw hole cards, no 169-class string
        dp = _preflop_dp(hero_hole=["Ah", "Kd"])
        dp.pop("hero_hole_class", None)
        vec, _ = extract_preflop(dp, eq)
        group_i = vec[6:11]
        assert float(abs(group_i).sum()) > 0.0, f"Group I all-zero for AKo: {group_i}"
        # Serve-time DP: hero_hole_class is 4-char canonical cards ('AcKd')
        dp2 = _preflop_dp(hero_hole=["Ac", "Kd"], hero_hole_class="AcKd")
        vec2, _ = extract_preflop(dp2, eq)
        assert float(abs(vec2[6:11]).sum()) > 0.0, "Group I all-zero for canonical-card class"

    def test_hole_to_class_converter(self):
        """_hole_to_class maps raw cards to 169-class notation."""
        from tools.feature_extractors.preflop import _hole_to_class

        assert _hole_to_class(["Ac", "Kd"]) == "AKo"
        assert _hole_to_class(["Ah", "Kh"]) == "AKs"
        assert _hole_to_class(["Ah", "Ad"]) == "AA"
        assert _hole_to_class(["2c", "7d"]) == "72o"  # high rank first
        assert _hole_to_class([]) == ""

    def test_pot_odds_not_clamped_at_half(self):
        """Fix #2: pot_odds (dim 20) must exceed 0.5 for large bets (jam/4bet/5bet)."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        # facing_size_pot_frac = 3.0 → call/(1+call) = 3/4 = 0.75, previously clamped to 0.5
        dp = _preflop_dp(facing_size_pot_frac=3.0)
        vec, _ = extract_preflop(dp, eq)
        pot_odds = float(vec[20])
        assert pot_odds > 0.5, f"pot_odds {pot_odds:.4f} still clamped at 0.5 for jam spot"
        assert pot_odds <= 1.0, f"pot_odds {pot_odds:.4f} exceeds 1.0"

    def test_players_yet_to_act_is_zero(self):
        """Fix #6: dim 19 (players_yet_to_act) must be 0.0 — not reconstructable from schema v1.1."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        # Even with many actions in action_so_far_street, dim 19 must be 0
        dp = _preflop_dp(
            n_players_at_street=6,
            action_so_far_street=[
                {"pos": "UTG", "action": "fold"},
            ],
        )
        vec, _ = extract_preflop(dp, eq)
        players_yet = float(vec[19])
        assert players_yet == 0.0, f"players_yet_to_act dim 19 = {players_yet}, expected 0.0"

    def test_preflop_dim_count_34(self):
        """Fix #5: Preflop extractor produces exactly 34 dims after blocker expansion."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp()
        vec, _ = extract_preflop(dp, eq)
        assert vec.shape == (34,), f"Expected 34 dims, got {vec.shape}"

    def test_blocker_dims_in_unit_range(self):
        """Fix #5: blocker dims 33/34 (indices 32/33) are in [0,1]."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        dp = _preflop_dp()
        vec, _ = extract_preflop(dp, eq)
        assert 0.0 <= float(vec[32]) <= 1.0, f"value-blocker dim 32 = {vec[32]}"
        assert 0.0 <= float(vec[33]) <= 1.0, f"range-blocker dim 33 = {vec[33]}"

    def test_blocker_dims_respond_to_ace_vs_aa(self):
        """Fix #5: hero holding Ace vs villain AA/AK premium range should have blocker > 0."""
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        # Hero holds As Kh — blocks AA and AK combos significantly
        range_lookup = {"BTN/open": {"AA": 100.0, "KK": 100.0, "AKs": 50.0, "AKo": 50.0}}
        dp_with_ace = _preflop_dp(hero_hole=["As", "Kh"], hero_hole_class="AKo")
        vec_with_ace, _ = extract_preflop(dp_with_ace, eq, range_lookup=range_lookup)

        # Hero holds 7c 2h — doesn't block premium combos
        dp_no_blocker = _preflop_dp(hero_hole=["7c", "2h"], hero_hole_class="72o")
        vec_no_blocker, _ = extract_preflop(dp_no_blocker, eq, range_lookup=range_lookup)

        val_blocker_ace = float(vec_with_ace[32])
        val_blocker_low = float(vec_no_blocker[32])
        assert val_blocker_ace > val_blocker_low, (
            f"Ace should block more than 72o: {val_blocker_ace:.4f} vs {val_blocker_low:.4f}"
        )

    @pytest.mark.parametrize(
        "pot_type,action_stem",
        [("srp", "open"), ("3bet", "3bet"), ("4bet", "4bet"), ("5bet+", "5bet")],
    )
    def test_blocker_resolves_for_all_pot_types(self, pot_type, action_stem):
        """Blocker dims must resolve for every preflop_pot_type label, incl. deep '5bet+'.

        pot_type labels come from tools.position.preflop_pot_type — '5bet+' for 4+ raises.
        A missing label→action mapping silently returns zeros for all deep spots.
        """
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        range_lookup = {f"BTN/{action_stem}": {"AA": 100.0, "KK": 100.0, "AKs": 50.0, "AKo": 50.0}}
        dp = _preflop_dp(hero_hole=["As", "Kh"], hero_hole_class="AKo", pot_type=pot_type)
        vec, _ = extract_preflop(dp, eq, range_lookup=range_lookup)
        assert float(vec[32]) > 0.0, (
            f"value-blocker is zero for pot_type={pot_type!r} — label not mapped to {action_stem!r}"
        )

    @pytest.mark.parametrize(
        "pot_type,action_stem",
        [("srp", "open"), ("3bet", "3bet"), ("4bet", "4bet"), ("5bet+", "5bet")],
    )
    def test_tightness_resolves_for_all_pot_types(self, pot_type, action_stem):
        """Group J tightness must use the observed palette for every pot_type, incl. '5bet+'.

        Otherwise deep spots silently fall back to the position proxy (0.33 for BTN).
        """
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        observed_tightness = 0.91  # distinct from any _TIGHTNESS_BY_POS proxy
        tightness_lookup = {f"BTN/{action_stem}": observed_tightness}
        dp = _preflop_dp(pot_type=pot_type)
        vec, _ = extract_preflop(dp, eq, tightness_lookup=tightness_lookup)
        # Group J tightness is the last dim before blockers: index 31 (A6+I5+B8+...).
        assert abs(float(vec[31]) - observed_tightness) < 1e-5, (
            f"tightness fell back to proxy for pot_type={pot_type!r} (got {float(vec[31]):.4f})"
        )


# ---------------------------------------------------------------------------
# Tests: Postflop extractor
# ---------------------------------------------------------------------------


class TestPostflopExtractor:
    def test_postflop_dim_count(self):
        """Postflop extractor produces exactly 80 dims."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        dp = _flop_dp()
        vec, _ = extract_postflop(dp, eq)
        assert vec.shape == (80,), f"Expected 80 dims, got {vec.shape}"

    def test_postflop_dtype_float32(self):
        """Postflop vector dtype is float32."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        dp = _flop_dp()
        vec, _ = extract_postflop(dp, eq)
        assert vec.dtype == np.float32

    def test_postflop_all_values_in_unit_range(self):
        """All postflop dims are in [0, 1] after weight application."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        dp = _flop_dp()
        vec, _ = extract_postflop(dp, eq)
        assert np.all(vec >= 0.0), f"Negative values at indices: {np.where(vec < 0)[0].tolist()}"
        assert np.all(vec <= 1.0), f"Values > 1 at indices: {np.where(vec > 1)[0].tolist()}"

    def test_postflop_determinism(self):
        """Same DP produces bitwise identical vectors on two calls."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        dp = _flop_dp()
        v1, _ = extract_postflop(dp, eq)
        v2, _ = extract_postflop(dp, eq)
        np.testing.assert_array_equal(v1, v2)

    def test_postflop_filter_dict_fields(self):
        """extract_postflop returns correct hard-filter fields."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        dp = _flop_dp()
        _, filt = extract_postflop(dp, eq)
        assert filt["street_class"] == "postflop"
        assert filt["pot_type"] == "srp"
        assert filt["n_players_active"] == 2

    def test_river_dp_all_deciles_equal(self):
        """River DP: all 9 equity decile dims (0-8) equal the mean equity value (raw, no weight).

        Phase 2 Decision 1: group weights are applied at upsert time (tools/upsert_milvus.py),
        NOT in the extractor. Extractor returns raw min-maxed values.
        """
        from tools.feature_extractors.postflop import extract_postflop

        mock_mean = 0.6
        eq = MagicMock()
        sample_eq = {
            "p10": 0.1,
            "p20": 0.2,
            "p30": 0.3,
            "p40": 0.4,
            "p50": 0.5,
            "p60": 0.6,
            "p70": 0.7,
            "p80": 0.8,
            "p90": 0.9,
            "mean": mock_mean,
            "variance": 0.0,
        }
        eq.get.return_value = sample_eq
        eq.population_mean_equity.return_value = sample_eq

        dp = _river_dp()
        vec, _ = extract_postflop(dp, eq)

        # Dims 0-8 are the 9 deciles; on river they should all equal mean (RAW, no weight applied)
        # Phase 2 Decision 1: weight_A (1.5) is deferred to upsert time
        expected = float(np.clip(mock_mean, 0.0, 1.0))
        for i in range(9):
            assert abs(float(vec[i]) - expected) < 1e-5, (
                f"River decile dim {i} = {vec[i]:.6f}, expected {expected:.6f} (raw, no weight)"
            )

    def test_postflop_equity_miss_no_nan(self):
        """When equity lookup returns None, postflop vector has no NaN values."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup(exact_hit=False, fallback_hit=False)
        dp = _flop_dp()
        vec, _ = extract_postflop(dp, eq)
        assert not np.any(np.isnan(vec)), "NaN in postflop vector after equity miss"
        assert vec.shape == (80,)

    def test_postflop_group_weight_not_applied_by_extractor(self):
        """Extractor returns RAW (unweighted) equity dims — weights are deferred to upsert time.

        Phase 2 Decision 1: group weights (A=1.5, etc.) are applied at upsert time in
        tools/upsert_milvus.py, NOT in the extractor. This test verifies that the extractor
        does NOT multiply by the group weight (so zscore_fit operates on unweighted population).
        """
        from tools.feature_extractors.postflop import extract_postflop

        # Mock returns p10=0.2 (all deciles = 0.2 for testability)
        sample_eq = {
            "p10": 0.2,
            "p20": 0.2,
            "p30": 0.2,
            "p40": 0.2,
            "p50": 0.2,
            "p60": 0.2,
            "p70": 0.2,
            "p80": 0.2,
            "p90": 0.2,
            "mean": 0.2,
            "variance": 0.0,
        }
        eq = MagicMock()
        eq.get.return_value = sample_eq
        eq.population_mean_equity.return_value = sample_eq

        dp = _flop_dp()
        vec, _ = extract_postflop(dp, eq)

        # Group A weight=1.5 is NOT applied in extractor; dims 0-8 should be raw ≈ 0.2
        # (NOT 0.2 * 1.5 = 0.3 — that was the old apply order)
        for i in range(9):
            assert abs(float(vec[i]) - 0.2) < 1e-5, (
                f"Group A dim {i} = {vec[i]:.6f}, expected 0.20 (raw, no weight applied by extractor)"
            )

    def test_postflop_river_draws_zeroed(self):
        """River DP: all draw flags (Group B, dims 12-19) are 0."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        dp = _river_dp()
        vec, _ = extract_postflop(dp, eq)
        # Group B starts at dim 12 (0-based), length 8
        b_dims = vec[12:20]
        assert np.all(b_dims == 0.0), f"Draw dims not zeroed on river: {b_dims}"

    def test_postflop_missing_villain_handled(self):
        """DP with null facing_pos (no facing villain) does not raise."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        dp = _flop_dp(facing_pos=None, preflop_aggressor=None)
        vec, _ = extract_postflop(dp, eq)
        assert vec.shape == (80,)
        assert np.all(vec >= 0.0)

    @pytest.mark.parametrize(
        "street,expected_idx",
        [
            ("flop", 0.0),
            ("turn", 0.5),
            ("river", 1.0),
        ],
    )
    def test_postflop_street_index_dim(self, street, expected_idx):
        """Group F street_index dim (42nd = index 41) reflects street correctly."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        board_by_street = {
            "flop": ["9c", "7d", "2h"],
            "turn": ["9c", "7d", "2h", "Ks"],
            "river": ["9c", "7d", "2h", "Ks", "3c"],
        }
        dp = _flop_dp(street=street, board=board_by_street[street])
        vec, _ = extract_postflop(dp, eq)
        # Group F starts at dim 41 (0-based): A(12)+B(8)+C(6)+D(8)+E(7) = 41
        street_idx_dim = float(vec[41])
        assert abs(street_idx_dim - expected_idx) < 1e-5, (
            f"street_index for {street!r}: expected {expected_idx}, got {street_idx_dim}"
        )

    def test_group_f_total_invested_is_hero_own_investment(self):
        """Group F total_invested (dim 52) is hero's OWN this-street investment / pot,
        not the villain bet hero faces."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        # Hero (BTN) bet 3bb into a 6bb pot; villain then raises to 9bb (the bet hero faces).
        dp = _flop_dp(
            pot_bb=6.0,
            action_so_far_street=[
                {"pos": "BB", "action": "check"},
                {"pos": "BTN", "action": "bet", "size_bb": 3.0},
                {"pos": "BB", "action": "raise", "size_bb": 9.0},
            ],
            facing="raise_to",
            facing_size_pot_frac=1.5,
        )
        vec, _ = extract_postflop(dp, eq)
        expected = 3.0 / 6.0  # hero invested 3bb of a 6bb pot
        assert abs(float(vec[52]) - expected) < 1e-5, (
            f"total_invested dim 52 = {float(vec[52]):.4f}, expected hero's own {expected:.4f}"
        )
        # And it must NOT echo the villain bet hero faces (facing_size_pot_frac=1.5).
        assert abs(float(vec[52]) - 1.5) > 1e-3, "dim 52 is the villain bet, not hero's investment"

    def test_group_f_total_invested_zero_when_hero_passive(self):
        """Hero who only checked has invested nothing this street → dim 52 == 0."""
        from tools.feature_extractors.postflop import extract_postflop

        eq = _make_mock_lookup()
        dp = _flop_dp(
            pot_bb=6.0,
            action_so_far_street=[
                {"pos": "BTN", "action": "check"},
                {"pos": "BB", "action": "bet", "size_bb": 4.0},
            ],
            facing="bet_to",
            facing_size_pot_frac=0.66,
        )
        vec, _ = extract_postflop(dp, eq)
        assert float(vec[52]) == 0.0, f"hero invested nothing but dim 52 = {float(vec[52])}"


# ---------------------------------------------------------------------------
# Tests: straight-draw classification (Group B)
# ---------------------------------------------------------------------------


class TestAnalyzeDraws:
    def test_jqka_is_gutshot_not_oesd(self):
        """J-Q-K-A is one-way (only T completes) → gutshot/4 outs, never OESD."""
        from tools.feature_extractors.postflop import _analyze_draws

        d = _analyze_draws(["Jh", "Qd"], ["Kc", "As", "2h"], "flop")
        assert d["oesd"] == 0, "JQKA wrongly flagged open-ended at the deck boundary"
        assert d["gutshot"] == 1
        assert d["outs"] == 4

    def test_jt98_is_oesd(self):
        """J-T-9-8 completes both ends (7 or Q) → OESD/8 outs."""
        from tools.feature_extractors.postflop import _analyze_draws

        d = _analyze_draws(["Jh", "Td"], ["9c", "8s", "2h"], "flop")
        assert d["oesd"] == 1
        assert d["gutshot"] == 0
        assert d["outs"] == 8

    def test_low_end_boundary_a2345_run_is_gutshot(self):
        """2-3-4-5 completes only on the high end (6); A below 2 does not extend → gutshot."""
        from tools.feature_extractors.postflop import _analyze_draws

        d = _analyze_draws(["2h", "3d"], ["4c", "5s", "Kh"], "flop")
        assert d["oesd"] == 0
        assert d["gutshot"] == 1
        assert d["outs"] == 4

    def test_wheel_draw_a234_is_gutshot(self):
        """A-2-3-4 needs only the 5 → one-way gutshot, not OESD."""
        from tools.feature_extractors.postflop import _analyze_draws

        d = _analyze_draws(["Ah", "2d"], ["3c", "4s", "Kh"], "flop")
        assert d["oesd"] == 0
        assert d["gutshot"] == 1
        assert d["outs"] == 4

    def test_middle_gutshot_unaffected(self):
        """An interior one-gapper (J-T-9-7, missing 8) stays a gutshot."""
        from tools.feature_extractors.postflop import _analyze_draws

        d = _analyze_draws(["Jh", "Td"], ["9c", "7s", "2h"], "flop")
        assert d["oesd"] == 0
        assert d["gutshot"] == 1


# ---------------------------------------------------------------------------
# Tests: snapshot vectors (5 reference DPs)
# ---------------------------------------------------------------------------

SNAPSHOT_PATH = Path("tests/fixtures/embedding_snapshots/reference_vectors.json")


def _make_reference_dps() -> list[dict]:
    """5 reference DPs covering different scenarios."""
    return [
        _preflop_dp(
            id="ref_pf_1", hero_hole_class="AAs", hero_hole=["Ah", "Ad"], pot_type="srp", hero_pos_rel="IP"
        ),
        _preflop_dp(
            id="ref_pf_2", hero_hole_class="72o", hero_hole=["7c", "2h"], pot_type="3bp", hero_pos_rel="OOP"
        ),
        _flop_dp(id="ref_po_1", street="flop"),
        _flop_dp(id="ref_po_2", street="turn", board=["9c", "7d", "2h", "Ks"]),
        _river_dp(id="ref_po_3"),
    ]


class TestSnapshotVectors:
    """Snapshot regression tests — write on first run, compare on subsequent runs."""

    def _compute_snapshots(self) -> list[dict]:
        from tools.feature_extractors.postflop import extract_postflop
        from tools.feature_extractors.preflop import extract_preflop

        eq = _make_mock_lookup()
        pop_mean = eq.population_mean_equity()
        snapshots = []
        for dp in _make_reference_dps():
            street = dp.get("street", "")
            if street == "preflop":
                vec, filt = extract_preflop(dp, eq, pop_mean)
            else:
                vec, filt = extract_postflop(dp, eq, pop_mean)
            snapshots.append(
                {
                    "id": dp["id"],
                    "dims": len(vec),
                    "street_class": filt["street_class"],
                    "embedding": vec.tolist(),
                }
            )
        return snapshots

    def test_snapshot_5_reference_vectors(self):
        """5 reference vectors are stable across runs (snapshot regression)."""
        snapshots = self._compute_snapshots()
        assert len(snapshots) == 5

        if not SNAPSHOT_PATH.exists():
            # First run — write snapshot
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT_PATH.write_text(json.dumps(snapshots, indent=2))
            pytest.skip("Snapshot written on first run — re-run to validate")

        # Compare to stored snapshot
        stored = json.loads(SNAPSHOT_PATH.read_text())
        assert len(stored) == len(snapshots)
        for stored_s, current_s in zip(stored, snapshots):
            assert stored_s["id"] == current_s["id"]
            stored_vec = np.array(stored_s["embedding"], dtype=np.float32)
            current_vec = np.array(current_s["embedding"], dtype=np.float32)
            np.testing.assert_array_equal(
                stored_vec, current_vec, err_msg=f"Snapshot mismatch for DP {stored_s['id']!r}"
            )

    def test_snapshot_dim_counts(self):
        """Snapshot DPs have correct dim counts per street_class."""
        snapshots = self._compute_snapshots()
        for s in snapshots:
            if s["street_class"] == "preflop":
                assert s["dims"] == 34, f"DP {s['id']}: expected 34, got {s['dims']}"
            else:
                assert s["dims"] == 80, f"DP {s['id']}: expected 80, got {s['dims']}"
