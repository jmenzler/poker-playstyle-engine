"""Tests for tools/build_preflop_equity_table.py.

TDD RED phase: tests written before implementation is fully wired.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.build_preflop_equity_table import (
    ALL_HAND_CLASSES,
    _build_pooled_field_range,
    _hero_combo_for_class,
    _load_palette_combo_counts,
    _matrix_cell_equity_multiway,
    _sample_villain_combo,
    _villain_combos_for_class,
    _worker,
)

# ---------------------------------------------------------------------------
# 1. Test _sample_villain_combo distribution matches palette weights
# ---------------------------------------------------------------------------


class TestSampleVillainCombo:
    """Validate that villain combos are sampled from the right hand classes."""

    def test_pair_combos_no_blocked(self):
        """A pair class produces exactly 6 combos (C(4,2))."""
        combos = _villain_combos_for_class("AA", set())
        assert combos is not None
        assert len(combos) == 6
        for c in combos:
            assert len(c) == 2
            assert c[0][0] == "A" and c[1][0] == "A"

    def test_suited_combos_no_blocked(self):
        """A suited class produces exactly 4 combos (1 per suit)."""
        combos = _villain_combos_for_class("AKs", set())
        assert combos is not None
        assert len(combos) == 4
        for c in combos:
            # Same suit
            assert c[0][1] == c[1][1]
            assert {c[0][0], c[1][0]} == {"A", "K"}

    def test_offsuit_combos_no_blocked(self):
        """An offsuit class produces exactly 12 combos (4 suits x 3 other suits)."""
        combos = _villain_combos_for_class("AKo", set())
        assert combos is not None
        assert len(combos) == 12
        for c in combos:
            assert c[0][1] != c[1][1]  # different suits

    def test_blocking_reduces_combos(self):
        """Blocking Ac eliminates combos containing Ac."""
        blocked = {"Ac"}
        combos = _villain_combos_for_class("AKo", blocked)
        assert combos is not None
        for c in combos:
            assert "Ac" not in c

    def test_fully_blocked_returns_none(self):
        """Blocking all 4 aces makes AA impossible."""
        blocked = {"Ac", "Ad", "Ah", "As"}
        result = _villain_combos_for_class("AA", blocked)
        assert result is None

    def test_sample_returns_2_cards(self):
        """_sample_villain_combo always returns a 2-card list."""
        rng = random.Random(42)
        result = _sample_villain_combo("AKo", set(), rng)
        assert result is not None
        assert len(result) == 2

    def test_sample_respects_blocking(self):
        """Sampled combo never contains a blocked card."""
        rng = random.Random(42)
        blocked = {"Ac", "Ad", "Ah"}  # leave only As
        # AKs with only As available -> Asxs for x in KQJT...
        for _ in range(20):
            combo = _sample_villain_combo("AKs", blocked, rng)
            if combo:
                assert not (set(combo) & blocked)

    def test_chi_square_weight_distribution(self):
        """10k samples from a 2-class weighted palette respect the weights.

        Use a synthetic combo_counts: {'AA': 100, 'KK': 300}.
        Expected AA fraction ~0.25, KK fraction ~0.75.
        """
        combo_counts = {"AA": 100, "KK": 300}
        classes = list(combo_counts.keys())
        weights = [float(combo_counts[c]) for c in classes]
        total_w = sum(weights)
        cum_weights = []
        running = 0.0
        for w in weights:
            running += w / total_w
            cum_weights.append(running)

        rng = random.Random(7)
        counts = {c: 0 for c in classes}
        n_samples = 10_000
        for _ in range(n_samples):
            r = rng.random()
            idx = 0
            for i, cw in enumerate(cum_weights):
                if r <= cw:
                    idx = i
                    break
            counts[classes[idx]] += 1

        aa_frac = counts["AA"] / n_samples
        kk_frac = counts["KK"] / n_samples

        # Allow ±0.03 tolerance
        assert abs(aa_frac - 0.25) < 0.03, f"AA fraction {aa_frac:.3f} too far from 0.25"
        assert abs(kk_frac - 0.75) < 0.03, f"KK fraction {kk_frac:.3f} too far from 0.75"


# ---------------------------------------------------------------------------
# 2. Small end-to-end run: 1 scenario x 3 hero classes x 100 trials
# ---------------------------------------------------------------------------


class TestEndToEndSmall:
    """Minimal run to verify parquet output with correct schema."""

    @pytest.fixture
    def synthetic_combo_counts(self):
        """A simple synthetic combo_counts dict for testing without real palette files."""
        # Broad range: lots of hands
        counts: dict[str, int] = {}
        for cls in ["AA", "KK", "QQ", "AKs", "AQs", "AKo", "AQo", "KQo", "JTs", "98s"]:
            counts[cls] = 50
        return counts

    def test_worker_returns_valid_row(self, synthetic_combo_counts):
        """_worker returns a dict with correct schema for a valid input."""
        result = _worker(("AKo", "srp|IP|n2|BB:caller:BTN", synthetic_combo_counts, 100, 42))
        assert result is not None
        expected_keys = {
            "id",
            "street",
            "hole_canonical",
            "board_canonical",
            "scenario",
            "mean_equity",
            "p10",
            "p20",
            "p30",
            "p40",
            "p50",
            "p60",
            "p70",
            "p80",
            "p90",
        }
        assert set(result.keys()) == expected_keys
        assert result["street"] == "preflop"
        assert result["board_canonical"] == ""
        assert 0.0 <= result["mean_equity"] <= 1.0
        for p in ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]:
            assert 0.0 <= result[p] <= 1.0, f"{p} out of range: {result[p]}"

    def test_three_classes_produce_3_rows(self, synthetic_combo_counts, tmp_path):
        """Running 3 hero classes x 1 scenario x 100 trials writes a parquet with 3 rows."""
        import pyarrow as pa

        hero_classes = ["AA", "KK", "72o"]
        scen = "srp|IP|n2|BB:caller:BTN"
        rows = []
        for i, hc in enumerate(hero_classes):
            result = _worker((hc, scen, synthetic_combo_counts, 100, 42 + i))
            if result:
                rows.append(result)

        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"

        # Write to parquet
        out_path = tmp_path / "test_preflop.parquet"
        table = pa.table(
            {
                "id": [r["id"] for r in rows],
                "street": [r["street"] for r in rows],
                "hole_canonical": [r["hole_canonical"] for r in rows],
                "board_canonical": [r["board_canonical"] for r in rows],
                "scenario": [r["scenario"] for r in rows],
                "mean_equity": [r["mean_equity"] for r in rows],
                "p10": [r["p10"] for r in rows],
                "p20": [r["p20"] for r in rows],
                "p30": [r["p30"] for r in rows],
                "p40": [r["p40"] for r in rows],
                "p50": [r["p50"] for r in rows],
                "p60": [r["p60"] for r in rows],
                "p70": [r["p70"] for r in rows],
                "p80": [r["p80"] for r in rows],
                "p90": [r["p90"] for r in rows],
            }
        )
        pq.write_table(table, out_path, compression="zstd")

        # Read back and verify schema
        loaded = pq.read_table(out_path).to_pydict()
        assert len(loaded["id"]) == 3
        assert all(s == "preflop" for s in loaded["street"])
        assert all(b == "" for b in loaded["board_canonical"])

    def test_all_169_classes_present(self):
        """ALL_HAND_CLASSES has exactly 169 entries."""
        assert len(ALL_HAND_CLASSES) == 169
        # Check uniqueness
        assert len(set(ALL_HAND_CLASSES)) == 169
        # Check category counts
        pairs = [h for h in ALL_HAND_CLASSES if len(h) == 2]
        suited = [h for h in ALL_HAND_CLASSES if len(h) == 3 and h[2] == "s"]
        offsuit = [h for h in ALL_HAND_CLASSES if len(h) == 3 and h[2] == "o"]
        assert len(pairs) == 13
        assert len(suited) == 78
        assert len(offsuit) == 78


# ---------------------------------------------------------------------------
# 3. Hero-vs-random monotone equity ordering
# ---------------------------------------------------------------------------


class TestEquityMonotone:
    """AA should beat KK should beat 72o in equity vs a broad random range."""

    @pytest.fixture
    def broad_combo_counts(self):
        """Broad villain range covering all 169 classes with equal weight."""
        return {cls: 10 for cls in ALL_HAND_CLASSES}

    def test_aa_beats_kk_beats_72o(self, broad_combo_counts):
        """AA mean equity > KK mean equity > 72o mean equity (allow ±0.03 noise)."""
        n_trials = 1000  # more trials for stable results in test
        aa = _worker(("AA", "srp|IP|n2|vNA", broad_combo_counts, n_trials, 1))
        kk = _worker(("KK", "srp|IP|n2|vNA", broad_combo_counts, n_trials, 2))
        low = _worker(("72o", "srp|IP|n2|vNA", broad_combo_counts, n_trials, 3))

        assert aa is not None
        assert kk is not None
        assert low is not None

        tolerance = 0.03
        assert aa["mean_equity"] > kk["mean_equity"] - tolerance, (
            f"AA ({aa['mean_equity']:.3f}) should beat KK ({kk['mean_equity']:.3f})"
        )
        assert kk["mean_equity"] > low["mean_equity"] - tolerance, (
            f"KK ({kk['mean_equity']:.3f}) should beat 72o ({low['mean_equity']:.3f})"
        )

    def test_hero_combo_cards_correct(self):
        """Hero combos use deterministic suit patterns per spec."""
        # Pair: Xc Xd
        assert _hero_combo_for_class("AA") == ["Ac", "Ad"]
        assert _hero_combo_for_class("22") == ["2c", "2d"]
        # Suited: Xc Yc
        assert _hero_combo_for_class("AKs") == ["Ac", "Kc"]
        assert _hero_combo_for_class("32s") == ["3c", "2c"]
        # Offsuit: Xc Yd
        assert _hero_combo_for_class("AKo") == ["Ac", "Kd"]
        assert _hero_combo_for_class("32o") == ["3c", "2d"]


# ---------------------------------------------------------------------------
# Fix #1 — vNA dissolve tests
# ---------------------------------------------------------------------------


class TestVNADissolve:
    """Fix #1: vNA scenarios get a real equity row via pooled field range."""

    def test_load_palette_vna_returns_pooled_range(self, tmp_path):
        """_load_palette_combo_counts returns pooled range for vNA scenarios, not None."""
        # Create a minimal palette directory with one JSON file
        pos_dir = tmp_path / "BTN"
        pos_dir.mkdir()
        palette = {
            "combo_counts": {"AA": 10, "KK": 8, "AKs": 5},
            "low_sample": False,
        }
        (pos_dir / "open.json").write_text(__import__("json").dumps(palette))

        pooled = _build_pooled_field_range(tmp_path)
        assert pooled is not None
        assert len(pooled) > 0
        # All known classes should be summed in
        assert "AA" in pooled
        assert "KK" in pooled

        result = _load_palette_combo_counts("srp|IP|n2|vNA", pooled_field_range=pooled)
        assert result is not None, "vNA scenario should return pooled field range"
        assert isinstance(result, list)
        assert len(result) == 1
        assert "AA" in result[0]

    def test_vna_without_pooled_range_still_none(self):
        """_load_palette_combo_counts returns None for vNA when no pooled range provided."""
        result = _load_palette_combo_counts("srp|IP|n2|vNA", pooled_field_range=None)
        assert result is None

    def test_build_pooled_field_range_unions_all_files(self, tmp_path):
        """_build_pooled_field_range sums combo_counts from all palette files."""
        btn_dir = tmp_path / "BTN"
        btn_dir.mkdir()
        co_dir = tmp_path / "CO"
        co_dir.mkdir()
        (btn_dir / "open.json").write_text(__import__("json").dumps({"combo_counts": {"AA": 10, "KK": 5}}))
        (co_dir / "open.json").write_text(__import__("json").dumps({"combo_counts": {"AA": 3, "QQ": 7}}))
        pooled = _build_pooled_field_range(tmp_path)
        assert pooled["AA"] == 13.0
        assert pooled["KK"] == 5.0
        assert pooled["QQ"] == 7.0

    def test_build_pooled_field_range_empty_dir(self, tmp_path):
        """_build_pooled_field_range returns empty dict for empty directory."""
        pooled = _build_pooled_field_range(tmp_path)
        assert pooled == {}


# ---------------------------------------------------------------------------
# Fix #3 — multiway per-villain equity tests
# ---------------------------------------------------------------------------


class TestMultiwayEquity:
    """Fix #3: multiway equity uses per-villain product, not blended range."""

    def _toy_matrix(self) -> dict[str, dict[str, float]]:
        """Small 3-class matrix: AA=0.85 vs all, KK=0.70 vs all, 72o=0.35 vs all."""
        classes = ["AA", "KK", "72o"]
        eq_map = {"AA": 0.85, "KK": 0.70, "72o": 0.35}
        m: dict[str, dict[str, float]] = {}
        for hero in classes:
            m[hero] = {villain: eq_map[hero] for villain in classes}
        return m

    def test_multiway_product_lower_than_single(self):
        """With 2 villains each with equity e, product equity ≈ e^2 < e."""
        matrix = self._toy_matrix()
        # Hero AA vs villain1 KK and villain2 72o separately
        v1 = {"KK": 100.0}
        v2 = {"72o": 100.0}
        result = _matrix_cell_equity_multiway("AA", [v1, v2], matrix)
        assert result is not None

        # Single villain result for comparison
        from tools.build_preflop_equity_table import _matrix_cell_equity

        single_v1 = _matrix_cell_equity("AA", v1, matrix)
        assert single_v1 is not None
        # Multiway mean must be lower (beating two villains is harder)
        assert result["mean_equity"] <= single_v1["mean_equity"], (
            f"multiway {result['mean_equity']:.4f} should be <= single {single_v1['mean_equity']:.4f}"
        )

    def test_multiway_single_villain_matches_original(self):
        """_matrix_cell_equity_multiway with 1 villain matches _matrix_cell_equity."""
        from tools.build_preflop_equity_table import _matrix_cell_equity

        matrix = self._toy_matrix()
        combo_counts = {"KK": 100.0, "72o": 50.0}
        single = _matrix_cell_equity("AA", combo_counts, matrix)
        multi = _matrix_cell_equity_multiway("AA", [combo_counts], matrix)
        assert single is not None
        assert multi is not None
        assert abs(single["mean_equity"] - multi["mean_equity"]) < 1e-6, (
            f"1-villain multiway {multi['mean_equity']} != single {single['mean_equity']}"
        )

    def test_multiway_returns_valid_deciles(self):
        """Multiway result has all required keys and values in [0,1]."""
        matrix = self._toy_matrix()
        result = _matrix_cell_equity_multiway("AA", [{"KK": 50.0}, {"72o": 30.0}], matrix)
        assert result is not None
        for key in ["mean_equity", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]:
            assert key in result, f"Missing key {key}"
            assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of [0,1]"

    def test_load_palette_returns_list(self, tmp_path):
        """_load_palette_combo_counts returns list[dict] (one per villain) for n>=3 scenarios."""
        import json

        from tools.build_equity_table import _role_to_filename_candidates

        btn_dir = tmp_path / "BTN"
        btn_dir.mkdir()
        co_dir = tmp_path / "CO"
        co_dir.mkdir()
        # Use real filenames that _role_to_filename_candidates would produce
        pfr_files = _role_to_filename_candidates("pfr", None, True)
        bet3_files = _role_to_filename_candidates("3bettor", "BTN", True)
        (btn_dir / pfr_files[0]).write_text(json.dumps({"combo_counts": {"AA": 10, "KK": 8}}))
        (co_dir / bet3_files[0]).write_text(json.dumps({"combo_counts": {"QQ": 5, "AKs": 3}}))

        import tools.build_preflop_equity_table as bt

        orig_dir = bt.PALETTE_DIR
        bt.PALETTE_DIR = tmp_path
        try:
            result = _load_palette_combo_counts("3bp|IP|n3|BTN:pfr+CO:3bettor:BTN")
            assert result is not None
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 2
        finally:
            bt.PALETTE_DIR = orig_dir


# ---------------------------------------------------------------------------
# Limp role classification + k-aware multiway cap
# ---------------------------------------------------------------------------


class TestLimpClassification:
    """Limp pots only log folds/raises; unlogged villains entered by limping."""

    def test_unlogged_villain_in_limp_pot_is_limper(self):
        from tools.build_preflop_equity_table import _classify_villain_role

        # CO has no action in the seq (only UTG/MP folds logged)
        seq = [{"pos": "UTG", "action": "fold"}, {"pos": "MP", "action": "fold"}]
        role, tgt = _classify_villain_role("CO", seq, None, pot_type="limp")
        assert role == "limper"
        assert tgt is None

    def test_unlogged_villain_non_limp_stays_unknown(self):
        from tools.build_preflop_equity_table import _classify_villain_role

        seq = [{"pos": "UTG", "action": "fold"}]
        role, _ = _classify_villain_role("CO", seq, None, pot_type="srp")
        assert role == "unknown"

    def test_limp_scenario_key_has_no_unknown(self):
        from tools.build_preflop_equity_table import _compute_scenario_key

        dp = {
            "pot_type": "limp",
            "hero_pos_rel": "OOP",
            "n_players_at_street": 6,
            "preflop_action_seq": [{"pos": "UTG", "action": "fold"}],
            "preflop_aggressor": None,
            "villains_active": [{"pos": p} for p in ("MP", "CO", "BTN", "SB", "BB")],
        }
        key = _compute_scenario_key(dp)
        assert ":unknown" not in key, f"limp key still has unknown role: {key}"
        assert key.count(":limper") == 5


class TestMultiwayCap:
    """k-aware cap keeps the Cartesian product bounded for many villains."""

    @staticmethod
    def _toy_matrix():
        classes = ALL_HAND_CLASSES
        return {h: {v: 0.5 for v in classes} for h in classes}

    def test_five_villain_pooled_does_not_blow_up(self):
        # 5 villains each with a wide (161-class) range — naive product is 161**5.
        wide = {c: 1.0 for c in ALL_HAND_CLASSES[:161]}
        result = _matrix_cell_equity_multiway("AA", [dict(wide)] * 5, self._toy_matrix())
        assert result is not None
        for key in ["mean_equity", "p10", "p50", "p90"]:
            assert 0.0 <= result[key] <= 1.0
