"""Tests for EquityLookup accepting multiple parquet paths.

TDD RED phase — tests for the multi-path list[Path] feature before it is implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.equity_lookup import EquityLookup

# ---------------------------------------------------------------------------
# Fixtures: two tiny parquet files with non-overlapping keys
# ---------------------------------------------------------------------------


def _make_equity_row(
    hole: str,
    board: str,
    scen: str,
    mean: float = 0.5,
    street: str = "postflop",
) -> dict:
    """Helper to create a single equity table row."""
    return {
        "id": f"{hole}|{board}|{scen}",
        "street": street,
        "hole_canonical": hole,
        "board_canonical": board,
        "scenario": scen,
        "mean_equity": mean,
        "p10": mean - 0.1,
        "p20": mean - 0.08,
        "p30": mean - 0.06,
        "p40": mean - 0.03,
        "p50": mean,
        "p60": mean + 0.03,
        "p70": mean + 0.06,
        "p80": mean + 0.08,
        "p90": mean + 0.1,
    }


_EQUITY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("street", pa.string()),
        pa.field("hole_canonical", pa.string()),
        pa.field("board_canonical", pa.string()),
        pa.field("scenario", pa.string()),
        pa.field("mean_equity", pa.float64()),
        pa.field("p10", pa.float64()),
        pa.field("p20", pa.float64()),
        pa.field("p30", pa.float64()),
        pa.field("p40", pa.float64()),
        pa.field("p50", pa.float64()),
        pa.field("p60", pa.float64()),
        pa.field("p70", pa.float64()),
        pa.field("p80", pa.float64()),
        pa.field("p90", pa.float64()),
    ]
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
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
        },
        schema=_EQUITY_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


@pytest.fixture
def path_a(tmp_path: Path) -> Path:
    """Postflop equity table with one key."""
    rows = [
        _make_equity_row(
            "AhKh",
            "QhJh2s",
            "srp|IP|n2|BB:caller:BTN",
            mean=0.72,
            street="flop",
        )
    ]
    p = tmp_path / "equity_table_a.parquet"
    _write_parquet(p, rows)
    return p


@pytest.fixture
def path_b(tmp_path: Path) -> Path:
    """Preflop equity table with a different key (board_canonical='')."""
    rows = [
        _make_equity_row(
            "AcKd",
            "",
            "srp|IP|n2|BB:caller:BTN",
            mean=0.65,
            street="preflop",
        )
    ]
    p = tmp_path / "preflop_equity_table.parquet"
    _write_parquet(p, rows)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEquityLookupMultiPath:
    """EquityLookup([path_a, path_b]) merges keys from both tables."""

    def test_single_path_still_works(self, path_a: Path) -> None:
        """Existing single-path usage is not broken."""
        eq = EquityLookup(path_a)
        result = eq.get("AhKh", "QhJh2s", "srp|IP|n2|BB:caller:BTN")
        assert result is not None
        assert abs(result["mean"] - 0.72) < 1e-6

    def test_list_path_merges_keys(self, path_a: Path, path_b: Path) -> None:
        """EquityLookup([path_a, path_b]) contains keys from both files."""
        eq = EquityLookup([path_a, path_b])
        # Key from path_a
        r_a = eq.get("AhKh", "QhJh2s", "srp|IP|n2|BB:caller:BTN")
        assert r_a is not None, "Key from path_a not found after merge"
        # Key from path_b (preflop, empty board)
        r_b = eq.get("AcKd", "", "srp|IP|n2|BB:caller:BTN")
        assert r_b is not None, "Key from path_b not found after merge"

    def test_list_path_correct_values(self, path_a: Path, path_b: Path) -> None:
        """Values from each table are correctly retained after merge."""
        eq = EquityLookup([path_a, path_b])
        r_a = eq.get("AhKh", "QhJh2s", "srp|IP|n2|BB:caller:BTN")
        r_b = eq.get("AcKd", "", "srp|IP|n2|BB:caller:BTN")
        assert abs(r_a["mean"] - 0.72) < 1e-6, f"path_a mean wrong: {r_a['mean']}"
        assert abs(r_b["mean"] - 0.65) < 1e-6, f"path_b mean wrong: {r_b['mean']}"

    def test_n_total_is_sum(self, path_a: Path, path_b: Path) -> None:
        """n_total reflects combined row count from all tables."""
        eq = EquityLookup([path_a, path_b])
        assert eq.n_total == 2

    def test_missing_file_in_list_is_skipped(self, path_a: Path, tmp_path: Path) -> None:
        """A missing file in the list is skipped gracefully (only existing loaded)."""
        missing = tmp_path / "nonexistent.parquet"
        eq = EquityLookup([path_a, missing])
        # Should still load path_a
        r = eq.get("AhKh", "QhJh2s", "srp|IP|n2|BB:caller:BTN")
        assert r is not None
        assert eq.n_total == 1

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        """Passing an empty list raises ValueError."""
        with pytest.raises((ValueError, FileNotFoundError)):
            EquityLookup([])

    def test_list_with_one_path_same_as_single(self, path_a: Path) -> None:
        """EquityLookup([path]) behaves identically to EquityLookup(path)."""
        eq_single = EquityLookup(path_a)
        eq_list = EquityLookup([path_a])
        assert eq_single.n_total == eq_list.n_total
        r1 = eq_single.get("AhKh", "QhJh2s", "srp|IP|n2|BB:caller:BTN")
        r2 = eq_list.get("AhKh", "QhJh2s", "srp|IP|n2|BB:caller:BTN")
        assert r1 == r2

    def test_family_fallback_independent_of_file_order(self, tmp_path: Path) -> None:
        """Same hole|board|prefix in two files: the fallback row must not depend on
        which file (or row) loaded first."""
        rows_x = [_make_equity_row("AhKh", "QhJh2s", "srp|IP|n2|BB:caller:BTN", mean=0.10)]
        rows_y = [_make_equity_row("AhKh", "QhJh2s", "srp|IP|n2|CO:caller:BTN", mean=0.90)]
        px = tmp_path / "x.parquet"
        py = tmp_path / "y.parquet"
        _write_parquet(px, rows_x)
        _write_parquet(py, rows_y)

        miss = ("AhKh", "QhJh2s", "srp|IP|n2|UTG:caller:BTN")
        r_xy = EquityLookup([px, py]).get(*miss)
        r_yx = EquityLookup([py, px]).get(*miss)
        assert r_xy is not None and r_yx is not None
        assert r_xy["mean"] == r_yx["mean"]

    def test_duplicate_exact_key_across_files_raises(self, tmp_path: Path) -> None:
        """Two merged files carrying the same exact key fail loud, not silently shadow."""
        rows = [_make_equity_row("AhKh", "QhJh2s", "srp|IP|n2|BB:caller:BTN", mean=0.5)]
        px = tmp_path / "dup_x.parquet"
        py = tmp_path / "dup_y.parquet"
        _write_parquet(px, rows)
        _write_parquet(py, rows)
        with pytest.raises(ValueError, match=r"(?i)duplicate"):
            EquityLookup([px, py])
