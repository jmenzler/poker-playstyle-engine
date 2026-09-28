"""Unit tests for the ASCII sparkline utility (CLI-06, D-15).

Plan: 06-02. Pure function; no I/O. See src/study/sparkline.py.
"""

from __future__ import annotations

from src.study.sparkline import sparkline

_BLOCKS = "▁▂▃▄▅▆▇█"


def test_empty_input() -> None:
    """Empty input → empty string (no NaN, no error)."""
    assert sparkline([]) == ""


def test_returns_correct_length() -> None:
    """len(sparkline(values)) == len(values) for any non-empty input."""
    assert len(sparkline([1.0, 2.0, 3.0])) == 3
    assert len(sparkline([0.5] * 100)) == 100


def test_flat_line_lowest_block() -> None:
    """hi == lo degenerate case: every cell renders as the lowest block."""
    assert sparkline([1, 1, 1]) == _BLOCKS[0] * 3
    assert sparkline([0.42] * 5) == _BLOCKS[0] * 5


def test_monotonic_increasing_has_non_decreasing_blocks() -> None:
    """Monotonically increasing input → non-decreasing block heights; last is tallest."""
    out = sparkline([1, 2, 3, 4, 5, 6, 7, 8])
    for i in range(1, len(out)):
        assert _BLOCKS.index(out[i]) >= _BLOCKS.index(out[i - 1])
    assert out[-1] == _BLOCKS[-1]


def test_only_block_chars() -> None:
    """Every character of output is in the 8-block alphabet."""
    out = sparkline([0.1, 5.2, 2.7, 8.4, 3.0])
    for ch in out:
        assert ch in _BLOCKS, f"non-block char in output: {ch!r}"


def test_peak_in_middle() -> None:
    """Single peak in middle → middle char is the tallest in the row."""
    out = sparkline([1, 2, 3, 4, 3, 2, 1])
    peak_height = _BLOCKS.index(out[3])
    for i in (0, 1, 2, 4, 5, 6):
        assert _BLOCKS.index(out[i]) <= peak_height


def test_negative_values_handled() -> None:
    """Range may include negatives; relative scaling still works (min → ▁, max → █)."""
    out = sparkline([-1.0, 0.0, 1.0])
    assert len(out) == 3
    assert out[0] == _BLOCKS[0]
    assert out[-1] == _BLOCKS[-1]


def test_five_block_alphabet_size() -> None:
    """The block alphabet is exactly 8 characters."""
    assert len(_BLOCKS) == 8


def test_no_whitespace_in_output() -> None:
    """Output contains no whitespace, no ASCII chars, no non-block glyphs."""
    out = sparkline([10, 20, 30, 40, 50])
    assert " " not in out
    assert all(c in _BLOCKS for c in out)
