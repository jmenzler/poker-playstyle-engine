"""Unit tests for src/solver/postflop_cli.py.

All subprocess interaction is mocked — no real binary required.
TDD: Task 1 (dataclasses), Task 2 (is_available), Task 3 (solve JSON payload),
     Task 4 (non-zero exit — skipped, activates in Phase 5).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Task 1 — SolverSpot and SolverResult dataclasses
# ---------------------------------------------------------------------------


def test_solver_spot_construction():
    """Minimal SolverSpot builds without error."""
    from src.solver.postflop_cli import SolverSpot

    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    assert spot.pot == 56
    assert spot.effective_stack == 172
    assert spot.board == ["Ah", "7c", "2d"]
    assert spot.range_ip == "AA"
    assert spot.range_oop == "AA"


def test_solver_result_construction():
    """Minimal SolverResult builds without error."""
    from src.solver.postflop_cli import SolverResult

    result = SolverResult(
        action_dist={"call": 1.0},
        exploitability_pct=0.5,
        solve_time_ms=100,
    )
    assert result.action_dist == {"call": 1.0}
    assert result.exploitability_pct == 0.5
    assert result.solve_time_ms == 100


def test_solver_spot_is_frozen():
    """SolverSpot is immutable — assigning to a field raises."""
    from src.solver.postflop_cli import SolverSpot

    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    with pytest.raises((TypeError, AttributeError)):
        spot.pot = 100  # type: ignore[misc]


def test_solver_result_is_frozen():
    """SolverResult is immutable — assigning to a field raises."""
    from src.solver.postflop_cli import SolverResult

    result = SolverResult(
        action_dist={"call": 1.0},
        exploitability_pct=0.5,
        solve_time_ms=100,
    )
    with pytest.raises((TypeError, AttributeError)):
        result.exploitability_pct = 9.9  # type: ignore[misc]


def test_solver_spot_optional_fields():
    """Optional bet_sizes_* and iteration fields default to None / defaults."""
    from src.solver.postflop_cli import SolverSpot

    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    # Optional fields must have defaults (None or reasonable value)
    assert spot.max_iterations is None or isinstance(spot.max_iterations, int)
    assert spot.target_exploitability_pct is None or isinstance(spot.target_exploitability_pct, float)


# ---------------------------------------------------------------------------
# Task 2 — is_available()
# ---------------------------------------------------------------------------


def test_is_available_returns_true_for_existing_file(tmp_path: Path):
    """is_available() is True when the binary path points to an existing file."""
    from src.solver.postflop_cli import PostflopCliBackend

    fake_bin = tmp_path / "fake-postflop-cli"
    fake_bin.touch()
    backend = PostflopCliBackend(binary_path=fake_bin)
    assert backend.is_available() is True


def test_is_available_returns_false_for_nonexistent_path():
    """is_available() is False when the binary path does not exist."""
    from src.solver.postflop_cli import PostflopCliBackend

    backend = PostflopCliBackend(binary_path=Path("/nonexistent/postflop-cli"))
    assert backend.is_available() is False


def test_is_available_returns_false_for_directory(tmp_path: Path):
    """is_available() is False when path is a directory (not a file)."""
    from src.solver.postflop_cli import PostflopCliBackend

    backend = PostflopCliBackend(binary_path=tmp_path)
    assert backend.is_available() is False


# ---------------------------------------------------------------------------
# Task 3 — solve() JSON payload + NotImplementedError
# ---------------------------------------------------------------------------


def _make_mock_run_result(returncode: int = 0) -> MagicMock:
    mock_result = MagicMock()
    mock_result.returncode = returncode
    mock_result.stdout = '{"exploitability_pct": 0.457, "time_ms": 93914}'
    mock_result.stderr = ""
    return mock_result


def test_solve_calls_subprocess_run_with_binary(tmp_path: Path):
    """solve() passes binary_path as first arg to subprocess.run."""
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    fake_bin = tmp_path / "postflop-cli"
    fake_bin.touch()
    backend = PostflopCliBackend(binary_path=fake_bin)
    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    with patch("subprocess.run", return_value=_make_mock_run_result()) as mock_run:
        with pytest.raises(NotImplementedError):
            backend.solve(spot)
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    # First positional arg is the command list; first element is binary path str
    cmd = call_args[0][0]
    assert cmd[0] == str(fake_bin)


def test_solve_sends_correct_json_payload(tmp_path: Path):
    """solve() serializes SolverSpot fields into the subprocess input JSON."""
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    fake_bin = tmp_path / "postflop-cli"
    fake_bin.touch()
    backend = PostflopCliBackend(binary_path=fake_bin)
    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA,KK",
        range_oop="AA:0.5,KK",
    )
    with patch("subprocess.run", return_value=_make_mock_run_result()) as mock_run:
        with pytest.raises(NotImplementedError):
            backend.solve(spot)
    call_kwargs = mock_run.call_args[1]
    input_str = call_kwargs["input"]
    payload = json.loads(input_str)
    # Required postflop-cli input keys per SOLVER.md §postflop-cli Interface
    assert payload["pot"] == 56
    assert payload["effective_stack"] == 172
    assert payload["board"] == ["Ah", "7c", "2d"]
    assert payload["range_ip"] == "AA,KK"
    assert payload["range_oop"] == "AA:0.5,KK"


def test_solve_uses_correct_subprocess_flags(tmp_path: Path):
    """solve() passes capture_output=True, text=True, and timeout kwarg."""
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    fake_bin = tmp_path / "postflop-cli"
    fake_bin.touch()
    backend = PostflopCliBackend(binary_path=fake_bin)
    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    with patch("subprocess.run", return_value=_make_mock_run_result()) as mock_run:
        with pytest.raises(NotImplementedError):
            backend.solve(spot, timeout_s=120.0)
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs.get("capture_output") is True or (
        call_kwargs.get("stdout") is not None and call_kwargs.get("stderr") is not None
    )
    assert call_kwargs.get("text") is True
    assert call_kwargs.get("timeout") == 120.0


def test_solve_raises_not_implemented_with_phase5_and_oq1(tmp_path: Path):
    """solve() raises NotImplementedError mentioning Phase 5 and OQ-1."""
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    fake_bin = tmp_path / "postflop-cli"
    fake_bin.touch()
    backend = PostflopCliBackend(binary_path=fake_bin)
    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    with patch("subprocess.run", return_value=_make_mock_run_result()):
        with pytest.raises(NotImplementedError, match="Phase 5"):
            backend.solve(spot)

    with patch("subprocess.run", return_value=_make_mock_run_result()):
        with pytest.raises(NotImplementedError, match="OQ-1"):
            backend.solve(spot)


# ---------------------------------------------------------------------------
# Task 4 — non-zero exit raises NoStrategyError (skipped — activates Phase 5)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Activates in Phase 5 when solve() removes NotImplementedError per OQ-1")
def test_solve_raises_no_strategy_error_on_nonzero_exit(tmp_path: Path):
    """Non-zero subprocess exit raises NoStrategyError with exit code in message."""
    from src._errors import NoStrategyError
    from src.solver.postflop_cli import PostflopCliBackend, SolverSpot

    fake_bin = tmp_path / "postflop-cli"
    fake_bin.touch()
    backend = PostflopCliBackend(binary_path=fake_bin)
    spot = SolverSpot(
        pot=56,
        effective_stack=172,
        board=["Ah", "7c", "2d"],
        range_ip="AA",
        range_oop="AA",
    )
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "bad input: invalid range"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(NoStrategyError, match="postflop-cli exited 1"):
            backend.solve(spot)
