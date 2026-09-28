"""On-the-fly equity for query spots that miss the sparse corpus equity table."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from tools.equity_compute import EquityComputer, _default_range_for_scenario
from tools.equity_lookup import EquityLookup

_DECILES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90]


def _ok_response(deciles: list[float] = _DECILES, mean: float = 0.78) -> str:
    return json.dumps({"id": "q", "deciles": deciles, "mean": mean, "n_runouts": 1081}) + "\n"


def _patched_run(stdout: str, returncode: int = 0):
    def run(*args, **kwargs):
        m = MagicMock()
        m.stdout = stdout
        m.stderr = ""
        m.returncode = returncode
        return m

    return run


def _executable(tmp_path: Path) -> Path:
    p = tmp_path / "equity-decile"
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return p


# --- default range selection ---------------------------------------------------


def test_default_range_differs_by_pot_type() -> None:
    srp = _default_range_for_scenario("srp|IP|n2")
    threebet = _default_range_for_scenario("3bet|OOP|n2")
    assert srp and threebet and srp != threebet


def test_default_range_unknown_pot_type_falls_back() -> None:
    assert _default_range_for_scenario("weird|x|n9")  # non-empty fallback


# --- EquityComputer.compute ----------------------------------------------------


def test_compute_parses_deciles_into_equity_dict(tmp_path: Path) -> None:
    ec = EquityComputer(_executable(tmp_path))
    with patch("tools.equity_compute.subprocess.run", _patched_run(_ok_response())):
        out = ec.compute("KhKs", "9h7c2s", "srp|IP|n2")
    assert out is not None
    assert out["p10"] == 0.50 and out["p90"] == 0.90
    assert abs(out["mean"] - 0.78) < 1e-6
    assert "variance" in out


def test_compute_sends_hole_board_and_pot_type_range(tmp_path: Path) -> None:
    ec = EquityComputer(_executable(tmp_path))
    captured: dict = {}

    def run(*args, **kwargs):
        captured["input"] = kwargs.get("input")
        m = MagicMock()
        m.stdout = _ok_response()
        m.returncode = 0
        return m

    with patch("tools.equity_compute.subprocess.run", run):
        ec.compute("KhKs", "9h7c2s", "srp|IP|n2")
    req = json.loads(captured["input"].strip())
    assert req["hole"] == "KhKs"
    assert req["board"] == "9h7c2s"
    assert req["villain_range"] == _default_range_for_scenario("srp|IP|n2")


def test_compute_returns_none_on_error_response(tmp_path: Path) -> None:
    ec = EquityComputer(_executable(tmp_path))
    err = json.dumps({"id": "q", "error": "bad range"}) + "\n"
    with patch("tools.equity_compute.subprocess.run", _patched_run(err)):
        assert ec.compute("KhKs", "9h7c2s", "srp|IP|n2") is None


def test_compute_caches_by_key(tmp_path: Path) -> None:
    ec = EquityComputer(_executable(tmp_path))
    calls = {"n": 0}

    def run(*args, **kwargs):
        calls["n"] += 1
        m = MagicMock()
        m.stdout = _ok_response()
        m.returncode = 0
        return m

    with patch("tools.equity_compute.subprocess.run", run):
        ec.compute("KhKs", "9h7c2s", "srp|IP|n2")
        ec.compute("KhKs", "9h7c2s", "srp|IP|n2")
    assert calls["n"] == 1


def test_unavailable_binary_returns_none_without_crashing() -> None:
    ec = EquityComputer(Path("/nonexistent/equity-decile"))
    assert not ec.is_available()
    assert ec.compute("KhKs", "9h7c2s", "srp|IP|n2") is None


# --- EquityLookup on_miss hook -------------------------------------------------


def _one_row_table(tmp_path: Path) -> Path:
    path = tmp_path / "eq.parquet"
    cols = {
        "hole_canonical": ["AhAd"],
        "board_canonical": ["2c2d2h"],
        "scenario": ["srp|IP|n2|BTN:pfr"],
        "mean_equity": [0.9],
        **{c: [0.9] for c in ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]},
    }
    pq.write_table(pa.table(cols), path)
    return path


def test_equity_lookup_invokes_on_miss(tmp_path: Path) -> None:
    sentinel = {"p10": 0.1, "mean": 0.5, "variance": 0.0}
    calls: list = []

    def on_miss(hole: str, board: str, scen: str) -> dict:
        calls.append((hole, board, scen))
        return sentinel

    eq = EquityLookup(_one_row_table(tmp_path), on_miss=on_miss)
    out = eq.get("KhKs", "9h7c2s", "srp|IP|n2")  # absent (hole,board) → double miss
    assert out is sentinel
    assert calls == [("KhKs", "9h7c2s", "srp|IP|n2")]


def test_equity_lookup_without_on_miss_returns_none(tmp_path: Path) -> None:
    eq = EquityLookup(_one_row_table(tmp_path))
    assert eq.get("KhKs", "9h7c2s", "srp|IP|n2") is None
