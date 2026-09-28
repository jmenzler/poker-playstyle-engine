"""Fabricated inputs for isolated tests, never production data or solver output."""

import json
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.solver import queue_driver
from tools.charts import build_preflop_charts
from tools.preflop_combo import ALL_COMBOS_169


def _normalization(dim: int) -> bytes:
    return json.dumps({
        "provenance": "fabricated test-only normalization",
        "feature_spec_version": 3,
        "dim_stats": [{"mean": 0.25, "std": 2.0} for _ in range(dim)],
    }).encode()


def _write_chart_inputs(root: Path) -> None:
    for seat, rate in [("BTN", 0.4), ("UTG", 0.18)]:
        folder = root / seat
        folder.mkdir(parents=True, exist_ok=True)
        rates = dict.fromkeys(ALL_COMBOS_169, rate)
        rates.update({"AA": 1.0, "AKo": 0.8, "72o": 0.0})
        (folder / "open.json").write_text(json.dumps({"inferred_rates": rates}))
    (root / "BTN/3bet_vs_CO.json").write_text(json.dumps({
        "inferred_rates": {"AA": 1.0, "AKo": 0.8, "KJo": 0.5},
    }))
    (root / "BTN/defend_vs_CO.json").write_text(json.dumps({
        "per_combo": {
            "44": {"n_total_dealt": 10, "breakdown": {"call": 7}},
            "KJo": {"n_total_dealt": 10, "breakdown": {"call": 2}},
        },
    }))
    (root / "BB").mkdir()
    (root / "BB/defend_vs_BTN.json").write_text(json.dumps({"combo_counts": {"AA": 2, "KQs": 1}}))
    (root / "CO").mkdir()
    (root / "CO/4bet_vs_BTN.json").write_text(json.dumps({"inferred_rates": {"AA": 0.8}}))


def _equity_table(path: Path) -> None:
    row = {"hole_canonical": ["AcKd"], "board_canonical": [""], "scenario": ["test-only"]}
    row.update({f"p{i}": [i / 100] for i in range(10, 100, 10)})
    row["mean_equity"] = [0.5]
    pq.write_table(pa.table(row), path)


def _install(root: Path, relative: str, content: bytes, cleanup: ExitStack) -> None:
    path = root / relative
    missing = []
    parent = path.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir()
        cleanup.callback(directory.rmdir)
    with path.open("xb") as output:
        output.write(content)
    cleanup.callback(path.unlink)


@pytest.fixture(scope="session", autouse=True)
def synthetic_export_inputs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    root = Path(__file__).resolve().parent.parent
    temporary = tmp_path_factory.mktemp("fabricated-export-inputs")
    source = temporary / "chart-inputs"
    _write_chart_inputs(source)
    equity = temporary / "equity.parquet"
    _equity_table(equity)
    with ExitStack() as cleanup, pytest.MonkeyPatch.context() as patch:
        # Legacy callers hard-code these paths; files exist only during pytest.
        _install(root, "tools/zscore_preflop.json", _normalization(34), cleanup)
        _install(root, "tools/zscore_postflop.json", _normalization(80), cleanup)
        _install(root, "tools/equity_table.parquet", equity.read_bytes(), cleanup)
        patch.setattr(build_preflop_charts, "_SRC_DIR", source)
        patch.setattr(queue_driver, "PALETTE_DIR", source)
        charts = temporary / "charts"
        build_preflop_charts.build(charts)
        for path in sorted(charts.rglob("*.json")):
            relative = "charts/preflop/inferred_6max/" + path.relative_to(charts).as_posix()
            _install(root, relative, path.read_bytes(), cleanup)
        yield
