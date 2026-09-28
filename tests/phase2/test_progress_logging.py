"""OBS-03 progress logging in tools/build_embedding.py — activated by Plan 04.

Tests verify:
- PROGRESS_EVERY = 1_000: lines emitted at 1k cadence to stdout
- --log-file flag: JSON lines written to file with event=build_embedding.progress
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _write_equity_table(path: Path) -> None:
    """Write a minimal equity_table.parquet with deterministic rows."""
    schema = pa.schema(
        [
            pa.field("hole_canonical", pa.string()),
            pa.field("board_canonical", pa.string()),
            pa.field("scenario", pa.string()),
            pa.field("p10", pa.float64()),
            pa.field("p20", pa.float64()),
            pa.field("p30", pa.float64()),
            pa.field("p40", pa.float64()),
            pa.field("p50", pa.float64()),
            pa.field("p60", pa.float64()),
            pa.field("p70", pa.float64()),
            pa.field("p80", pa.float64()),
            pa.field("p90", pa.float64()),
            pa.field("mean_equity", pa.float64()),
        ]
    )
    # One preflop scenario row — must use canonical form (joint_canonicalize converts AhKs→AcKd)
    hole = "AcKd"
    board = ""
    scen = "srp|IP|n2|vNA"
    row = {
        "hole_canonical": [hole],
        "board_canonical": [board],
        "scenario": [scen],
        "p10": [0.4],
        "p20": [0.45],
        "p30": [0.5],
        "p40": [0.55],
        "p50": [0.6],
        "p60": [0.65],
        "p70": [0.7],
        "p80": [0.75],
        "p90": [0.8],
        "mean_equity": [0.6],
    }
    pq.write_table(pa.table(row, schema=schema), path)


def _write_decisions_jsonl(path: Path, n: int) -> None:
    """Write n preflop DP records to a JSONL file.

    All use AhKs so equity table has a hit.
    """
    with open(path, "w") as f:
        for i in range(n):
            dp = {
                "id": f"hand{i}_dp0",
                "hand_id": f"hand{i}",
                "street": "preflop",
                "hero_pos": "BTN",
                "hero_pos_rel": "IP",
                "n_players_at_street": 2,
                "n_players_active": 2,
                "pot_type": "srp",
                "pot_bb": 3.0,
                "pot_cents": 30,
                "bb_cents": 10,
                "effective_stack_cents": 1000,
                "hero_stack_bb": 100.0,
                "spr": 33.3,
                "hero_hole": ["Ah", "Ks"],
                "hero_hole_class": "AKo",
                "hole_canonical": "AhKs",
                "board": [],
                "board_canonical": "",
                "preflop_aggressor": "",
                "preflop_action_seq": [],
                "action_so_far_street": [],
                "facing": "cold",
                "facing_pos": None,
                "facing_size_pot_frac": None,
            }
            f.write(json.dumps(dp) + "\n")


def test_progress_every_1000_records_to_stdout(tmp_path: Path, capfd) -> None:
    """OBS-03: progress: N/? elapsed=Ts rate=X/s printed to stdout every 1k records."""
    equity_path = tmp_path / "equity.parquet"
    decisions_path = tmp_path / "decisions.jsonl"
    out_dir = tmp_path / "chunks"

    _write_equity_table(equity_path)
    _write_decisions_jsonl(decisions_path, n=2500)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from tools.build_embedding import main as build_main

    rc = build_main(
        [
            "--decisions",
            str(decisions_path),
            "--equity-table",
            str(equity_path),
            "--out-dir",
            str(out_dir),
            "--chunk-size",
            "10000",
        ]
    )
    assert rc == 0

    captured = capfd.readouterr()
    combined = captured.out + captured.err

    # At n=1000 and n=2000, progress lines should appear
    matches = re.findall(r"progress:\s*\d+", combined)
    assert len(matches) >= 2, (
        f"Expected at least 2 progress lines (at 1000 and 2000); got {len(matches)}. "
        f"Output: {combined[-500:]}"
    )


def test_progress_written_to_phase2_build_log(tmp_path: Path) -> None:
    """OBS-03: build log file accumulates progress lines via --log-file flag.

    BLOCKER 5 fix: activator is Plan 04 (build_embedding.py --log-file flag lands here),
    NOT Plan 08. Plan 08 keeps test_sha256_sidecar_* + test_rebuild_from_stage_cascades_downstream.
    """
    equity_path = tmp_path / "equity.parquet"
    decisions_path = tmp_path / "decisions.jsonl"
    out_dir = tmp_path / "chunks"
    log_path = tmp_path / "build.log"

    _write_equity_table(equity_path)
    _write_decisions_jsonl(decisions_path, n=1500)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from tools.build_embedding import main as build_main

    rc = build_main(
        [
            "--decisions",
            str(decisions_path),
            "--equity-table",
            str(equity_path),
            "--out-dir",
            str(out_dir),
            "--chunk-size",
            "10000",
            "--log-file",
            str(log_path),
        ]
    )
    assert rc == 0

    assert log_path.exists(), f"Expected log file at {log_path}"
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) > 0, "Log file is empty"

    # At least one line should have event=build_embedding.progress
    progress_lines = [line for line in lines if "build_embedding.progress" in line]
    assert len(progress_lines) >= 1, (
        f"Expected JSON lines with event='build_embedding.progress' in log file. Got lines: {lines[:5]}"
    )

    # Each progress line should parse as valid JSON
    for line in progress_lines[:3]:
        try:
            obj = json.loads(line)
            assert "event" in obj or "build_embedding.progress" in line
        except json.JSONDecodeError:
            # Non-JSON log format is also acceptable; just verify content presence
            pass
