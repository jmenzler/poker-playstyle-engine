"""Equity-miss ledger in tools/build_embedding.py — activated by Plan 04.

Tests verify CONTEXT.md Decision 2C:
- DPs with no equity hit are NOT written to hm_embeddings_chunk_*.parquet
- Equity-miss DPs ARE written to equity_misses.parquet with correct schema
- No zero-vector embeddings appear in chunk parquets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _write_equity_table(path: Path, *, with_miss_hole: bool = False) -> None:
    """Write a minimal equity_table.parquet.

    with_miss_hole=True means we deliberately do NOT include a row for "AcKd"
    so that DPs with hero_hole=["Ah","Ks"] are equity misses.
    """
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
    # Always include a row for "ThTs" (pair, canonical = ThTs or TcTd)
    # This row covers DPs that SHOULD be included in chunk output
    from tools.joint_canonicalize import joint_canonicalize

    t_can, _ = joint_canonicalize(["Th", "Ts"], [])

    rows: dict[str, list] = {
        "hole_canonical": [t_can],
        "board_canonical": [""],
        "scenario": ["srp|IP|n2|vNA"],
        "p10": [0.5],
        "p20": [0.55],
        "p30": [0.6],
        "p40": [0.65],
        "p50": [0.7],
        "p60": [0.72],
        "p70": [0.75],
        "p80": [0.78],
        "p90": [0.8],
        "mean_equity": [0.65],
    }

    if not with_miss_hole:
        # Also include AcKd so AhKs DPs DO get equity hits
        ak_can, _ = joint_canonicalize(["Ah", "Ks"], [])
        rows["hole_canonical"].append(ak_can)
        rows["board_canonical"].append("")
        rows["scenario"].append("srp|IP|n2|vNA")
        for col in ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]:
            rows[col].append(0.6)
        rows["mean_equity"].append(0.6)

    pq.write_table(pa.table(rows, schema=schema), path)


def _make_preflop_dp(dp_id: str, hero_hole: list[str], hole_class: str) -> dict:
    """Create a minimal preflop DP dict."""
    from tools.joint_canonicalize import joint_canonicalize

    h_can, _ = joint_canonicalize(hero_hole, [])
    return {
        "id": dp_id,
        "hand_id": dp_id.split("_")[0],
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
        "hero_hole": hero_hole,
        "hero_hole_class": hole_class,
        "hole_canonical": h_can,
        "board": [],
        "board_canonical": "",
        "preflop_aggressor": "",
        "preflop_action_seq": [],
        "action_so_far_street": [],
        "facing": "cold",
        "facing_pos": None,
        "facing_size_pot_frac": None,
    }


def test_equity_miss_skips_dp_from_embedding_output(tmp_path: Path) -> None:
    """Decision 2C: DP with no equity lookup is NOT written to hm_embeddings_chunk_*.parquet."""
    equity_path = tmp_path / "equity.parquet"
    decisions_path = tmp_path / "decisions.jsonl"
    out_dir = tmp_path / "chunks"

    # Table WITHOUT AhKs row — AhKs DPs will be equity misses
    _write_equity_table(equity_path, with_miss_hole=True)

    miss_dp = _make_preflop_dp("hand_miss_dp0", ["Ah", "Ks"], "AKo")
    hit_dp = _make_preflop_dp("hand_hit_dp0", ["Th", "Ts"], "TTo")

    with open(decisions_path, "w") as f:
        f.write(json.dumps(miss_dp) + "\n")
        f.write(json.dumps(hit_dp) + "\n")

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

    # Collect all ids from chunk parquets
    chunk_ids: set[str] = set()
    for chunk_file in out_dir.glob("hm_embeddings_chunk_*_preflop.parquet"):
        tbl = pq.read_table(chunk_file)
        chunk_ids.update(tbl.column("id").to_pylist())

    assert "hand_miss_dp0" not in chunk_ids, "Equity-miss DP should NOT appear in chunk parquet"
    assert "hand_hit_dp0" in chunk_ids, "Non-miss DP should appear in chunk parquet"


def test_equity_miss_written_to_ledger_parquet(tmp_path: Path) -> None:
    """Decision 2C: ledger row has dp_id, hole_class, board_canonical, street, pot_type, n_players_active."""
    equity_path = tmp_path / "equity.parquet"
    decisions_path = tmp_path / "decisions.jsonl"
    out_dir = tmp_path / "chunks"

    # Table WITHOUT AhKs row
    _write_equity_table(equity_path, with_miss_hole=True)

    miss_dp = _make_preflop_dp("hand_miss_dp0", ["Ah", "Ks"], "AKo")

    with open(decisions_path, "w") as f:
        f.write(json.dumps(miss_dp) + "\n")

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

    ledger_path = out_dir / "equity_misses.parquet"
    assert ledger_path.exists(), "equity_misses.parquet should be created"

    tbl = pq.read_table(ledger_path)
    assert tbl.num_rows == 1, f"Expected 1 ledger row; got {tbl.num_rows}"

    row = {col: tbl.column(col)[0].as_py() for col in tbl.schema.names}
    assert row["dp_id"] == "hand_miss_dp0"
    assert row["hole_class"] == "AKo"
    assert row["board_canonical"] == ""
    assert row["street"] == "preflop"
    assert row["pot_type"] == "srp"
    assert row["n_players_active"] == 2

    # Verify ledger schema columns
    expected_cols = {"dp_id", "hole_class", "board_canonical", "street", "pot_type", "n_players_active"}
    assert expected_cols.issubset(set(tbl.schema.names)), (
        f"Missing ledger columns: {expected_cols - set(tbl.schema.names)}"
    )


def test_no_silent_zero_vector_in_chunks(tmp_path: Path) -> None:
    """Invariant: every embedding row in chunks has norm > 0 (no zero-vector placeholders)."""
    equity_path = tmp_path / "equity.parquet"
    decisions_path = tmp_path / "decisions.jsonl"
    out_dir = tmp_path / "chunks"

    # Table WITH AhKs row — all DPs have equity hits
    _write_equity_table(equity_path, with_miss_hole=False)

    n = 50
    with open(decisions_path, "w") as f:
        for i in range(n):
            dp = _make_preflop_dp(f"hand{i}_dp0", ["Ah", "Ks"], "AKo")
            f.write(json.dumps(dp) + "\n")

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

    for chunk_file in out_dir.glob("hm_embeddings_chunk_*_preflop.parquet"):
        tbl = pq.read_table(chunk_file)
        embeddings = tbl.column("embedding").to_pylist()
        assert len(embeddings) > 0, f"Chunk {chunk_file.name} is empty"
        for i, emb in enumerate(embeddings):
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            assert norm > 0, (
                f"Zero-vector found at row {i} in {chunk_file.name}. "
                "Equity misses must go to ledger, not chunks."
            )
