"""Tests for tools/phase2_pipeline.py — SHA256 sidecar-based resume (BOOT-05).

Stubs created during Phase 2 Plan 03 (test infra scaffolding). Activated by
Plan 08 — each test replaces the pytest.skip() body with real assertions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# pytestmark — uncomment when this file's tests need the live Milvus stack
# pytestmark = pytest.mark.integration


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_sha256_sidecar_skip_when_input_unchanged(tmp_path: Path) -> None:
    """BOOT-05: stage with matching input hash + present output -> SKIPPED."""
    from tools.phase2_pipeline import _input_sha256, _stage_done, _write_sidecar

    # Create a fake input file
    input_file = tmp_path / "input.jsonl"
    input_file.write_bytes(b"line1\nline2\n")

    # Create a fake output file (simulates a completed stage)
    output_file = tmp_path / "output.parquet"
    output_file.write_bytes(b"fake_parquet_data")

    # Compute the hash of the input file using stdlib directly (do NOT reuse impl)
    h = hashlib.sha256()
    h.update(input_file.read_bytes())
    expected_hash = h.hexdigest()

    # Write a matching sidecar
    _write_sidecar(output_file, expected_hash)

    # Now compute via _input_sha256 and assert _stage_done returns True
    current_hash = _input_sha256([input_file])
    assert current_hash == expected_hash
    assert _stage_done(output_file, current_hash) is True


def test_sha256_sidecar_rerun_when_input_changed(tmp_path: Path) -> None:
    """BOOT-05: input file modified -> sidecar hash mismatch -> stage RE-RUN."""
    from tools.phase2_pipeline import _stage_done, _write_sidecar

    # Create a fake input file and output file
    input_file = tmp_path / "input.jsonl"
    input_file.write_bytes(b"original_content")

    output_file = tmp_path / "output.parquet"
    output_file.write_bytes(b"fake_parquet_data")

    # Write a sidecar with a WRONG (stale) hash
    stale_hash = hashlib.sha256(b"different_content").hexdigest()
    _write_sidecar(output_file, stale_hash)

    # Compute the real current hash of the input
    current_hash = hashlib.sha256(b"original_content").hexdigest()

    # stale_hash != current_hash -> should return False (must re-run)
    assert stale_hash != current_hash
    assert _stage_done(output_file, current_hash) is False


def test_rebuild_from_stage_cascades_downstream() -> None:
    """--rebuild-from embed forces embed + zscore + upsert + validate to re-run."""
    from tools.phase2_pipeline import _rebuild_from_cascade

    forced = _rebuild_from_cascade("embed")
    assert forced == {"embed", "zscore", "upsert", "validate"}
    assert _rebuild_from_cascade("all") == {
        "replay",
        "extract",
        "preflop_equity",
        "embed",
        "zscore",
        "upsert",
        "validate",
    }
