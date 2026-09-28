"""RunCheckpoint atomic read/write round-trip and resume-safety tests."""

from __future__ import annotations

from pathlib import Path

from src.autoloop.checkpoint import (
    RunCheckpoint,
    checkpoint_path_for,
    read_checkpoint,
    write_checkpoint,
)


def _sample(run_id: str = "r1", last_cycle: int = 2) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=run_id,
        last_cycle=last_cycle,
        patches_accepted=3,
        patches_rejected=1,
        base_seed=7,
        run_epoch="2026-05-21T00:00:00+00:00",
        session_cursor=f"{run_id}_c{last_cycle}",
    )


def test_read_missing_returns_none(tmp_path: Path) -> None:
    assert read_checkpoint(tmp_path / "nope.json") is None, "absent checkpoint is None"


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    ckpt = _sample()
    path = tmp_path / "r1.json"
    write_checkpoint(path, ckpt)

    back = read_checkpoint(path)
    assert back == ckpt, "round-trip equality"


def test_write_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "deep" / "r1.json"
    write_checkpoint(path, _sample())
    assert path.exists(), "parent dirs created"


def test_no_tmp_file_left(tmp_path: Path) -> None:
    path = tmp_path / "r1.json"
    write_checkpoint(path, _sample())
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"no .tmp residue after atomic write, found {leftovers}"


def test_checkpoint_path_for(tmp_path: Path) -> None:
    assert checkpoint_path_for("soak-1", tmp_path) == tmp_path / "soak-1.json"


def test_overwrite_advances_last_cycle(tmp_path: Path) -> None:
    path = tmp_path / "r1.json"
    write_checkpoint(path, _sample(last_cycle=1))
    write_checkpoint(path, _sample(last_cycle=5))

    back = read_checkpoint(path)
    assert back is not None
    assert back.last_cycle == 5, "overwrite reflects new last_cycle"
