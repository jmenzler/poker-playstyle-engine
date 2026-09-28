"""Autonomous-run checkpoint: atomic on-disk JSON state for crash-safe resume.

One checkpoint file per run_id. Written atomically (temp file + os.replace) so a
crash mid-write never corrupts the existing checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import msgspec


class RunCheckpoint(msgspec.Struct, frozen=True, kw_only=True):
    """Per-run progress state. Resume starts at last_cycle + 1.

    run_epoch is the frozen run-start timestamp (ISO-8601 UTC) used to re-derive
    each cycle's deterministic session_started_at on resume.
    """

    run_id: str
    last_cycle: int
    patches_accepted: int
    patches_rejected: int
    base_seed: int
    run_epoch: str
    session_cursor: str


def checkpoint_path_for(run_id: str, checkpoint_dir: str | Path) -> Path:
    """Path of the checkpoint file for a run, under checkpoint_dir."""
    return Path(checkpoint_dir) / f"{run_id}.json"


def read_checkpoint(path: str | Path) -> RunCheckpoint | None:
    """Return the checkpoint at path, or None if the file is absent.

    A corrupt file raises msgspec.DecodeError (fail loudly — never resume from garbage).
    """
    p = Path(path)
    if not p.exists():
        return None
    return msgspec.json.decode(p.read_bytes(), type=RunCheckpoint)


def write_checkpoint(path: str | Path, checkpoint: RunCheckpoint) -> None:
    """Atomically write a checkpoint to path (temp file in same dir + os.replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(msgspec.json.encode(checkpoint))
    os.replace(tmp, p)  # atomic rename — a crash leaves the old or new file, never a partial
