"""rollback subcommand: roll back a single patch by patch_id.

Usage:
    poker-engine rollback <patch_id-uuid-string>

Output on success:
    JSON to stdout: {"status": "rolled_back", "patch_id": "<uuid>"}
    Exit code 0.

Output on invalid UUID:
    JSON to stderr: {"error": "invalid patch_id: ..."}
    Exit code 1.

Output on rollback failure (LookupError or RuntimeError):
    JSON to stderr: {"error": "..."}
    Exit code 1.

Output on unexpected error:
    JSON to stderr: {"error": "..."}
    Exit code 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger
from src.patch_engine import PatchEngine, PatchRecord

log = get_logger("cli.rollback")


@dataclass(frozen=True)
class RollbackResult:
    """Library return type for rollback_patch — decoupled from CLI output."""

    rollback_patch_id: str  # UUID of the new rollback audit row
    restored_node_id: str | None  # UUID of the reactivated node (None if first-ever patch)
    source: str  # inherited from original patch
    ts: str  # ISO-8601 UTC timestamp


def rollback_patch(
    patch_id: _uuid.UUID,
    *,
    tsdb_conn: Any = None,
    milvus_client: Any = None,
) -> RollbackResult:
    """Roll back a patch by patch_id — pure library function.

    Wraps PatchEngine.rollback without any I/O or argparse. Suitable for
    direct use by the FastAPI rollback endpoint.

    Args:
        patch_id: UUID of the patch to roll back.
        tsdb_conn: optional TSDB connection (test-injection hook).
        milvus_client: optional Milvus client (test-injection hook).

    Returns:
        RollbackResult with rollback metadata.

    Raises:
        LookupError: if patch_id not found.
        RuntimeError: if patch already rolled back.
    """
    engine = PatchEngine()
    record: PatchRecord = engine.rollback(patch_id, _tsdb_conn=tsdb_conn, _milvus=milvus_client)
    return RollbackResult(
        rollback_patch_id=str(record.patch_id),
        restored_node_id=str(record.prev_node_id) if record.prev_node_id is not None else None,
        source=record.status,  # 'rolled_back'
        ts=record.ts,
    )


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Roll back a patch and print result JSON to stdout.

    CLI shim around rollback_patch. Formats the result and writes to stdout.

    Args:
        args: parsed argparse.Namespace with .patch_id (UUID string).
        _tsdb_conn: test-injection hook passed through to PatchEngine.rollback.
        _milvus: test-injection hook passed through to PatchEngine.rollback.

    Returns:
        0 on success, 1 on invalid UUID or rollback failure, 2 on unexpected error.
    """
    try:
        patch_id = _uuid.UUID(args.patch_id)
    except ValueError as exc:
        print(json.dumps({"error": f"invalid patch_id: {exc}"}), file=sys.stderr)
        log.error("cli.rollback.invalid_uuid", arg=args.patch_id)
        return 1

    try:
        result = rollback_patch(patch_id, tsdb_conn=_tsdb_conn, milvus_client=_milvus)
    except (LookupError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.rollback.failed", patch_id=str(patch_id), error=str(exc))
        return 1
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.rollback.unexpected", patch_id=str(patch_id), error=str(exc))
        return 2

    print(json.dumps({"status": "rolled_back", "patch_id": result.rollback_patch_id}))
    log.info("cli.rollback.done", patch_id=result.rollback_patch_id, status="rolled_back")
    return 0
