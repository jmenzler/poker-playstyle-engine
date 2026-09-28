"""CLI-03 — `poker-engine edit-node <cluster_key>` shim. D-11.

Mutually exclusive flags:
  --set 'fold:0.3,call:0.5,bet_50:0.2'  — script-friendly, parsed inline.
  --editor                              — opens $EDITOR with JSON template.

If neither flag is given, defaults to --editor.

Exit codes:
    0 — success
    1 — expected error (parse failure, ValidationError, PatchForbiddenError,
        PatchConflictError, or bare ValueError from the backend)
    2 — unexpected error
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.edit_node")


def _parse_set_string(s: str) -> dict[str, float]:
    """Parse ``'fold:0.3,call:0.5,bet_50:0.2'`` → ``{'fold':0.3,...}``.

    Raises:
        ValueError: a token does not contain ':' (e.g. ``'fold0.3'``).
    """
    out: dict[str, float] = {}
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"--set token missing ':': {token!r}")
        key, value = token.split(":", 1)
        out[key.strip()] = float(value.strip())
    return out


def _editor_loop(cluster_key: str, current_dist: dict[str, float] | None) -> dict[str, float]:
    """Open $EDITOR on a temp JSON file; return the parsed action_dist on save.

    Mitigates T-06-25 (path traversal): the temp file path is built by
    ``tempfile.NamedTemporaryFile(prefix='poker-engine-edit-')`` — never accepts
    user-supplied paths.
    """
    editor = os.environ.get("EDITOR", "vi")
    template = {
        "_help": ("Edit action_dist. Sum must be 1.000 ± 0.001. Keys must be canonical actions."),
        "cluster_key": cluster_key,
        "action_dist": current_dist or {"check": 1.0},
    }
    with tempfile.NamedTemporaryFile(
        "w+",
        suffix=".json",
        prefix="poker-engine-edit-",
        delete=False,
    ) as f:
        json.dump(template, f, indent=2)
        tmp_path = f.name
    try:
        try:
            subprocess.run([editor, tmp_path], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            raise ValueError(f"editor {editor!r} failed: {exc}") from exc
        with open(tmp_path) as f:
            edited = json.load(f)
        if "action_dist" not in edited:
            raise ValueError("edited JSON is missing the 'action_dist' key")
        return edited["action_dist"]
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Validate inputs → resolve action_dist (--set or --editor) → call edit_node.

    Args:
        args: parsed argparse.Namespace with .cluster_key, .set, .editor,
              .reason, .format.
        _tsdb_conn: test-injection psycopg connection.
        _milvus: test-injection MilvusClient.

    Returns:
        0 on success, 1 on expected errors, 2 on unexpected errors.
    """
    log.info("command_started", command="edit-node", cluster_key=args.cluster_key)
    try:
        set_value = getattr(args, "set", None)
        editor_flag = getattr(args, "editor", False)
        if set_value:
            action_dist = _parse_set_string(set_value)
        elif editor_flag or not set_value:
            # Default to editor mode when neither flag is given.
            action_dist = _editor_loop(args.cluster_key, current_dist=None)
        else:
            raise ValueError("either --set or --editor must be provided")
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.edit_node.parse_failed", error=str(exc))
        return 1

    try:
        from src.study.edit_node import edit_node

        result = edit_node(
            cluster_key=args.cluster_key,
            action_dist=action_dist,
            reason=getattr(args, "reason", "") or "",
            _tsdb_conn=_tsdb_conn,
            _milvus=_milvus,
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.edit_node.failed", error=str(exc))
        return 1
    except Exception as exc:
        # Catch src._errors.ValidationError + PatchForbiddenError + PatchConflictError
        # by class name so this module does not need to import them eagerly.
        if exc.__class__.__name__ in (
            "ValidationError",
            "PatchForbiddenError",
            "PatchConflictError",
        ):
            print(
                json.dumps({"error": str(exc), "type": exc.__class__.__name__}),
                file=sys.stderr,
            )
            log.error(
                "cli.edit_node.expected",
                error=str(exc),
                type=exc.__class__.__name__,
            )
            return 1
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.edit_node.unexpected", error=str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            f"patch applied: patch_id={result.get('patch_id')} "
            f"cluster_key={args.cluster_key} source={result.get('source', 'manual')}"
        )
    log.info("command_complete", command="edit-node", patch_id=result.get("patch_id"))
    return 0
