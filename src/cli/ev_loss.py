"""ev-loss subcommand: compute KL divergence for a cluster_key.

Thin CLI shim over ``src.metrics.ev_loss.ev_loss``. Opens no DB connections
directly — the ev_loss function manages both TimescaleDB and Milvus connections
from env vars (or test-injected mocks).

Output on success:
    JSON to stdout: {"cluster_key": "...", "session_id": null, "ev_loss": 0.1234}
    Exit code 0.

Output on NoStrategyError:
    JSON to stderr: {"error": "no strategy", "detail": "..."}
    Exit code 1.

Usage (after ``uv pip install -e .`` or via src/cli/main.py dispatcher):
    poker-engine ev-loss "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop"
    poker-engine ev-loss <cluster_key> --session <session_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger

log = get_logger("cli.ev_loss")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Execute ev_loss for args.cluster_key and print JSON to stdout.

    Args:
        args: parsed argparse.Namespace with .cluster_key and .session attributes.
        _tsdb_conn: optional pre-built psycopg connection for test injection.
        _milvus: optional pre-built MilvusClient for test injection.

    Returns:
        0 on success, 1 on NoStrategyError, 2 on unexpected error.
    """
    from src._errors import NoStrategyError
    from src.metrics.ev_loss import ev_loss

    cluster_key: str = args.cluster_key
    session_id: str | None = getattr(args, "session", None)

    log.info("cli.ev_loss.start", cluster_key=cluster_key, session_id=session_id)

    try:
        kl = ev_loss(
            cluster_key,
            session_id=session_id,
            _tsdb_conn=_tsdb_conn,
            _milvus=_milvus,
        )
    except NoStrategyError as exc:
        error_payload = {"error": "no strategy", "detail": str(exc)}
        print(json.dumps(error_payload), file=sys.stderr)
        log.warning("cli.ev_loss.no_strategy", cluster_key=cluster_key, detail=str(exc))
        return 1
    except Exception as exc:  # pragma: no cover
        error_payload = {"error": "unexpected error", "detail": str(exc)}
        print(json.dumps(error_payload), file=sys.stderr)
        log.error("cli.ev_loss.error", cluster_key=cluster_key, error=str(exc))
        return 2

    result = {
        "cluster_key": cluster_key,
        "session_id": session_id,
        "ev_loss": kl,
    }
    print(json.dumps(result))
    log.info("cli.ev_loss.done", cluster_key=cluster_key, kl=round(kl, 6))
    return 0


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Compute ev_loss KL divergence for a cluster.")
    _parser.add_argument(
        "cluster_key",
        type=str,
        help="cluster_key string (e.g. 'street_class=postflop|pot_type=srp|...')",
    )
    _parser.add_argument(
        "--session",
        type=str,
        default=None,
        metavar="ID",
        help="Filter to a specific session_id (default: all sessions).",
    )
    sys.exit(run(_parser.parse_args()))
