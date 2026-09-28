"""CLI-06 — `poker-engine dashboard` shim. D-15 + D-NEW-27 sectioned text.

Wraps ``src.study.dashboard.dashboard_snapshot``. Six-section vertical layout
per D-NEW-27 (verdict-led):

  1. LOOP HEALTH         — verdict + narrative
  2. EV_LOSS TREND       — ASCII sparkline (hero metric)
  3. RECENT ACTIVITY     — last-10 events
  4. SUBSYSTEM HEALTH    — db / milvus / solver / fastapi
  5. SESSION             — session in flight (or '(no session in flight)')
  6. KB GROWTH           — total + last-7d delta by source

Help text:
    CLI-06: Print 5-section dashboard (loop_health, ev_loss_trend, ...).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src._log import get_logger
from src.study.sparkline import sparkline

log = get_logger("cli.dashboard")


def run(args: argparse.Namespace, *, _tsdb_conn: Any = None, _milvus: Any = None) -> int:
    """Render the operator dashboard snapshot.

    Args:
        args: parsed argparse.Namespace with .format.
        _tsdb_conn: test-injection psycopg connection.
        _milvus: unused; accepted for shim symmetry.

    Returns:
        0 on success, 2 on unexpected error.
    """
    log.info("command_started", command="dashboard")
    try:
        from src.study.dashboard import dashboard_snapshot

        result = dashboard_snapshot(_tsdb_conn=_tsdb_conn)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        log.error("cli.dashboard.unexpected", error=str(exc))
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_render_text(result))
    log.info("command_complete", command="dashboard")
    return 0


def _render_text(s: dict) -> str:
    """Six-section vertical render per D-NEW-27."""
    out: list[str] = []

    # 1. LOOP HEALTH
    health = s.get("loop_health") or {}
    verdict = str(health.get("verdict", "unknown"))
    marker = {"improving": "✓", "stalled": "✗", "mixed": "~"}.get(verdict, "?")
    out.append("## LOOP HEALTH")
    out.append(f"  {marker} {verdict} · {health.get('narrative', '')}")
    out.append("")

    # 2. EV_LOSS TREND (hero) with ASCII sparkline
    trend = s.get("ev_loss_trend") or []
    vals = [float(r["ev_loss"]) for r in trend if r.get("ev_loss") is not None]
    out.append("## EV_LOSS TREND")
    out.append(f"  last {len(vals)} sessions: {sparkline(vals) if vals else '(no data)'}")
    if vals:
        out.append(f"  range: {min(vals):.3f} → {max(vals):.3f}    current: {vals[-1]:.3f}")
    out.append("")

    # 3. RECENT ACTIVITY
    out.append("## RECENT ACTIVITY")
    for r in (s.get("recent_activity") or [])[:10]:
        out.append(f"  {r.get('ts', '')}  {r.get('event_type', '')!s:<12}  {r.get('summary', '')}")
    out.append("")

    # 4. SUBSYSTEM HEALTH
    out.append("## SUBSYSTEM HEALTH")
    for k, v in (s.get("health") or {}).items():
        out.append(f"  ■ {k:<12} {v}")
    out.append("")

    # 5. SESSION
    sess = s.get("session")
    out.append("## SESSION")
    if sess is None:
        out.append("  (no session in flight)")
    else:
        out.append(f"  session_id: {sess.get('session_id')}  n_obs: {sess.get('n_obs')}")
        out.append(f"  started: {sess.get('start_ts')}   last activity: {sess.get('last_ts')}")
    out.append("")

    # 6. KB GROWTH
    kb = s.get("kb_growth") or {}
    out.append("## KB GROWTH")
    out.append(f"  total nodes: {kb.get('total_nodes', 0)}")
    for src, n in (kb.get("by_source") or {}).items():
        delta = (kb.get("last_7d_delta") or {}).get(src, 0)
        out.append(f"    {src:<10}  {n:>8}    last-7d: +{delta}")

    return "\n".join(out)
