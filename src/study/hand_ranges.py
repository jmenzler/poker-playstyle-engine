"""Orchestration layer: solve + persist + contract assembly for per-hand ranges."""  # long-ok-file

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src._log import get_logger
from src.solver.hand_to_harvest import build_hand_harvest
from src.solver.postflop_cli import PostflopCliBackend
from src.study.hands import get_replay_by_hand
from src.study.harvest_ranges import (
    HarvestRangeRow,
    build_seat_payloads,
    load_harvest_ranges_by_hand,
    persist_harvest_range,
)

__all__ = ["load_hand_ranges_contract", "solve_and_persist_hand_ranges"]

log = get_logger("study.hand_ranges")


def _default_solver() -> PostflopCliBackend:
    """Build a PostflopCliBackend from POSTFLOP_CLI_BIN env var."""
    binary = Path(
        os.environ.get("POSTFLOP_CLI_BIN", "~/postflop-cli/target/release/postflop-cli")
    ).expanduser()
    return PostflopCliBackend(binary_path=binary)


def _maybe_push(job: Any, event: str, data: dict[str, Any]) -> None:
    """Push an event to job.events if job is provided. Non-blocking; never raises."""
    if job is None:
        return
    try:
        job.events.put_nowait({"event": event, "data": data})
    except Exception:
        log.warning("study.hand_ranges.event_push_failed", event=event)


def _map_hero_action(raw: str | None, actions: list[str]) -> str | None:
    """Map raw action_taken token to one of the solver action labels.

    Returns the matching label if found; None if unmappable (fail-soft — the
    cell highlight is optional; we must not fabricate a label).
    """
    if raw is None:
        return None
    # Exact match first.
    if raw in actions:
        return raw
    # Normalized match: case-insensitive, stripped.
    raw_norm = raw.strip().upper()
    for label in actions:
        if label.strip().upper() == raw_norm:
            return label
    return None


def solve_and_persist_hand_ranges(
    hand_id: str,
    *,
    force: bool = False,
    conn: Any,
    solver: Any = None,
    job: Any = None,
) -> dict[str, Any]:
    """Solve all HU postflop DPs for a hand and persist both seats per nav_ok DP.

    On cache hit (rows already exist for hand_id), returns immediately unless
    force=True.  Fails loudly on unknown hand_id, unavailable solver, or
    per-block invariant violations.

    Returns:
        {"status": "cached", "n_dps": int}  on cache hit
        {"status": "solved", "n_dps": int, "exploitability_pct": float}  on solve
    """
    if not force:
        existing = load_harvest_ranges_by_hand(hand_id, _tsdb_conn=conn)
        if existing:
            n_dps = len({r.decision_id for r in existing})
            log.info("study.hand_ranges.cache_hit", hand_id=hand_id, n_dps=n_dps)
            _maybe_push(job, "done", {"status": "cached", "hand_id": hand_id, "n_dps": n_dps})
            return {"status": "cached", "n_dps": n_dps}

    hand_rows = get_replay_by_hand(hand_id, _tsdb_conn=conn)
    if not hand_rows:
        raise ValueError(f"unknown hand_id {hand_id!r} — no observations found")

    active_solver = solver if solver is not None else _default_solver()
    if not active_solver.is_available():
        raise RuntimeError(
            "postflop-cli binary not available — deploy the harvest binary to POSTFLOP_CLI_BIN"
        )

    target_expl = float(os.environ.get("POSTFLOP_TARGET_EXPL", "1.0"))
    solve_timeout_s = float(os.environ.get("POSTFLOP_SOLVE_TIMEOUT_S", "600"))
    mem_budget_mb = int(os.environ.get("POSTFLOP_MEM_BUDGET_MB", "6000"))
    spec = build_hand_harvest(
        hand_rows, target_exploitability_pct=target_expl, memory_budget_mb=mem_budget_mb
    )

    _maybe_push(job, "progress", {"stage": "solving", "hand_id": hand_id})

    top, results = active_solver.solve_harvest_raw(spec.spot, spec.nav_lines, timeout_s=solve_timeout_s)

    dp_row_by_id = {r["decision_id"]: r for r in hand_rows}

    n_dps_solved = 0
    for meta, res in zip(spec.dp_meta, results, strict=True):
        multiway_turn_river = meta.multiway and meta.street in ("turn", "river")
        sentinel_payload_base = {"street": meta.street, "hero_seat": meta.hero_seat}

        if multiway_turn_river:
            _persist_sentinel(
                hand_id, meta.decision_id, "multiway_hu_unsupported", sentinel_payload_base, conn
            )
            n_dps_solved += 1
            _maybe_push(
                job,
                "progress",
                {"stage": "dp", "decision_id": meta.decision_id, "narrowing": "multiway_hu_unsupported"},
            )
            continue

        if not res["nav_ok"]:
            _persist_sentinel(hand_id, meta.decision_id, "nav_failed", sentinel_payload_base, conn)
            n_dps_solved += 1
            _maybe_push(
                job, "progress", {"stage": "dp", "decision_id": meta.decision_id, "narrowing": "nav_failed"}
            )
            continue

        hero = meta.hero_seat

        # hero_action: top-level action_taken from the matching hand row.
        hand_row = dp_row_by_id.get(meta.decision_id)
        raw_action = hand_row.get("action_taken") if hand_row else None
        hero_action = _map_hero_action(raw_action, res["actions"])

        hero_payload, villain_payload = build_seat_payloads(
            res,
            street=meta.street,
            hero_seat=hero,
            hero_action=hero_action,
            decision_id=meta.decision_id,
        )

        persist_harvest_range(
            HarvestRangeRow(hand_id=hand_id, decision_id=meta.decision_id, seat=hero, payload=hero_payload),
            _tsdb_conn=conn,
        )
        persist_harvest_range(
            HarvestRangeRow(
                hand_id=hand_id, decision_id=meta.decision_id, seat=1 - hero, payload=villain_payload
            ),
            _tsdb_conn=conn,
        )
        n_dps_solved += 1
        _maybe_push(job, "progress", {"stage": "dp", "decision_id": meta.decision_id, "narrowing": "ok"})

    if "exploitability_pct" not in top:
        raise ValueError("solver harvest result missing exploitability_pct")
    expl = top["exploitability_pct"]
    out: dict[str, Any] = {"status": "solved", "n_dps": n_dps_solved, "exploitability_pct": expl}
    _maybe_push(job, "done", out)
    log.info("study.hand_ranges.solved", hand_id=hand_id, n_dps=n_dps_solved, exploitability_pct=expl)
    return out


def _persist_sentinel(
    hand_id: str,
    decision_id: str,
    narrowing: str,
    base: dict[str, Any],
    conn: Any,
) -> None:
    """Persist sentinel rows (seat 0 and seat 1) for multiway/nav_failed DPs."""
    payload = {"narrowing": narrowing, **base}
    for seat in (0, 1):
        persist_harvest_range(
            HarvestRangeRow(hand_id=hand_id, decision_id=decision_id, seat=seat, payload=payload),
            _tsdb_conn=conn,
        )


def load_hand_ranges_contract(hand_id: str, *, conn: Any) -> list[dict[str, Any]]:
    """Load the frozen per-DP dual-seat JSON contract for a hand.

    Returns one entry per hero DP (dp0..dpN order).  Entries where narrowing
    != "ok" carry no oop/ip.  multiway is DERIVED from narrowing — never read
    from a stored field.
    """
    rows = load_harvest_ranges_by_hand(hand_id, _tsdb_conn=conn)
    if not rows:
        return []

    # Group rows by decision_id, preserving dp0..dpN order.
    seen: dict[str, list[HarvestRangeRow]] = {}
    order: list[str] = []
    for row in rows:
        if row.decision_id not in seen:
            seen[row.decision_id] = []
            order.append(row.decision_id)
        seen[row.decision_id].append(row)

    contract: list[dict[str, Any]] = []
    for decision_id in order:
        group = seen[decision_id]
        # Read street/hero_seat/narrowing from the first row's payload.
        ref_payload = group[0].payload
        street = ref_payload["street"]
        hero_seat = ref_payload["hero_seat"]
        narrowing = ref_payload["narrowing"]

        # DERIVE multiway — never read a stored multiway field.
        multiway = narrowing == "multiway_hu_unsupported"

        if narrowing != "ok":
            contract.append(
                {
                    "decision_id": decision_id,
                    "street": street,
                    "hero_seat": hero_seat,
                    "multiway": multiway,
                    "narrowing": narrowing,
                    "oop": None,
                    "ip": None,
                }
            )
            continue

        # Build per-seat objects from seat-keyed rows.
        row_by_seat = {r.seat: r for r in group}
        oop_row = row_by_seat.get(0)
        ip_row = row_by_seat.get(1)

        oop_obj = _build_seat_obj(oop_row.payload if oop_row else {}, hero_seat == 0)
        ip_obj = _build_seat_obj(ip_row.payload if ip_row else {}, hero_seat == 1)

        contract.append(
            {
                "decision_id": decision_id,
                "street": street,
                "hero_seat": hero_seat,
                "multiway": False,
                "narrowing": "ok",
                "oop": oop_obj,
                "ip": ip_obj,
            }
        )

    return contract


def _build_seat_obj(payload: dict[str, Any], is_hero: bool) -> dict[str, Any]:
    """Assemble one seat's contract object from a persisted payload.

    Always includes combos/weights/equity. When is_hero=True also includes
    strategy/ev_detail/actions/hero_action (these keys exist only in the
    hero-seat payload).
    """
    obj: dict[str, Any] = {
        "combos": payload.get("combos", []),
        "weights": payload.get("weights", []),
        "equity": payload.get("equity", []),
    }
    if is_hero:
        obj["strategy"] = payload.get("strategy", [])
        obj["ev_detail"] = payload.get("ev_detail", [])
        obj["actions"] = payload.get("actions", [])
        obj["hero_action"] = payload.get("hero_action")
    return obj
