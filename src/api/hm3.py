"""GET /api/hm3/hand|/api/hm3/replay — fetch + parse HH from the HM3 mirror.

The Phase 2 ingest pipeline writes Milvus only (D-07-11e); the observations
table is intentionally empty in the kNN topology. To replay a decision the
operator needs the original HH text, which lives in the HM3 SQLite database
mounted read-only at ``HM3_PATH`` inside the container.

The HM3 ``handhistories`` table keys on ``gamenumber``. Pokerstars zoom hands
share their integer hand_id directly with the gamenumber; Rush & Cash hands
have an ``RC`` prefix (e.g. ``RC1001``) — the Milvus ``decision_id``
strips that prefix when it captures the digits, so this endpoint tries both
shapes before giving up.

``/hm3/replay`` parses the HH text into observation-shaped events that the
existing React replayer can consume directly. Parsing is best-effort; if a
required marker is missing the endpoint returns an empty list rather than
raising — the UI then renders the raw HH text as a fallback.
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api", tags=["hm3"])

_HM3_PATH = os.environ.get("HM3_PATH", "/data/hm3.sqlite")


def _open() -> sqlite3.Connection:
    if not os.path.exists(_HM3_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"HM3 mirror not mounted at {_HM3_PATH!r}",
        )
    return sqlite3.connect(f"file:{_HM3_PATH}?mode=ro", uri=True)


def _fetch_hh(hand_id: str) -> tuple[str, str, str] | None:
    """Return ``(gamenumber, handtimestamp, handhistory)`` or None if absent."""
    conn = _open()
    try:
        cur = conn.cursor()
        for candidate in (f"RC{hand_id}", hand_id):
            cur.execute(
                "SELECT gamenumber, handtimestamp, handhistory "
                "FROM handhistories WHERE gamenumber = ? LIMIT 1",
                (candidate,),
            )
            row = cur.fetchone()
            if row is not None:
                return row[0], row[1], row[2]
        return None
    finally:
        conn.close()


@router.get("/hm3/hand")
async def get_hand(
    hand_id: Annotated[str, Query(min_length=1, max_length=64)],
) -> dict:
    """Return the raw handhistory text for ``hand_id``."""
    if not hand_id.isalnum():
        raise HTTPException(status_code=400, detail="hand_id must be alphanumeric")

    row = _fetch_hh(hand_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"hand_id {hand_id!r} not found in HM3 (tried RC{hand_id} and {hand_id})",
        )
    gamenumber, ts, hh = row
    return {
        "hand_id": hand_id,
        "gamenumber": gamenumber,
        "handtimestamp": ts,
        "handhistory": hh,
    }


_STREET_RE = re.compile(r"^\*\*\*\s+(FLOP|TURN|RIVER)\s+\*\*\*(.+)$")
_HOLE_RE = re.compile(r"^Dealt to (.+?)\s+\[(.+?)\]\s*$")  # group1=hero name, group2=cards
_ACTION_RE = re.compile(r"^(.+?):\s+(folds|checks|calls|bets|raises)\b(.*)$")

# Grammar locked in 08-WAVE0-FINDINGS.md (real HM3 data).
_SEAT_RE = re.compile(r"^Seat (\d+):\s+(.+?)\s+\(\$?([\d.]+) in chips")
_BUTTON_RE = re.compile(r"Seat #(\d+) is the button")  # not ^-anchored: on the Table line
_AMOUNT_RE = re.compile(r"\$?([\d.]+)")
_SHOWN_RE = re.compile(r"^(.+?):\s+shows\s+\[(.+?)\]")  # showdown reveal
_POST_RE = re.compile(r"^(.+?):\s+posts\b(.*)$")  # blinds + antes feed the pot


def _bet_amount(verb: str, rest: str) -> float | None:
    """Amount staked by this action. raises X to Y -> Y; folds/checks -> None."""
    if verb in ("folds", "checks"):
        return None
    nums = _AMOUNT_RE.findall(rest)
    if not nums:
        return None
    return float(nums[-1])  # for `raises X to Y` the last number is the to-amount


def _parse_observations(hh: str, hand_id: str) -> list[dict]:
    """Best-effort: HH text → list of observation-shaped events.

    One row per actor action across preflop/flop/turn/river. Each row carries
    the full current board + hero hole + actor name + action verb. The existing
    React replayer consumes ``street``, ``hero_pos_rel``, ``cluster_key``,
    ``action_taken``, and ``spot_features.{board,hero_hole}`` — only those need
    to be populated.
    """
    lines = hh.splitlines()
    hero_hole: list[str] = []
    board: list[str] = []
    street: str = "preflop"
    events: list[dict] = []

    # Hand-level constants: stable across every event (best-effort, never raise).
    seat_map: dict[str, dict] = {}
    button_seat: int | None = None
    shown_cards: dict[str, list[str]] = {}  # player name -> revealed cards (showdown)
    hero_name: str | None = None
    for line in lines:
        line = line.strip()
        m_seat = _SEAT_RE.match(line)
        if m_seat:
            seat_no, name, chips = m_seat.groups()
            seat_map[seat_no] = {"name": name, "stack": float(chips)}
            continue
        m_hole = _HOLE_RE.match(line)
        if m_hole:
            # "Dealt to <name>" identifies the hero by their actual screen name —
            # PokerStars uses it, not the literal "Hero" (RushAndCash anonymized).
            hero_name = m_hole.group(1)
            hero_hole = m_hole.group(2).split()
            continue
        m_shown = _SHOWN_RE.match(line)
        if m_shown:
            shown_cards[m_shown.group(1)] = m_shown.group(2).split()
            continue
        if button_seat is None:
            m_button = _BUTTON_RE.search(line)
            if m_button:
                button_seat = int(m_button.group(1))
    hero_seat: int | None = next((int(n) for n, s in seat_map.items() if s["name"] == hero_name), None)

    pot: float = 0.0
    big_blind: float | None = None
    # Chips each player has put in on the CURRENT street; reset at each new street.
    # Used to add only the incremental contribution to the pot (a `raises X to Y`
    # tops the player's commitment up to Y, not by Y) so the pot stays correct.
    committed: dict[str, float] = {}
    # Cumulative chips each player committed across ALL streets (never reset) — used
    # to render running stacks (start_stack - total committed) as the hand plays out.
    total_committed: dict[str, float] = {}
    # Chips from COMPLETED streets only — the displayed pot. Current-street bets show
    # as chips in front of each seat (street_committed) and fold in at street end, so
    # money isn't double-counted (in front AND in the pot) mid-street.
    pot_settled: float = 0.0
    folded: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        m_street = _STREET_RE.match(line)
        if m_street:
            label = m_street.group(1).lower()
            street = label
            pot_settled += sum(committed.values())  # current-street chips settle into the pot
            committed = {}  # new betting round
            # Board cards are bracketed; collect every [..] group on the marker line.
            board_groups = re.findall(r"\[([^\]]+)\]", m_street.group(2))
            collected: list[str] = []
            for grp in board_groups:
                collected.extend(grp.split())
            if collected:
                board = collected
            continue

        m_post = _POST_RE.match(line)
        if m_post:
            name, rest = m_post.groups()
            nums = _AMOUNT_RE.findall(rest)
            if nums:
                amt = float(nums[-1])
                pot += amt
                committed[name] = committed.get(name, 0.0) + amt
                total_committed[name] = total_committed.get(name, 0.0) + amt
                if "big blind" in rest and big_blind is None:
                    big_blind = amt
            continue

        m_action = _ACTION_RE.match(line)
        if m_action:
            actor, verb, _rest = m_action.groups()
            if verb == "folds":
                folded.append(actor)
            bet_amount = _bet_amount(verb, _rest)
            # Incremental pot contribution: bets/calls add the stated amount;
            # `raises ... to Y` adds only Y minus what the actor already committed.
            if bet_amount is not None:
                if verb == "raises":
                    delta = max(0.0, bet_amount - committed.get(actor, 0.0))
                    committed[actor] = bet_amount
                else:
                    delta = bet_amount
                    committed[actor] = committed.get(actor, 0.0) + bet_amount
                pot += delta
                total_committed[actor] = total_committed.get(actor, 0.0) + delta
            seat_map_now = {
                n: {"name": s["name"], "stack": round(s["stack"] - total_committed.get(s["name"], 0.0), 2)}
                for n, s in seat_map.items()
            }
            events.append(
                {
                    "obs_id": f"{hand_id}_evt{len(events) + 1}",
                    "hand_id": hand_id,
                    "ts": "",
                    "street": street,
                    "hero_pos_rel": "IP" if actor == hero_name else "OOP",
                    "cluster_key": f"hh_{hand_id}_{street}",
                    "action_taken": f"{actor}: {verb}",
                    "spot_features": {
                        "board": list(board),
                        "hero_hole": list(hero_hole),
                    },
                    "seat_map": seat_map_now,
                    "button_seat": button_seat,
                    "hero_seat": hero_seat,
                    "actor": actor,
                    "bet_amount": bet_amount,
                    "pot": round(pot_settled, 2),
                    "folded": list(folded),
                    "big_blind": big_blind,
                    # Per-player chips committed THIS street (snapshot) — the chips
                    # rendered in front of each seat until the round ends.
                    "street_committed": {n: round(v, 2) for n, v in committed.items() if v > 0},
                    "shown_cards": shown_cards,
                }
            )

    return events


def parse_hm_replay(hand_id: str) -> list[dict]:
    """Public wrapper: fetch HH from HM3 and return parsed observation-shaped events.

    Returns an empty list when the hand_id is not found in the HM3 mirror.
    Callers (e.g. the unified /api/hands/by-hand/{hand_id}/replay endpoint) use this
    to avoid duplicating the HH parsing logic from _parse_observations().
    """
    row = _fetch_hh(hand_id)
    if row is None:
        return []
    _gn, _ts, hh = row
    return _parse_observations(hh, hand_id)


@router.get("/hm3/replay")
async def replay(
    hand_id: Annotated[str, Query(min_length=1, max_length=64)],
) -> list[dict]:
    """Return observation-shaped events parsed from the HH text."""
    if not hand_id.isalnum():
        raise HTTPException(status_code=400, detail="hand_id must be alphanumeric")

    row = _fetch_hh(hand_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"hand_id {hand_id!r} not found in HM3 (tried RC{hand_id} and {hand_id})",
        )
    _gn, _ts, hh = row
    return _parse_observations(hh, hand_id)
