"""Parser contracts exercised with fabricated hand-history grammar fixtures."""

from __future__ import annotations

from src.api.hm3 import _parse_observations

FIXTURE = """\
Poker Hand #RC1001: Hold'em No Limit  ($0.5/$1) - 2000/01/01 00:00:00
Table 'SyntheticSixMax' 6-max Seat #1 is the button
Seat 1: PlayerOne ($40 in chips)
Seat 2: SmallBlind ($50 in chips)
Seat 3: Hero ($80 in chips)
Seat 4: Raiser ($60 in chips)
Seat 5: Caller ($70 in chips)
Seat 6: PlayerSix ($90 in chips)
SmallBlind: posts small blind $0.5
Hero: posts big blind $1
*** HOLE CARDS ***
Dealt to Hero [Ac Qd]
Raiser: raises $2 to $3
Caller: calls $3
PlayerSix: folds
PlayerOne: folds
SmallBlind: folds
Hero: calls $2
*** FLOP *** [Qh 5d 8c]
Hero: checks
Raiser: bets $4
Caller: folds
Hero: calls $4
*** TURN *** [Qh 5d 8c] [2s]
Hero: checks
Raiser: checks
*** SHOWDOWN ***
Hero collected $17.5 from pot
*** SUMMARY ***
Total pot $17.5 | Rake $0
"""

HAND_ID = "RC1001"


def _events():
    evs = _parse_observations(FIXTURE, HAND_ID)
    assert evs, "fixture must yield at least one action event"
    return evs


def test_seat_map():
    """Each event carries a seat_map {seat_no: {name, stack}} from the Seat lines."""
    ev = _events()[0]
    assert "seat_map" in ev
    sm = ev["seat_map"]
    assert sm["3"]["name"] == "Hero"
    assert sm["3"]["stack"] == 80.0
    assert sm["1"]["name"] == "PlayerOne"
    assert len(sm) == 6


def test_button_seat():
    """button_seat parsed from the `Seat #N is the button` marker on the Table line."""
    ev = _events()[0]
    assert ev["button_seat"] == 1


def test_hero_seat():
    """hero_seat = the seat whose name == 'Hero'."""
    ev = _events()[0]
    assert ev["hero_seat"] == 3


def test_bet_amount_present_for_wagering_actions():
    """bets/calls/raises carry a float bet_amount; raises use the `to` amount."""
    evs = _events()
    raise_ev = next(e for e in evs if "raises" in e["action_taken"])
    assert raise_ev["bet_amount"] == 3.0
    bet_ev = next(e for e in evs if "bets" in e["action_taken"])
    assert bet_ev["bet_amount"] == 4.0
    call_ev = next(e for e in evs if "calls" in e["action_taken"])
    assert call_ev["bet_amount"] is not None


def test_bet_amount_none_for_passive_actions():
    """folds/checks carry bet_amount = None."""
    evs = _events()
    fold_ev = next(e for e in evs if "folds" in e["action_taken"])
    assert fold_ev["bet_amount"] is None
    check_ev = next(e for e in evs if "checks" in e["action_taken"])
    assert check_ev["bet_amount"] is None


def test_actor_field():
    """Each event carries the acting player's name as `actor` (for seat highlight wiring)."""
    ev = _events()[0]
    assert ev["actor"] == "Raiser"


def test_additive_fields():
    """Additive guarantee: every existing key is still present on each event."""
    expected = {
        "obs_id",
        "hand_id",
        "street",
        "hero_pos_rel",
        "cluster_key",
        "action_taken",
        "spot_features",
    }
    for ev in _events():
        assert expected <= set(ev), f"missing existing keys: {expected - set(ev)}"
        assert "board" in ev["spot_features"]
        assert "hero_hole" in ev["spot_features"]


# Truncated HH: action lines but no Seat/button markers (T-08-01 fail-safe).
MALFORMED_FIXTURE = "Hero: bets 5\nvillain: folds\n"


def test_malformed_hh_degrades_gracefully():
    """Missing seat/button lines never crash; v1 keys intact, new fields empty/None."""
    evs = _parse_observations(MALFORMED_FIXTURE, "TRUNC1")
    assert evs, "action lines should still yield events"
    ev = evs[0]
    assert {"obs_id", "hand_id", "street", "action_taken", "spot_features"} <= set(ev)
    assert ev["seat_map"] == {}
    assert ev["button_seat"] is None
    assert ev["hero_seat"] is None


def test_empty_hh_returns_empty():
    """No markers at all -> empty list (route renders raw-HH fallback)."""
    assert _parse_observations("", "X") == []


SHOWDOWN_FIXTURE = """\
Poker Hand #RC9: Hold'em No Limit  ($0.1/$0.25) - 2000/01/01 00:00:00
Table 'RC9' 6-max Seat #1 is the button
Seat 1: villain ($30 in chips)
Seat 2: Hero ($30 in chips)
*** HOLE CARDS ***
Dealt to Hero [As Kd]
villain: bets $1
Hero: calls $1
*** SHOW DOWN ***
villain: shows [Qh Qd] (a pair of Queens)
Hero: shows [As Kd] (high card Ace)
Hero collected $2 from pot
*** SUMMARY ***
"""


WALK_FIXTURE = """\
Poker Hand #RC8: Hold'em No Limit  ($0.1/$0.25) - 2000/01/01 00:00:00
Table 'RC8' 6-max Seat #1 is the button
Seat 1: villain ($30 in chips)
Seat 2: Hero ($30 in chips)
villain: posts small blind $0.1
Hero: posts big blind $0.25
*** HOLE CARDS ***
Dealt to Hero [As Kd]
villain: folds
*** SUMMARY ***
"""


def test_pot_includes_blinds():
    """Posted blinds feed the pot even on a walk (no bets/raises/calls)."""
    evs = _parse_observations(WALK_FIXTURE, "RC8")
    # First (only) action event: pot = SB 0.1 + BB 0.25 = 0.35.
    assert evs[0]["pot"] == 0.35


def test_big_blind_parsed():
    """big_blind is captured from the `posts big blind` line for bb-normalized display."""
    evs = _parse_observations(WALK_FIXTURE, "RC8")
    assert evs[0]["big_blind"] == 0.25


def test_street_committed_tracks_blinds_in_front():
    """street_committed exposes per-player chips in front (blinds before any action)."""
    evs = _parse_observations(WALK_FIXTURE, "RC8")
    sc = evs[0]["street_committed"]
    assert sc["Hero"] == 0.25  # BB in front of Hero
    assert sc["villain"] == 0.1  # SB in front of villain


RAISE_FIXTURE = """\
Poker Hand #RC7: Hold'em No Limit  ($0.5/$1) - 2000/01/01 00:00:00
Table 'RC7' 6-max Seat #1 is the button
Seat 1: villain ($100 in chips)
Seat 2: Hero ($100 in chips)
villain: posts small blind $0.5
Hero: posts big blind $1
*** HOLE CARDS ***
Dealt to Hero [As Kd]
villain: raises $1 to $2
Hero: calls $1
*** SUMMARY ***
"""


def test_pot_counts_raise_increment_not_to_amount():
    """`raises X to Y` tops the actor up to Y; the pot gains only the increment.

    SB 0.5 + BB 1 = 1.5. villain (SB) raises to 2 -> +1.5 (2 - 0.5 already in) = 3.0.
    Hero calls 1 -> +1 = 4.0. (Naively adding the to-amount would give 4.5.)
    """
    evs = _parse_observations(RAISE_FIXTURE, "RC7")
    assert evs[0]["pot"] == 3.0  # after villain's raise
    assert evs[-1]["pot"] == 4.0  # after Hero's call


def test_shown_cards_from_showdown():
    """Showdown `shows [..]` lines populate the additive hand-level shown_cards map."""
    evs = _parse_observations(SHOWDOWN_FIXTURE, "RC9")
    ev = evs[0]
    assert ev["shown_cards"]["villain"] == ["Qh", "Qd"]
    assert ev["shown_cards"]["Hero"] == ["As", "Kd"]
