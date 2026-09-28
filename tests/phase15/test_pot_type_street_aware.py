"""pot_type must count PREFLOP raises only.  long-ok

Root cause (2026-06-01): action_sequence flattens all streets with no delimiter,
so _infer_pot_type counted postflop raises (raise_half_pot/raise_pot) as preflop
raise levels -> limped pots mislabeled 3bet. Fix: a "/" street delimiter; the
derivation slices to the preflop segment (before the first "/").
"""

from __future__ import annotations

import pytest

from src.canonicalizer.encoder import _infer_pot_type

# (action_sequence, expected pot_type)
CASES = [
    # ---- preflop-only sequences (no delimiter) ----
    (("UTG:fold", "CO:fold", "BTN:call", "SB:fold", "BB:check"), "limp"),
    (("BTN:raise_2.5", "BB:call"), "srp"),
    (("BTN:raise_2.5", "BB:raise_pot", "BTN:call"), "3bet"),
    (("BTN:raise_2.5", "BB:raise_pot", "BTN:raise_pot", "BB:call"), "4bet"),
    (("SB:raise", "BB:raise", "SB:raise", "BB:raise", "SB:call"), "5bet+"),
    # ---- with street delimiter: postflop raises MUST NOT inflate the count ----
    # The KK bug: BTN limps, BB checks, then postflop aggression -> still 'limp'.
    (("BTN:call", "BB:check", "/", "BB:raise_half_pot", "BTN:raise_pot"), "limp"),
    (("BTN:raise_2.5", "BB:call", "/", "BB:bet", "BTN:raise_pot"), "srp"),
    (("BTN:raise_2.5", "BB:raise_pot", "BTN:call", "/", "BB:raise_half_pot"), "3bet"),
    # multiple streets (flop/turn/river) all delimited -> only preflop counts.
    (
        (
            "BTN:call",
            "BB:check",
            "/",
            "BB:check",
            "BTN:check",
            "/",
            "BB:raise_pot",
            "BTN:raise_pot",
            "/",
            "BB:raise_pot",
        ),
        "limp",
    ),
]


@pytest.mark.parametrize("seq,expected", CASES)
def test_infer_pot_type_counts_preflop_only(seq, expected):
    assert _infer_pot_type(seq) == expected, f"{seq} -> expected {expected}"


def test_action_tokens_inserts_delimiters_at_boundaries():
    from src.sim.adapter import _action_tokens

    rec = [(0, "x"), (1, "x"), (1, "x"), (0, "x"), (1, "x")]
    toks = _action_tokens(rec, dealer_id=0, street_boundaries=[3])
    assert toks[3] == "/", f"delimiter should precede the 4th action, got {toks}"
    assert toks.count("/") == 1
    assert "/" not in _action_tokens(rec, dealer_id=0), "no boundaries -> no delimiter"


def test_derive_preflop_info_villain_is_survivor_not_folded_raiser():
    from src.solver.queue_driver import _derive_preflop_info

    # BTN limps, SB raises, BB(hero) 3bets, BTN calls, SB folds.
    # Flop villain is BTN (the caller still in), NOT SB (folded raiser).
    seq = (
        "UTG:fold",
        "MP:fold",
        "CO:fold",
        "BTN:call",
        "SB:raise_half_pot",
        "BB:raise_pot",
        "BTN:call",
        "SB:fold",
    )
    _pot_type, villain, _opener, _bettor, _preflop = _derive_preflop_info(list(seq), "BB")
    assert villain == "BTN"


def test_derive_preflop_info_villain_is_last_aggressor_still_in():
    from src.solver.queue_driver import _derive_preflop_info

    # BTN opens, SB 3bets, BTN calls; hero BTN faces SB.
    seq = ("BTN:raise_2.5", "SB:raise_pot", "BB:fold", "BTN:call")
    _pot_type, villain, _opener, _bettor, _preflop = _derive_preflop_info(list(seq), "BTN")
    assert villain == "SB"


def test_derive_preflop_info_ignores_postflop_raises():
    from src.solver.queue_driver import _derive_preflop_info

    # BTN opens, BB 3bets, BTN calls; postflop both raise (must NOT affect derivation).
    seq = ("BTN:raise_2.5", "BB:raise_pot", "BTN:call", "/", "BB:bet", "BTN:raise_pot", "BB:raise_pot")
    pot_type, villain, opener, _bettor, preflop = _derive_preflop_info(seq, "BTN")
    assert pot_type == "3bet"
    assert villain == "BB"
    assert opener == "BTN"
    assert "/" not in preflop
