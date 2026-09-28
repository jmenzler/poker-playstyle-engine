"""Executable spec for the action_sequence -> cluster_key token contract.

Tokens are "POS:verb"; raise stems open*/raise*/3b*/4b*/5b* count as raises,
giving pot_type limp/srp/3bet/4bet/5bet+. Gated on equity_table.parquet.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_EQUITY_TABLE_PRESENT = (_ROOT / "tools" / "equity_table.parquet").exists() or (
    _ROOT / "equity_table.parquet"
).exists()

requires_equity_table = pytest.mark.skipif(
    not _EQUITY_TABLE_PRESENT,
    reason="PC-gated: requires equity_table.parquet for the real Canonicalizer",
)

# Representative raised-pot spots reused by Test A/B and the Test E equivalence proof.
_SRP_SPOT = {
    "street": "preflop",
    "hero_hole": ["Ah", "Kh"],
    "hero_position": "BB",
    "action_sequence": ["BTN:open_2.5", "SB:fold"],
}
_3BET_SPOT = {
    "street": "preflop",
    "hero_hole": ["Ah", "Kh"],
    "hero_position": "BTN",
    "action_sequence": ["CO:open_2.5", "BTN:3bet_8"],
}


@requires_equity_table
def test_single_open_buckets_srp_not_limp():
    """Test A: one raise-stem token -> pot_type=srp (NOT the empty-sequence limp default)."""
    from tools.build_embedding import encode_spot_to_cluster_key

    cluster_key = encode_spot_to_cluster_key(_SRP_SPOT)
    assert "pot_type=srp" in cluster_key
    assert "pot_type=limp" not in cluster_key


@requires_equity_table
def test_two_raises_bucket_3bet():
    """Test B: two raise-stem tokens (open + 3bet) -> pot_type=3bet."""
    from tools.build_embedding import encode_spot_to_cluster_key

    cluster_key = encode_spot_to_cluster_key(_3BET_SPOT)
    assert "pot_type=3bet" in cluster_key


@requires_equity_table
def test_empty_action_sequence_buckets_limp():
    """Test C: empty action_sequence -> pot_type=limp.

    Documents the Phase-10 default the spot-builder MUST avoid by emitting real
    tokens — an empty sequence matches no raised-pot cluster_key.
    """
    from tools.build_embedding import encode_spot_to_cluster_key

    spot = {"street": "preflop", "hero_hole": ["Ah", "Kh"], "hero_position": "BTN", "action_sequence": []}
    cluster_key = encode_spot_to_cluster_key(spot)
    assert "pot_type=limp" in cluster_key


@requires_equity_table
def test_hero_pos_rel_from_last_raiser_preflop():
    """Test D: hero_pos_rel derives from the last raiser's seat on preflop.

    BB facing a BTN open is OOP (BB acts before BTN postflop); BTN facing a UTG
    open is IP (BTN acts last).
    """
    from tools.build_embedding import encode_spot_to_cluster_key

    bb_vs_btn = encode_spot_to_cluster_key(
        {
            "street": "preflop",
            "hero_position": "BB",
            "hero_hole": ["Ah", "Kh"],
            "action_sequence": ["BTN:open_2.5"],
        }
    )
    assert "hero_pos_rel=OOP" in bb_vs_btn

    btn_vs_utg = encode_spot_to_cluster_key(
        {
            "street": "preflop",
            "hero_position": "BTN",
            "hero_hole": ["Ah", "Kh"],
            "action_sequence": ["UTG:open_2.5"],
        }
    )
    assert "hero_pos_rel=IP" in btn_vs_utg


_FLOP_SPOT_WITH_POSTFLOP = {
    "street": "flop",
    "hero_position": "BTN",
    "hero_hole": ["Ac", "Ah"],
    "board": ["Jc", "Jd", "Ad"],
    "action_sequence": ("BTN:open_2.5", "CO:call", "CO:check", "BTN:bet_50"),
}
_FLOP_SPOT_WITHOUT_POSTFLOP = {
    "street": "flop",
    "hero_position": "BTN",
    "hero_hole": ["Ac", "Ah"],
    "board": ["Jc", "Jd", "Ad"],
    "action_sequence": ("BTN:open_2.5", "CO:call"),
}


@requires_equity_table
def test_postflop_tape_tokens_affect_encoding():
    """Postflop action tape must affect the embedding vector even when the bucket cluster_key stays constant.

    v1 design (D-07-11b): postflop cluster_keys are bucket-only — {street_class, pot_type,
    hero_pos_rel, n_players_active}. The full postflop action history lives in the 80-dim
    embedding vector, not the hard_filter. Probe Mode B retrieval finds neighbors by COSINE
    over those vectors within the bucket, so two spots that differ ONLY in postflop tape
    SHOULD share a cluster_key but have distinct vectors. Extending hard_filter with
    postflop-tape-derived fields would break byte-identity with the 282k existing PC rows.
    """
    import numpy as np

    from tools.build_embedding import encode_spot_to_cluster_key, encode_spot_to_embedding

    key_with = encode_spot_to_cluster_key(_FLOP_SPOT_WITH_POSTFLOP)
    key_without = encode_spot_to_cluster_key(_FLOP_SPOT_WITHOUT_POSTFLOP)
    assert key_with == key_without  # v1 bucket-only contract

    vec_with = encode_spot_to_embedding(_FLOP_SPOT_WITH_POSTFLOP)
    vec_without = encode_spot_to_embedding(_FLOP_SPOT_WITHOUT_POSTFLOP)
    assert not np.array_equal(vec_with, vec_without)  # encoding DID see postflop tape

    seq = _FLOP_SPOT_WITH_POSTFLOP["action_sequence"]
    assert any(t.startswith(f"{_FLOP_SPOT_WITH_POSTFLOP['hero_position']}:") for t in seq)
    assert not any(t.startswith("HERO:") or t.startswith("VILL:") for t in seq)


def test_postflop_hero_aggression_dims_nonzero_when_hero_bets():
    """_parse_action_tokens attributes bet to the real seat, not a synthetic literal."""
    from src.canonicalizer.encoder import _parse_action_tokens

    hero_pos = "BTN"
    real_seat_tokens = ("BTN:open_2.5", "CO:call", "CO:check", "BTN:bet_50")
    parsed_real = _parse_action_tokens(real_seat_tokens)
    hero_bet_steps = [s for s in parsed_real if s["pos"] == hero_pos and s["action"] == "bet"]
    assert len(hero_bet_steps) == 1, "hero BTN:bet_50 must be attributed to pos=BTN, action=bet"

    synthetic_tokens = ("BTN:open_2.5", "CO:call", "CO:check", "HERO:bet_50")
    parsed_synthetic = _parse_action_tokens(synthetic_tokens)
    hero_bet_synthetic = [s for s in parsed_synthetic if s["pos"] == hero_pos and s["action"] == "bet"]
    assert len(hero_bet_synthetic) == 0, "synthetic HERO: token must not match real hero_pos"


@requires_equity_table
@pytest.mark.parametrize("spot", [_SRP_SPOT, _3BET_SPOT], ids=["srp", "3bet"])
def test_encode_endpoint_equals_probe_by_spot_cluster_key(spot):
    """probe_by_spot derives the SAME cluster_key encode does (shared encoder).

    Patches probe_by_cluster_key to capture its cluster_key arg (no DB/Milvus),
    then asserts the captured key == encode_spot_to_cluster_key(spot).
    """
    import src.study.probe as probe_mod
    from tools.build_embedding import encode_spot_to_cluster_key

    captured: dict[str, str] = {}

    def _capture(cluster_key, *args, **kwargs):
        captured["cluster_key"] = cluster_key
        return {}

    with patch.object(probe_mod, "probe_by_cluster_key", side_effect=_capture):
        probe_mod.probe_by_spot(spot)

    assert captured["cluster_key"] == encode_spot_to_cluster_key(spot)
