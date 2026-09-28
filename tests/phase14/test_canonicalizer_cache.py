from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_EQUITY_TABLE_PRESENT = (_ROOT / "tools" / "equity_table.parquet").exists() or (
    _ROOT / "equity_table.parquet"
).exists()

requires_equity_table = pytest.mark.skipif(
    not _EQUITY_TABLE_PRESENT,
    reason="PC-gated: requires equity_table.parquet for the real Canonicalizer",
)

_SRP_SPOT = {
    "street": "preflop",
    "hero_hole": ["Ah", "Kh"],
    "hero_position": "BB",
    "action_sequence": ["BTN:open_2.5", "SB:fold"],
}


@requires_equity_table
def test_default_cached_returns_same_instance():
    from src.canonicalizer.encoder import Canonicalizer

    a = Canonicalizer.default()
    b = Canonicalizer.default()
    assert a is b


@requires_equity_table
def test_cache_does_not_change_cluster_key():
    """Regression: calling encode twice must return the same cluster_key."""
    from tools.build_embedding import encode_spot_to_cluster_key

    key1 = encode_spot_to_cluster_key(_SRP_SPOT)
    key2 = encode_spot_to_cluster_key(_SRP_SPOT)
    assert key1 == key2
