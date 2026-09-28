from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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


def test_no_node_knn_returns_neighbors_not_empty_list():
    """When strategy_nodes row is None, _knn_neighbors must not early-return []."""
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    mock_client = MagicMock()
    mock_client.search.return_value = [[]]

    import src.study.probe as probe_mod

    fake_embedding = np.zeros(34, dtype=np.float64)
    with patch.object(probe_mod, "_encode_embedding_for_spot", return_value=fake_embedding):
        result = probe_mod._knn_neighbors(
            mock_conn,
            mock_client,
            "street_class=preflop|pot_type=srp|hero_pos_rel=IP|n_players_active=2",
            spot=_SRP_SPOT,
            k=5,
        )
    assert isinstance(result, list)
    assert mock_client.search.called


@requires_equity_table
def test_no_node_path_applies_zscore_normalize():
    """No-node encode branch must apply z-score normalization before Milvus search."""
    import src.study.probe as probe_mod

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    mock_client = MagicMock()
    mock_client.search.return_value = [[]]

    with pytest.MonkeyPatch().context() as mp:
        calls: list[str] = []
        orig = probe_mod._load_zscore_manifest

        def _track(collection: str):
            calls.append(collection)
            return orig(collection)

        mp.setattr(probe_mod, "_load_zscore_manifest", _track)
        probe_mod._knn_neighbors(
            mock_conn,
            mock_client,
            "street_class=preflop|pot_type=srp|hero_pos_rel=IP|n_players_active=2",
            spot=_SRP_SPOT,
            k=5,
        )

    assert len(calls) > 0, "_load_zscore_manifest not called on no-node path"
