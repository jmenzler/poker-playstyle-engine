"""Phase 7 / BLOCKER 3 / INTG-03 — encode_spot_to_cluster_key.

Closes CLI-02 Mode B + ERR-02 (unsupported-street raise).
Cf. .planning/v1.0-MILESTONE-AUDIT.md BLOCKER 3 + 07-CONTEXT.md D-07-5.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src._errors import CanonicalizeError


def test_encode_spot_to_cluster_key_returns_canonical_format():
    """CLI-02 Mode B: returns a sorted 'k=v|k=v' string per _cluster_key_from_filter.

    Mocks Canonicalizer.default().encode(gs) to avoid the equity_table.parquet
    requirement on dev machines (per plan note: PC-gated env; unit may mock).
    """
    from tools.build_embedding import encode_spot_to_cluster_key

    fake_enc = SimpleNamespace(
        hard_filter={
            "hero_pos_rel": "IP",
            "n_players_active": 2,
            "pot_type": "srp",
            "street_class": "postflop",
        }
    )
    spot = {
        "street": "flop",
        "hero_hole": ["As", "Ks"],
        "board": ["7h", "2d", "Jc"],
        "hero_pos_rel": "ip",
    }
    with patch("src.canonicalizer.Canonicalizer.default") as default_mock:
        default_mock.return_value.encode.return_value = fake_enc
        cluster_key = encode_spot_to_cluster_key(spot)

    assert isinstance(cluster_key, str)
    assert "|" in cluster_key
    parts = cluster_key.split("|")
    assert parts == sorted(parts), "_cluster_key_from_filter guarantees sorted output"
    assert cluster_key == ("hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop")


def test_unknown_street_raises_canonicalize_error():
    """ERR-02: unsupported street raises CanonicalizeError BEFORE GameState construction."""
    from tools.build_embedding import encode_spot_to_cluster_key

    with pytest.raises(CanonicalizeError, match="unsupported street"):
        encode_spot_to_cluster_key(
            {
                "street": "fifth_street",
                "hero_hole": ["As", "Ks"],
                "board": [],
            }
        )


@pytest.mark.integration
def test_post_api_probe_spot_e2e():
    """POST /api/probe/spot returns the probe response without ImportError (CLI-02 Mode B).

    PC-gated: requires equity_table.parquet for Canonicalizer.default() to
    initialize, plus a reachable TimescaleDB + Milvus for probe_by_cluster_key.
    Skips when TSDB_PASSWORD or MILVUS_HOST is absent (dev-Mac shape).
    """
    import os

    if "TSDB_PASSWORD" not in os.environ or "MILVUS_HOST" not in os.environ:
        pytest.skip("PC-gated: requires TSDB_PASSWORD + MILVUS_HOST env")

    from fastapi.testclient import TestClient

    from src.api.main import app  # locked 2026-05-20: src/api/main.py:59 — app = FastAPI(...)

    client = TestClient(app)
    resp = client.post(
        "/api/probe/spot",
        json={
            "spot": {
                "street": "flop",
                "hero_pos_rel": "ip",
                "board": ["7h", "2d", "Jc"],
                "hero_hole": ["As", "Ks"],
            }
        },
    )
    assert resp.status_code == 200, f"POST /api/probe/spot failed: {resp.text}"
    body = resp.json()
    # Response shape per src/study/probe.probe_by_cluster_key: 3 sections
    # (engine_response, knn_neighbors, solver_truth). The defining signal that
    # BLOCKER 3 is closed is that the request reached the encoder without
    # ImportError (status 200, structured body).
    assert "engine_response" in body, f"response missing engine_response: {body}"
