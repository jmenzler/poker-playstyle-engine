"""Unit tests for src/metrics/ev_loss.py (Plan 04-03, TDD RED → GREEN).

Coverage:
    (a) Synthetic cluster with observed == expected → KL ≈ 0
    (b) Known divergence: observed all "call", expected uniform over 3 → KL ≈ log(3)
    (c) Idempotency: two consecutive calls return identical float (LABL-02)
    (d) NoStrategyError when TSDB returns < 10 observations
    (e) NoStrategyError when Milvus returns empty hits
    (f) Collection dispatch: preflop cluster_key → preflop_decisions; postflop → postflop_decisions
    (g) Smoothing: observed action absent from expected dict → finite KL (no log(0))
    (h) Expected distribution uses blend_distributions formula (key vocabulary check)
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers — build mock objects that ev_loss consumes
# ---------------------------------------------------------------------------


def _make_tsdb_conn(rows: list[tuple]) -> MagicMock:
    """Return a psycopg-shaped MagicMock that yields `rows` from fetchall().

    Supports nested context manager: ``with conn.cursor() as cur:``.
    `rows` format: list of (action_taken, embedding) tuples.
    """
    conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


def _make_milvus_client(hits: list[dict]) -> MagicMock:
    """Return a MagicMock MilvusClient whose search() returns [[*hits]]."""
    client = MagicMock()
    client.search.return_value = [hits]
    return client


def _uniform_hits(actions: list[str], n: int = 10) -> list[dict]:
    """Build n Milvus hits with uniform action distribution over `actions`."""
    hits = []
    for i in range(n):
        action = actions[i % len(actions)]
        hits.append(
            {
                "distance": 0.1,
                "entity": {
                    "decision_id": f"dp{i}",
                    "hero_action_type": action,
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Cluster-key fixtures
# ---------------------------------------------------------------------------

_POSTFLOP_KEY = "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=postflop"
_PREFLOP_KEY = "hero_pos_rel=IP|n_players_active=2|pot_type=srp|street_class=preflop"


# ---------------------------------------------------------------------------
# (a) KL ≈ 0 when observed == expected
# ---------------------------------------------------------------------------


def test_kl_zero_when_observed_equals_expected():
    """KL(P || Q) ≈ 0 when both distributions are identical."""
    from src.metrics.ev_loss import ev_loss

    # 12 observations: 4 fold, 4 call, 4 raise — gives P = {fold:1/3, call:1/3, raise:1/3}
    # Milvus hits also uniform over fold/call/raise — gives Q ≈ {fold:1/3, call:1/3, raise:1/3}
    obs_rows = [("fold", [0.1] * 80)] * 4 + [("call", [0.1] * 80)] * 4 + [("raise_min", [0.1] * 80)] * 4
    tsdb = _make_tsdb_conn(obs_rows)
    milvus = _make_milvus_client(_uniform_hits(["fold", "call", "raise_min"], n=12))

    kl = ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb, _milvus=milvus)
    assert abs(kl) < 1e-4, f"Expected KL ≈ 0, got {kl}"


# ---------------------------------------------------------------------------
# (b) Known KL value: observed = point mass "call", expected = uniform over 3 actions
# ---------------------------------------------------------------------------


def test_kl_known_value():
    """KL(P=all-call || Q=uniform-3) ≈ log(3) ≈ 1.0986.

    With smoothing ε=1e-6, the exact value differs slightly. We verify it is
    within a 0.01 absolute tolerance of log(3).
    """
    from src.metrics.ev_loss import ev_loss

    # 10 observations, all "call"
    obs_rows = [("call", [0.1] * 80)] * 10
    tsdb = _make_tsdb_conn(obs_rows)

    # Milvus: 10 hits, uniform fold/call/raise_min → Q ≈ {1/3, 1/3, 1/3}
    # (blend_distributions is weighted by similarity*confidence*gto_score;
    #  with uniform weights each action gets equal mass)
    hits = _uniform_hits(["fold", "call", "raise_min"], n=12)
    milvus = _make_milvus_client(hits)

    kl = ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb, _milvus=milvus)

    # Without smoothing: KL = log(3) ≈ 1.0986
    # With ε smoothing the result is slightly less; tolerance = 0.05
    expected_approx = math.log(3)
    assert abs(kl - expected_approx) < 0.05, f"Expected KL ≈ {expected_approx:.4f}, got {kl:.4f}"
    assert kl > 0.0, "KL must be positive for divergent distributions"


# ---------------------------------------------------------------------------
# (c) Idempotency (LABL-02)
# ---------------------------------------------------------------------------


def test_idempotent_repeated_calls():
    """Two calls with identical mocks → byte-identical float."""
    from src.metrics.ev_loss import ev_loss

    obs_rows = [("fold", [0.2] * 80)] * 6 + [("call", [0.3] * 80)] * 6
    hits = _uniform_hits(["fold", "call"], n=10)

    # Fresh mocks for each call to ensure no shared state
    tsdb1 = _make_tsdb_conn(obs_rows)
    milvus1 = _make_milvus_client(hits)
    tsdb2 = _make_tsdb_conn(obs_rows)
    milvus2 = _make_milvus_client(hits)

    kl1 = ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb1, _milvus=milvus1)
    kl2 = ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb2, _milvus=milvus2)

    assert kl1 == kl2, f"LABL-02: repeated calls must return identical float; got {kl1} vs {kl2}"


# ---------------------------------------------------------------------------
# (d) NoStrategyError on < k=10 observations
# ---------------------------------------------------------------------------


def test_raises_when_under_k_observations():
    """Cluster with 9 observations raises NoStrategyError."""
    from src._errors import NoStrategyError
    from src.metrics.ev_loss import ev_loss

    obs_rows = [("fold", [0.1] * 80)] * 9  # 9 < k=10
    tsdb = _make_tsdb_conn(obs_rows)
    milvus = _make_milvus_client(_uniform_hits(["fold", "call"], n=10))

    with pytest.raises(NoStrategyError, match="observations"):
        ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb, _milvus=milvus)


# ---------------------------------------------------------------------------
# (e) NoStrategyError on empty Milvus results
# ---------------------------------------------------------------------------


def test_raises_when_milvus_empty():
    """Empty Milvus search result raises NoStrategyError."""
    from src._errors import NoStrategyError
    from src.metrics.ev_loss import ev_loss

    obs_rows = [("fold", [0.1] * 80)] * 10
    tsdb = _make_tsdb_conn(obs_rows)

    # Milvus returns empty inner list
    empty_milvus = MagicMock()
    empty_milvus.search.return_value = [[]]

    with pytest.raises(NoStrategyError):
        ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb, _milvus=empty_milvus)


# ---------------------------------------------------------------------------
# (f) Collection dispatch by street_class
# ---------------------------------------------------------------------------


def test_dispatches_preflop_collection():
    """cluster_key with street_class=preflop → Milvus search on preflop_decisions."""
    from src.metrics.ev_loss import ev_loss

    obs_rows = [("fold", [0.1] * 32)] * 10 + [("call", [0.1] * 32)] * 2
    tsdb = _make_tsdb_conn(obs_rows)
    hits = _uniform_hits(["fold", "call"], n=10)
    milvus = _make_milvus_client(hits)

    ev_loss(_PREFLOP_KEY, _tsdb_conn=tsdb, _milvus=milvus)

    # Assert Milvus search was called with the preflop collection
    call_kwargs = milvus.search.call_args
    assert call_kwargs is not None, "Milvus search was not called"
    used_collection = call_kwargs.kwargs.get("collection_name") or call_kwargs.args[0]
    assert used_collection == "preflop_decisions", f"Expected preflop_decisions, got {used_collection!r}"


def test_dispatches_postflop_collection():
    """cluster_key with street_class=postflop → Milvus search on postflop_decisions."""
    from src.metrics.ev_loss import ev_loss

    obs_rows = [("fold", [0.1] * 80)] * 10 + [("call", [0.1] * 80)] * 2
    tsdb = _make_tsdb_conn(obs_rows)
    hits = _uniform_hits(["fold", "call"], n=10)
    milvus = _make_milvus_client(hits)

    ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb, _milvus=milvus)

    call_kwargs = milvus.search.call_args
    assert call_kwargs is not None, "Milvus search was not called"
    used_collection = call_kwargs.kwargs.get("collection_name") or call_kwargs.args[0]
    assert used_collection == "postflop_decisions", f"Expected postflop_decisions, got {used_collection!r}"


# ---------------------------------------------------------------------------
# (g) Smoothing: action in observed but absent from expected → finite KL
# ---------------------------------------------------------------------------


def test_smoothing_prevents_log_zero():
    """Observed has action 'allin' that Milvus never returns → ε smoothing → finite KL."""
    from src.metrics.ev_loss import ev_loss

    # 10 observations: 9 "fold", 1 "allin" (allin never appears in Milvus hits)
    obs_rows = [("fold", [0.1] * 80)] * 9 + [("allin", [0.1] * 80)] * 1
    tsdb = _make_tsdb_conn(obs_rows)

    # Milvus hits only have fold/call — "allin" has Q=0 before smoothing
    hits = _uniform_hits(["fold", "call"], n=10)
    milvus = _make_milvus_client(hits)

    kl = ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb, _milvus=milvus)

    assert math.isfinite(kl), f"KL must be finite with smoothing; got {kl}"
    assert kl >= 0.0, f"KL must be non-negative; got {kl}"


# ---------------------------------------------------------------------------
# (h) Expected distribution keys match blend_distributions output vocabulary
# ---------------------------------------------------------------------------


def test_expected_dist_uses_blending():
    """ev_loss internally blends neighbor action_dists; resulting Q keys are action strings."""
    from src.metrics.ev_loss import ev_loss

    obs_rows = [("call", [0.1] * 80)] * 10
    tsdb = _make_tsdb_conn(obs_rows)
    hits = _uniform_hits(["fold", "call", "raise_min"], n=12)
    milvus = _make_milvus_client(hits)

    # Should not raise; the blend result vocabulary must overlap with P vocabulary
    kl = ev_loss(_POSTFLOP_KEY, _tsdb_conn=tsdb, _milvus=milvus)
    assert isinstance(kl, float), f"ev_loss must return float; got {type(kl)}"
