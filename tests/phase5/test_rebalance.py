"""tests/phase5/test_rebalance.py — Unit tests for rebalance.py (Plan 06).

Requirements covered:
- kNN-rebalance candidate distribution generation (RSCH-05 implementation):
  fetch representative observations, retrieve k=10 Milvus neighbors per rep,
  blend via blend_distributions(), average across representatives
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Helper: build Milvus hit dicts for mock search results
# ---------------------------------------------------------------------------

_POSTFLOP_CK = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=postflop"
_PREFLOP_CK = "hero_pos_rel=BTN|n_players_active=2|pot_type=srp|street_class=preflop"


def _make_hits(action: str, n: int = 10, confidence: float = 1.0, gto_score: float = 1.0) -> list[dict]:
    """Return n Milvus hit dicts all pointing to the same action."""
    return [
        {
            "distance": 0.1,
            "entity": {
                "decision_id": f"id_{i}",
                "hero_action_type": action,
                "confidence": confidence,
                "gto_score": gto_score,
            },
        }
        for i in range(n)
    ]


def _make_rebalance_tsdb(embedding_rows: list[tuple]) -> MagicMock:
    """Return a MagicMock TSDB connection for rebalance tests.

    embedding_rows: list of (embedding,) tuples where embedding is list[float].
    """
    conn = MagicMock()
    mock_cur = MagicMock()
    # rebalance SELECTs (embedding, decision_id); append a synthetic DP id per row.
    mock_cur.fetchall.return_value = [(r[0], f"dp_{i}") for i, r in enumerate(embedding_rows)]
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_dict_summing_to_one() -> None:
    """Returned action_dist sums to 1.0 ± 1e-9."""
    emb = [0.1] * 80
    embedding_rows = [(emb,) for _ in range(10)]
    tsdb = _make_rebalance_tsdb(embedding_rows)

    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [_make_hits("bet_50")]

    from src.autoloop.rebalance import generate_candidate_action_dist

    action_dist, _, _ = generate_candidate_action_dist(
        _POSTFLOP_CK,
        _tsdb_conn=tsdb,
        _milvus=mock_milvus,
    )
    assert abs(sum(action_dist.values()) - 1.0) < 1e-9


def test_returns_representative_embedding_as_second_tuple_element() -> None:
    """Second element of tuple is list(first_row_embedding)."""
    emb1 = [0.1] * 80
    emb2 = [0.2] * 80
    embedding_rows = [(emb1,), (emb2,)]
    tsdb = _make_rebalance_tsdb(embedding_rows)

    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [_make_hits("bet_50")]

    from src.autoloop.rebalance import generate_candidate_action_dist

    _, returned_emb, _ = generate_candidate_action_dist(
        _POSTFLOP_CK,
        _tsdb_conn=tsdb,
        _milvus=mock_milvus,
    )
    assert returned_emb == list(emb1)


def test_returns_first_rep_decision_id_as_third_tuple_element() -> None:
    """Third element is the first (latest-by-ts) rep's decision_id — the Milvus PK."""
    emb = [0.1] * 80
    tsdb = _make_rebalance_tsdb([(emb,), (emb,)])  # helper tags rows dp_0, dp_1
    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [_make_hits("bet_50")]

    from src.autoloop.rebalance import generate_candidate_action_dist

    _, _, decision_id = generate_candidate_action_dist(
        _POSTFLOP_CK,
        _tsdb_conn=tsdb,
        _milvus=mock_milvus,
    )
    assert decision_id == "dp_0"


def test_raises_when_rep_has_null_decision_id() -> None:
    """HM-ingest rep (decision_id NULL) cannot key a replayable patch → NoStrategyError."""
    from src._errors import NoStrategyError

    conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [([0.1] * 80, None)]  # embedding present, decision_id NULL
    conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    from src.autoloop.rebalance import generate_candidate_action_dist

    with pytest.raises(NoStrategyError, match="decision_id"):
        generate_candidate_action_dist(_POSTFLOP_CK, _tsdb_conn=conn, _milvus=MagicMock())


def test_raises_no_strategy_when_zero_observations() -> None:
    """Raises NoStrategyError when cluster has zero observations in TSDB."""
    tsdb = _make_rebalance_tsdb([])

    from src._errors import NoStrategyError
    from src.autoloop.rebalance import generate_candidate_action_dist

    with pytest.raises(NoStrategyError, match="zero observations"):
        generate_candidate_action_dist(
            _POSTFLOP_CK,
            _tsdb_conn=tsdb,
            _milvus=MagicMock(),
        )


def test_raises_no_strategy_when_milvus_empty_for_all_reps() -> None:
    """Raises NoStrategyError when Milvus returns empty for all representatives."""
    emb = [0.1] * 80
    embedding_rows = [(emb,), (emb,), (emb,)]
    tsdb = _make_rebalance_tsdb(embedding_rows)

    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [[]]  # empty for all reps

    from src._errors import NoStrategyError
    from src.autoloop.rebalance import generate_candidate_action_dist

    with pytest.raises(NoStrategyError, match="empty results for all"):
        generate_candidate_action_dist(
            _POSTFLOP_CK,
            _tsdb_conn=tsdb,
            _milvus=mock_milvus,
        )


def test_uses_preflop_collection_for_preflop_cluster() -> None:
    """Milvus is searched against 'preflop_decisions' for preflop cluster_key."""
    emb = [0.1] * 32
    embedding_rows = [(emb,)]
    tsdb = _make_rebalance_tsdb(embedding_rows)

    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [_make_hits("bet_50")]

    from src.autoloop.rebalance import generate_candidate_action_dist

    generate_candidate_action_dist(
        _PREFLOP_CK,
        _tsdb_conn=tsdb,
        _milvus=mock_milvus,
    )

    call_kwargs = mock_milvus.search.call_args.kwargs
    assert call_kwargs["collection_name"] == "preflop_decisions"


def test_uses_postflop_collection_for_postflop_cluster() -> None:
    """Milvus is searched against 'postflop_decisions' for postflop cluster_key."""
    emb = [0.1] * 80
    embedding_rows = [(emb,)]
    tsdb = _make_rebalance_tsdb(embedding_rows)

    mock_milvus = MagicMock()
    mock_milvus.search.return_value = [_make_hits("bet_50")]

    from src.autoloop.rebalance import generate_candidate_action_dist

    generate_candidate_action_dist(
        _POSTFLOP_CK,
        _tsdb_conn=tsdb,
        _milvus=mock_milvus,
    )

    call_kwargs = mock_milvus.search.call_args.kwargs
    assert call_kwargs["collection_name"] == "postflop_decisions"


def test_averages_across_representatives() -> None:
    """3 reps: rep1=all-fold, rep2=all-call, rep3=all-call => fold~1/3, call~2/3."""
    emb = [0.1] * 80
    embedding_rows = [(emb,), (emb,), (emb,)]
    tsdb = _make_rebalance_tsdb(embedding_rows)

    mock_milvus = MagicMock()
    mock_milvus.search.side_effect = [
        [_make_hits("fold")],  # rep 1: all fold
        [_make_hits("call")],  # rep 2: all call
        [_make_hits("call")],  # rep 3: all call
    ]

    from src.autoloop.rebalance import generate_candidate_action_dist

    action_dist, _, _ = generate_candidate_action_dist(
        _POSTFLOP_CK,
        _tsdb_conn=tsdb,
        _milvus=mock_milvus,
        k_representatives=3,
    )

    # fold should be ~1/3 and call ~2/3 of the distribution
    assert abs(action_dist["fold"] - 1 / 3) < 0.05
    assert abs(action_dist["call"] - 2 / 3) < 0.05
    assert abs(sum(action_dist.values()) - 1.0) < 1e-9


def test_skips_milvus_empty_for_one_rep() -> None:
    """When one rep returns empty Milvus results, averaging happens over remaining reps."""
    emb = [0.1] * 80
    embedding_rows = [(emb,), (emb,), (emb,)]
    tsdb = _make_rebalance_tsdb(embedding_rows)

    mock_milvus = MagicMock()
    mock_milvus.search.side_effect = [
        [_make_hits("bet_50")],  # rep 1: valid
        [[]],  # rep 2: empty (skip)
        [_make_hits("bet_50")],  # rep 3: valid
    ]

    from src.autoloop.rebalance import generate_candidate_action_dist

    action_dist, _, _ = generate_candidate_action_dist(
        _POSTFLOP_CK,
        _tsdb_conn=tsdb,
        _milvus=mock_milvus,
        k_representatives=3,
    )
    assert abs(sum(action_dist.values()) - 1.0) < 1e-9
    # bet_50 should dominate since both valid reps returned it
    assert action_dist["bet_50"] > 0.9


def test_handles_blend_distributions_no_strategy_error() -> None:
    """blend_distributions raises NoStrategyError for zero-weight hits (one rep).

    The rep whose blend fails is skipped; remaining reps are used for averaging.
    """
    emb = [0.1] * 80
    embedding_rows = [(emb,), (emb,), (emb,)]
    tsdb = _make_rebalance_tsdb(embedding_rows)

    # Rep 1: zero confidence + gto_score -> blend_distributions raises NoStrategyError
    zero_weight_hits = [
        {
            "distance": 0.1,
            "entity": {
                "decision_id": "id_0",
                "hero_action_type": "bet_50",
                "confidence": 0.0,
                "gto_score": 0.0,
            },
        }
    ]

    mock_milvus = MagicMock()
    mock_milvus.search.side_effect = [
        [zero_weight_hits],  # rep 1: zero weights -> NoStrategyError in blend
        [_make_hits("call")],  # rep 2: valid
        [_make_hits("call")],  # rep 3: valid
    ]

    from src.autoloop.rebalance import generate_candidate_action_dist

    action_dist, _, _ = generate_candidate_action_dist(
        _POSTFLOP_CK,
        _tsdb_conn=tsdb,
        _milvus=mock_milvus,
        k_representatives=3,
    )
    assert abs(sum(action_dist.values()) - 1.0) < 1e-9
    # 'call' should dominate since both valid reps returned it
    assert action_dist["call"] > 0.9
