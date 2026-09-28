"""Shared fixtures for Phase 3 (Play Loop) tests.

Provides tsdb_dsn / milvus_uri / milvus_token for integration tests (PC-gated),
plus Phase-3-specific fixtures: mock Milvus client, GameState samples for all
four streets, and a tmp directory with valid z-score manifest JSON files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.protocols.game_state import GameState  # noqa: I001


# --- Integration fixtures (PC-gated) -----------------------------------------


@pytest.fixture(scope="session")
def tsdb_dsn() -> str:
    """Build TimescaleDB DSN from env vars (prod DB, not test DB)."""
    host = os.environ.get("TSDB_HOST", "127.0.0.1")
    port = os.environ.get("TSDB_PORT", "55432")
    db = os.environ.get("TSDB_DB", "poker_engine")
    user = os.environ.get("TSDB_USER", "poker")
    password = os.environ["TSDB_PASSWORD"]  # fail loud if missing
    return f"host={host} port={port} dbname={db} user={user} password={password} connect_timeout=5"


@pytest.fixture(scope="session")
def milvus_uri() -> str:
    host = os.environ.get("MILVUS_HOST", "127.0.0.1")
    port = os.environ.get("MILVUS_PORT", "51530")
    return f"http://{host}:{port}"


@pytest.fixture(scope="session")
def milvus_token() -> str | None:
    """Milvus auth in ``root:password`` form (prepended if .env stores raw password)."""
    raw = os.environ.get("MILVUS_TOKEN")
    if raw is None:
        return None
    return raw if ":" in raw else f"root:{raw}"


# --- Mock Milvus clients ------------------------------------------------------


@pytest.fixture
def mock_milvus_client():
    """MagicMock search() returning pymilvus COSINE similarity (1.0=identical) in 'distance'."""
    client = MagicMock()
    client.search.return_value = [
        [
            {
                "distance": 0.9,
                "entity": {
                    "decision_id": "dp1",
                    "hero_action_type": "call",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
            {
                "distance": 0.8,
                "entity": {
                    "decision_id": "dp2",
                    "hero_action_type": "fold",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
            {
                "distance": 0.6,
                "entity": {
                    "decision_id": "dp3",
                    "hero_action_type": "call",
                    "confidence": 1.0,
                    "gto_score": 1.0,
                },
            },
        ]
    ]
    return client


@pytest.fixture
def mock_milvus_client_empty():
    """MagicMock that returns zero results — triggers NoStrategyError."""
    client = MagicMock()
    client.search.return_value = [[]]
    return client


# --- GameState fixtures (all kw_only, all required fields) --------------------


@pytest.fixture
def preflop_game_state() -> GameState:
    return GameState(
        street="preflop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kd"),
        board_cards=(),
        pot_size_bb=3.0,
        effective_stack_bb=100.0,
        hero_facing_bet_bb=2.5,
        hero_bet_size_bb=0.0,
        action_sequence=("UTG:fold", "MP:fold", "CO:fold"),
        opponents_remaining=2,
        prior_street_aggressor=None,
    )


@pytest.fixture
def flop_game_state() -> GameState:
    return GameState(
        street="flop",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kh"),
        board_cards=("Qh", "Jh", "2s"),
        pot_size_bb=6.5,
        effective_stack_bb=97.5,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("BB:check",),
        opponents_remaining=1,
        prior_street_aggressor="BTN",
    )


@pytest.fixture
def turn_game_state() -> GameState:
    return GameState(
        street="turn",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kh"),
        board_cards=("Qh", "Jh", "2s", "Td"),
        pot_size_bb=13.0,
        effective_stack_bb=91.0,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("BB:check",),
        opponents_remaining=1,
        prior_street_aggressor="BTN",
    )


@pytest.fixture
def river_game_state() -> GameState:
    return GameState(
        street="river",
        hero_position="BTN",
        hero_hole_cards=("Ah", "Kh"),
        board_cards=("Qh", "Jh", "2s", "Td", "5c"),
        pot_size_bb=26.0,
        effective_stack_bb=78.0,
        hero_facing_bet_bb=0.0,
        hero_bet_size_bb=0.0,
        action_sequence=("BB:check",),
        opponents_remaining=1,
        prior_street_aggressor="BTN",
    )


# --- Z-score manifest tmp dir -------------------------------------------------


@pytest.fixture
def tmp_manifest_dir(tmp_path: Path) -> Path:
    """Two valid z-score manifest JSON files matching tools/zscore_fit.py output.

    Both manifests use mean=0, std=1 per dim so normalize() is identity.
    Useful for unit-testing DecisionEngine without requiring real fit data.
    """
    out = tmp_path / "manifests"
    out.mkdir()
    for name, dim in (("preflop", 34), ("postflop", 80)):
        manifest = {
            "feature_spec_version": 3,
            "fit_population_n": 100,
            "fit_date": "2026-05-18",
            "collection": f"{name}_decisions",
            "dim_stats": [{"dim": i, "name": f"d{i}", "mean": 0.0, "std": 1.0} for i in range(dim)],
        }
        (out / f"zscore_{name}.json").write_text(json.dumps(manifest))
    return out


@pytest.fixture
def synthetic_kNN_results():
    """Synthetic Milvus search() return for direct blending tests.

    Distances chosen so weights are [0.9, 0.8, 0.6] -> normalized [0.39, 0.35, 0.26].
    """
    return [
        {
            "distance": 0.1,
            "entity": {
                "decision_id": "n1",
                "hero_action_type": "call",
                "confidence": 1.0,
                "gto_score": 1.0,
            },
        },
        {
            "distance": 0.2,
            "entity": {
                "decision_id": "n2",
                "hero_action_type": "raise_3x",
                "confidence": 1.0,
                "gto_score": 1.0,
            },
        },
        {
            "distance": 0.4,
            "entity": {
                "decision_id": "n3",
                "hero_action_type": "fold",
                "confidence": 1.0,
                "gto_score": 1.0,
            },
        },
    ]
