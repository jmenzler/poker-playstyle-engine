"""Shared fixtures for Phase 2 (Bootstrap kNN Knowledge Base) tests.

Inherits milvus_uri / milvus_token from tests/integration/conftest.py via the
pytest conftest hierarchy. Adds Phase-2-specific fixtures: sample DPs,
synthetic embedding arrays, tmp chunks directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Repo-root path injection (matches tools/ scripts pattern)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ─── Phase 1 filter-test residue IDs (RESEARCH.md §Runtime State Inventory) ───
PHASE1_FILTER_TEST_IDS: list[str] = [
    "filter-test-hu-preflop_decisions",
    "filter-test-mw-preflop_decisions",
    "filter-test-hu-postflop_decisions",
    "filter-test-mw-postflop_decisions",
]


@pytest.fixture(scope="session")
def milvus_uri() -> str:
    """Construct Milvus URI from env. Mirrors tests/integration/conftest.py."""
    host = os.environ.get("MILVUS_HOST", "127.0.0.1")
    port = os.environ.get("MILVUS_PORT", "19530")
    return f"http://{host}:{port}"


@pytest.fixture(scope="session")
def milvus_token() -> str | None:
    """Return Milvus auth token in ``root:password`` format.

    Reads from MILVUS_TOKEN env var; prepends 'root:' if no ':' present.
    Returns None if env var is unset (unauthenticated Milvus).
    """
    raw = os.environ.get("MILVUS_TOKEN")
    if raw is None:
        return None
    return raw if ":" in raw else f"root:{raw}"


@pytest.fixture
def sample_preflop_dp() -> dict:
    """Minimal valid preflop DP record matching tools/extract_decisions.py output.

    Field names match the JSONL emitted by extract_decisions.py (schema_version=2).
    Used by extractor + ledger + upsert tests.
    """
    return {
        "id": "hand123_dp0",
        "hand_id": "hand123",
        "decision_idx": 0,
        "_schema_version": 2,
        "street": "preflop",
        "street_class": "preflop",
        "hero_pos": "BTN",
        "hero_pos_rel": "IP",
        "n_players_at_street": 2,
        "n_players_active": 2,
        "pot_type": "srp",
        "pot_bb": 3.0,
        "pot_cents": 30,
        "bb_cents": 10,
        "effective_stack_cents": 1000,
        "hero_stack_cents": 1000,
        "hero_stack_bb": 100.0,
        "spr": 33.3,
        "hero_hole": ["Ah", "Ks"],
        "hero_hole_class": "AKo",
        "hole_canonical": "AhKs",
        "board": None,
        "board_canonical": "",
        "preflop_aggressor": "BTN",
        "preflop_action_seq": [
            {"pos": "BTN", "action": "raise", "size_cents": 25},
            {"pos": "BB", "action": "call", "size_cents": 25},
        ],
        "action_so_far_street": [],
        "facing": "cold",
        "facing_pos": None,
        "facing_size_pot_frac": None,
        "facing_size_cents": None,
        "hero_action_type": "raise",
        "hero_action_size_pot_frac": 2.5,
        "hero_action_to_bb": 2.5,
        "scenario_key": "srp|IP|n2|BB:caller:BTN",
    }


@pytest.fixture
def sample_postflop_dp() -> dict:
    """Minimal valid postflop DP record matching tools/extract_decisions.py output.

    Field names match the JSONL emitted by extract_decisions.py (schema_version=2).
    """
    return {
        "id": "hand456_dp2",
        "hand_id": "hand456",
        "decision_idx": 2,
        "_schema_version": 2,
        "street": "flop",
        "street_class": "postflop",
        "hero_pos": "BTN",
        "hero_pos_rel": "IP",
        "n_players_at_street": 2,
        "n_players_active": 2,
        "pot_type": "srp",
        "pot_bb": 6.5,
        "pot_cents": 65,
        "bb_cents": 10,
        "effective_stack_cents": 975,
        "hero_stack_cents": 975,
        "hero_stack_bb": 97.5,
        "spr": 15.0,
        "hero_hole": ["Ah", "Kh"],
        "hero_hole_class": "AKs",
        "hole_canonical": "AhKh",
        "board": ["Qh", "Jh", "2s"],
        "board_canonical": "QhJh2s",
        "preflop_aggressor": "BTN",
        "preflop_action_seq": [
            {"pos": "BTN", "action": "raise", "size_cents": 25},
            {"pos": "BB", "action": "call", "size_cents": 25},
        ],
        "action_so_far_street": [
            {"pos": "BB", "action": "check"},
        ],
        "facing": "check_to",
        "facing_pos": "BB",
        "facing_size_pot_frac": None,
        "facing_size_cents": None,
        "hero_action_type": "bet",
        "hero_action_size_pot_frac": 0.5,
        "hero_action_to_bb": 3.25,
        "scenario_key": "srp|IP|n2|BB:caller:BTN",
    }


@pytest.fixture
def synthetic_preflop_embedding() -> np.ndarray:
    """A deterministic 34-dim vector in [0,1] — pre-z-score, pre-weights."""
    rng = np.random.default_rng(seed=42)
    return rng.random(34).astype(np.float32)


@pytest.fixture
def synthetic_postflop_embedding() -> np.ndarray:
    """A deterministic 80-dim vector in [0,1] — pre-z-score, pre-weights."""
    rng = np.random.default_rng(seed=42)
    return rng.random(80).astype(np.float32)


@pytest.fixture
def tmp_chunks_dir(tmp_path: Path) -> Path:
    """Empty tmp directory for hm_embeddings_chunk_*.parquet output."""
    d = tmp_path / "chunks"
    d.mkdir()
    return d


@pytest.fixture
def chunked_parquet_dir_postflop(tmp_chunks_dir: Path) -> Path:
    """Two tiny chunked parquet files mimicking stage-3 output (postflop, 80d).

    Schema matches _build_schema_postflop() in tools/build_embedding.py exactly.
    Used by zscore_fit + upsert + dataset-iterator tests.
    """
    rng = np.random.default_rng(seed=7)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("hand_id", pa.string()),
            pa.field("street_class", pa.string()),
            pa.field("pot_type", pa.string()),
            pa.field("hero_pos_rel", pa.string()),
            pa.field("hero_pos", pa.string()),
            pa.field("n_players_active", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), 80)),
            pa.field("schema_version", pa.int32()),
        ]
    )
    for chunk_idx in (1, 2):
        n = 50
        embeddings = rng.random((n, 80)).astype(np.float32).tolist()
        table = pa.table(
            {
                "id": [f"c{chunk_idx}_dp{i}" for i in range(n)],
                "hand_id": [f"c{chunk_idx}_h{i}" for i in range(n)],
                "street_class": ["postflop"] * n,
                "pot_type": ["srp"] * n,
                "hero_pos_rel": ["IP"] * n,
                "hero_pos": ["BTN"] * n,
                "n_players_active": [2] * n,
                "embedding": embeddings,
                "schema_version": [2] * n,
            },
            schema=schema,
        )
        pq.write_table(
            table,
            tmp_chunks_dir / f"hm_embeddings_chunk_{chunk_idx:04d}.parquet",
        )
    return tmp_chunks_dir


@pytest.fixture
def expected_postflop_group_weights() -> np.ndarray:
    """80-dim postflop group-weight vector from the single source of truth."""
    from tools.upsert_milvus import build_postflop_weight_vector

    w = build_postflop_weight_vector()
    assert w.shape == (80,)
    return w
