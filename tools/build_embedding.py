"""Build hm_embeddings_chunk_NNNN_*.parquet from hm_decisions.jsonl + equity_table.parquet.

CLI dispatcher — reads per-DP JSONL, routes to preflop (32-dim) or postflop
(80-dim) extractor, writes chunked Parquet split by street_class. DPs with no
equity lookup result are routed to an equity-miss ledger (not silently zero-vectored).

Module also exports `encode_spot_to_cluster_key` (Phase 7 / BLOCKER 3) —
adapter used by Probe Mode B (POST /api/probe/spot, src/study/probe.py:91).

Usage:
    uv run python3 tools/build_embedding.py \\
        --decisions research/preflop-ranges/outputs/hm_decisions.jsonl \\
        --equity-table tools/equity_table.parquet \\
        --out-dir /tmp/embeddings/ \\
        [--chunk-size 25000] \\
        [--limit 1000] \\
        [--log-file /tmp/build.log]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

# These imports work both as a top-level script and via uv run
sys.path.insert(0, str(Path(__file__).parent.parent))

from src._log import configure_logging, get_logger
from tools.equity_lookup import EquityLookup
from tools.feature_extractors.postflop import extract_postflop
from tools.feature_extractors.preflop import extract_preflop
from tools.joint_canonicalize import joint_canonicalize

if TYPE_CHECKING:
    import numpy as np

    from src.protocols.game_state import GameState

log = get_logger("tools.build_embedding")

# Output Parquet schema constants (FEATURE_SPEC_VERSION=2 per RESEARCH.md §Schema Version Coordination)
FEATURE_SPEC_VERSION = 3
PROGRESS_EVERY = 1_000
BUFFER_SIZE = 1_000  # rows per write batch
DEFAULT_CHUNK_SIZE = 25_000


def _build_schema_preflop() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("hand_id", pa.string()),
            pa.field("street_class", pa.string()),
            pa.field("street", pa.string()),
            pa.field("pot_type", pa.string()),
            pa.field("hero_pos_rel", pa.string()),
            pa.field("hero_pos", pa.string()),
            pa.field("n_players_active", pa.int32()),
            pa.field("spr_x100", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32(), 34)),
            pa.field("hero_action_type", pa.string()),
            pa.field("hero_action_size_pot_frac", pa.float32()),
            pa.field("raise_ratio", pa.float32()),
            pa.field("hero_action_allin", pa.bool_()),
            pa.field("active", pa.bool_()),
            pa.field("confidence", pa.float32()),
            pa.field("gto_score", pa.float32()),
            pa.field("feature_spec_version", pa.int32()),
        ]
    )


def _size_fields(dp: dict) -> dict:
    """Raw postflop-snap inputs: bet pot-fraction, raise to/facing ratio, allin flag.
    raise_ratio = -1 when not a raise / no facing bet (snap then defaults)."""
    facing = dp.get("facing_size_bb")
    to_bb = dp.get("hero_action_to_bb")
    has_facing = bool(facing) and float(facing) > 0 and to_bb is not None
    raise_ratio = float(to_bb) / float(facing) if has_facing else -1.0
    return {
        "hero_action_size_pot_frac": float(dp.get("hero_action_size_pot_frac") or 0.0),
        "raise_ratio": raise_ratio,
        "hero_action_allin": bool(dp.get("hero_action_allin", False)),
    }


def _build_schema_postflop() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("hand_id", pa.string()),
            pa.field("street_class", pa.string()),
            pa.field("street", pa.string()),
            pa.field("pot_type", pa.string()),
            pa.field("hero_pos_rel", pa.string()),
            pa.field("hero_pos", pa.string()),
            pa.field("n_players_active", pa.int32()),
            pa.field("spr_x100", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32(), 80)),
            pa.field("hero_action_type", pa.string()),
            pa.field("hero_action_size_pot_frac", pa.float32()),
            pa.field("raise_ratio", pa.float32()),
            pa.field("hero_action_allin", pa.bool_()),
            pa.field("active", pa.bool_()),
            pa.field("confidence", pa.float32()),
            pa.field("gto_score", pa.float32()),
            pa.field("feature_spec_version", pa.int32()),
        ]
    )


def _build_schema_equity_miss() -> pa.Schema:
    return pa.schema(
        [
            pa.field("dp_id", pa.string()),
            pa.field("hole_class", pa.string()),
            pa.field("board_canonical", pa.string()),
            pa.field("street", pa.string()),
            pa.field("pot_type", pa.string()),
            pa.field("n_players_active", pa.int32()),
        ]
    )


class _Buffer:
    """Accumulates rows and flushes as Parquet row groups."""

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.schema = schema
        self.writer: pq.ParquetWriter | None = None
        self._rows: list[dict] = []
        self._total = 0

    def append(self, row: dict) -> None:
        self._rows.append(row)
        if len(self._rows) >= BUFFER_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        ids = [r["id"] for r in self._rows]
        hand_ids = [r["hand_id"] for r in self._rows]
        street_classes = [r["street_class"] for r in self._rows]
        streets = [r["street"] for r in self._rows]
        pot_types = [r["pot_type"] for r in self._rows]
        hero_pos_rels = [r["hero_pos_rel"] for r in self._rows]
        hero_poses = [r["hero_pos"] for r in self._rows]
        n_players = [r["n_players_active"] for r in self._rows]
        spr_x100s = [r["spr_x100"] for r in self._rows]
        embeddings = [r["embedding"].tolist() for r in self._rows]
        hero_action_types = [r["hero_action_type"] for r in self._rows]
        size_pot_fracs = [r["hero_action_size_pot_frac"] for r in self._rows]
        raise_ratios = [r["raise_ratio"] for r in self._rows]
        allins = [r["hero_action_allin"] for r in self._rows]
        actives = [r["active"] for r in self._rows]
        confidences = [r["confidence"] for r in self._rows]
        gto_scores = [r["gto_score"] for r in self._rows]
        versions = [FEATURE_SPEC_VERSION] * len(self._rows)

        table = pa.table(
            {
                "id": ids,
                "hand_id": hand_ids,
                "street_class": street_classes,
                "street": streets,
                "pot_type": pot_types,
                "hero_pos_rel": hero_pos_rels,
                "hero_pos": hero_poses,
                "n_players_active": pa.array(n_players, type=pa.int32()),
                "spr_x100": pa.array(spr_x100s, type=pa.int64()),
                "embedding": pa.array(embeddings, type=self.schema.field("embedding").type),
                "hero_action_type": hero_action_types,
                "hero_action_size_pot_frac": pa.array(size_pot_fracs, type=pa.float32()),
                "raise_ratio": pa.array(raise_ratios, type=pa.float32()),
                "hero_action_allin": pa.array(allins, type=pa.bool_()),
                "active": pa.array(actives, type=pa.bool_()),
                "confidence": pa.array(confidences, type=pa.float32()),
                "gto_score": pa.array(gto_scores, type=pa.float32()),
                "feature_spec_version": pa.array(versions, type=pa.int32()),
            },
            schema=self.schema,
        )

        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self._total += len(self._rows)
        self._rows = []

    def close(self) -> int:
        self._flush()
        if self.writer:
            self.writer.close()
        return self._total


class _MissBuffer:
    """Accumulates equity-miss ledger rows and flushes to Parquet."""

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.schema = schema
        self.writer: pq.ParquetWriter | None = None
        self._rows: list[dict] = []
        self._total = 0

    def add(self, row: dict) -> None:
        self._rows.append(row)
        if len(self._rows) >= BUFFER_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        table = pa.table(
            {
                "dp_id": [r["dp_id"] for r in self._rows],
                "hole_class": [r["hole_class"] for r in self._rows],
                "board_canonical": [r["board_canonical"] for r in self._rows],
                "street": [r["street"] for r in self._rows],
                "pot_type": [r["pot_type"] for r in self._rows],
                "n_players_active": pa.array([r["n_players_active"] for r in self._rows], type=pa.int32()),
            },
            schema=self.schema,
        )
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self._total += len(self._rows)
        self._rows = []

    def close(self) -> int:
        self._flush()
        if self.writer:
            self.writer.close()
        return self._total


def _check_equity_miss(dp: dict, equity_lookup: EquityLookup) -> bool:
    """Return True if this DP's equity lookup is a double miss (None).

    Uses joint_canonicalize to mirror what the extractors do internally.
    """
    hole = dp.get("hero_hole")
    board = dp.get("board") or []
    scenario = dp.get("scenario_key", "")

    if not hole or not scenario:
        return False

    h_can, b_can = joint_canonicalize(hole, board)
    result = equity_lookup.get(h_can, b_can, scenario)
    return result is None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build hm_embeddings chunks from hm_decisions.jsonl")
    ap.add_argument("--decisions", required=True, type=Path, help="Path to hm_decisions.jsonl")
    ap.add_argument("--equity-table", required=True, type=Path, help="Path to equity_table.parquet")
    ap.add_argument(
        "--preflop-equity-table",
        type=Path,
        default=None,
        help="Optional path to preflop_equity_table.parquet (merged with --equity-table if present)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for hm_embeddings_chunk_*.parquet + equity_misses.parquet",
    )
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--limit", type=int, default=0, help="Cap number of DPs processed (0=all)")
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If set, structlog also writes JSON lines here (OBS-03 build log)",
    )
    args = ap.parse_args(argv)

    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

    # Attach file sink if --log-file provided
    if args.log_file:
        fh = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.decisions.exists():
        log.error("build_embedding.decisions_not_found", path=str(args.decisions))
        return 1
    if not args.equity_table.exists():
        log.error("build_embedding.equity_table_not_found", path=str(args.equity_table))
        return 1

    # Build equity table path list (postflop required; preflop optional)
    equity_paths: list[Path] = [args.equity_table]
    preflop_table = args.preflop_equity_table
    if preflop_table is None:
        # Default: look for preflop table next to postflop table
        default_preflop = args.equity_table.parent / "preflop_equity_table.parquet"
        if default_preflop.exists():
            preflop_table = default_preflop
    if preflop_table is not None and preflop_table.exists():
        equity_paths.append(preflop_table)
        log.info("build_embedding.preflop_equity_table_found", path=str(preflop_table))
    else:
        log.info("build_embedding.preflop_equity_table_not_found_using_postflop_only")

    # Load equity table(s)
    log.info("build_embedding.loading_equity_table", paths=[str(p) for p in equity_paths])
    equity_lookup = EquityLookup(equity_paths if len(equity_paths) > 1 else equity_paths[0])
    pop_mean = equity_lookup.population_mean_equity()
    log.info("build_embedding.equity_table_loaded", n_rows=equity_lookup.n_total)

    # Villain-range tightness + raw range from observed palettes.
    from tools.villain_range_stats import build_range_lookup, build_tightness_lookup

    _palette_dir = Path(__file__).resolve().parent.parent / "research" / "preflop-ranges" / "outputs"
    tightness_lookup = build_tightness_lookup(_palette_dir) if _palette_dir.exists() else {}
    range_lookup = build_range_lookup(_palette_dir) if _palette_dir.exists() else {}
    log.info("build_embedding.tightness_lookup_built", n_entries=len(tightness_lookup))
    log.info("build_embedding.range_lookup_built", n_entries=len(range_lookup))

    schema_pre = _build_schema_preflop()
    schema_post = _build_schema_postflop()
    schema_miss = _build_schema_equity_miss()

    chunk_idx = 1
    chunk_count = 0  # DPs in current chunk (across both pre + post)

    buf_pf = _Buffer(args.out_dir / f"hm_embeddings_chunk_{chunk_idx:04d}_preflop.parquet", schema_pre)
    buf_po = _Buffer(args.out_dir / f"hm_embeddings_chunk_{chunk_idx:04d}_postflop.parquet", schema_post)
    ledger = _MissBuffer(args.out_dir / "equity_misses.parquet", schema_miss)

    n_preflop = 0
    n_postflop = 0
    n_equity_miss = 0
    n_errors = 0
    n_total = 0
    t0 = time.time()

    log.info("build_embedding.streaming", path=str(args.decisions))

    with args.decisions.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                dp = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("build_embedding.json_decode_error", error=str(exc))
                n_errors += 1
                continue

            # Build scenario_key for equity lookup
            dp = _inject_scenario_key(dp)

            street = dp.get("street", "")
            dp_id = dp.get("id", f"unknown_{n_total}")
            hand_id = dp.get("hand_id", "")
            hero_pos = dp.get("hero_pos", "")

            # Check for equity miss BEFORE extracting — route to ledger, not chunk
            if _check_equity_miss(dp, equity_lookup):
                ledger.add(
                    {
                        "dp_id": dp_id,
                        "hole_class": dp.get("hero_hole_class", dp.get("hole_canonical", "")),
                        "board_canonical": dp.get("board_canonical", ""),
                        "street": street,
                        "pot_type": dp.get("pot_type", ""),
                        "n_players_active": int(dp.get("n_players_active", dp.get("n_players_at_street", 0))),
                    }
                )
                n_equity_miss += 1
                continue

            try:
                spr_raw = dp.get("spr")
                spr_x100 = round(spr_raw * 100) if spr_raw is not None else -1

                if street == "preflop":
                    vec, filt = extract_preflop(dp, equity_lookup, pop_mean, tightness_lookup, range_lookup)
                    buf_pf.append(
                        {
                            "id": dp_id,
                            "hand_id": hand_id,
                            "street_class": "preflop",
                            "street": street,
                            "pot_type": filt["pot_type"],
                            "hero_pos_rel": filt["hero_pos_rel"],
                            "hero_pos": hero_pos,
                            "n_players_active": filt["n_players_active"],
                            "spr_x100": spr_x100,
                            "embedding": vec,
                            "hero_action_type": dp.get("hero_action_type") or "",
                            **_size_fields(dp),
                            "active": bool(dp.get("active", True)),
                            "confidence": float(dp.get("confidence", 1.0)),
                            "gto_score": float(dp.get("gto_score", 1.0)),
                        }
                    )
                    n_preflop += 1
                elif street in ("flop", "turn", "river"):
                    vec, filt = extract_postflop(dp, equity_lookup, pop_mean)
                    buf_po.append(
                        {
                            "id": dp_id,
                            "hand_id": hand_id,
                            "street_class": "postflop",
                            "street": street,
                            "pot_type": filt["pot_type"],
                            "hero_pos_rel": filt["hero_pos_rel"],
                            "hero_pos": hero_pos,
                            "n_players_active": filt["n_players_active"],
                            "spr_x100": spr_x100,
                            "embedding": vec,
                            "hero_action_type": dp.get("hero_action_type") or "",
                            **_size_fields(dp),
                            "active": bool(dp.get("active", True)),
                            "confidence": float(dp.get("confidence", 1.0)),
                            "gto_score": float(dp.get("gto_score", 1.0)),
                        }
                    )
                    n_postflop += 1
                else:
                    log.warning("build_embedding.unknown_street", street=street, dp_id=dp_id)
                    n_errors += 1
                    continue
            except Exception as exc:
                log.warning("build_embedding.extractor_error", dp_id=dp_id, error=str(exc))
                n_errors += 1
                continue

            n_total += 1
            chunk_count += 1

            if n_total % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = n_total / max(elapsed, 1e-9)
                log.info(
                    "build_embedding.progress",
                    n=n_total,
                    preflop=n_preflop,
                    postflop=n_postflop,
                    equity_miss=n_equity_miss,
                    errors=n_errors,
                    rate_per_s=round(rate, 1),
                )
                print(
                    f"progress: {n_total}/? elapsed={elapsed:.1f}s rate={rate:.0f}/s",
                    file=sys.stdout,
                    flush=True,
                )

            # Roll over to next chunk when chunk_size reached
            if chunk_count >= args.chunk_size:
                buf_pf.close()
                buf_po.close()
                chunk_idx += 1
                chunk_count = 0
                buf_pf = _Buffer(
                    args.out_dir / f"hm_embeddings_chunk_{chunk_idx:04d}_preflop.parquet",
                    schema_pre,
                )
                buf_po = _Buffer(
                    args.out_dir / f"hm_embeddings_chunk_{chunk_idx:04d}_postflop.parquet",
                    schema_post,
                )

            if args.limit and n_total >= args.limit:
                log.info("build_embedding.limit_reached", limit=args.limit)
                break

    total_pf = buf_pf.close()
    total_po = buf_po.close()
    total_miss = ledger.close()
    elapsed = time.time() - t0

    log.info(
        "build_embedding.complete",
        n_total=n_total,
        preflop=total_pf,
        postflop=total_po,
        equity_miss=total_miss,
        chunks=chunk_idx,
        n_errors=n_errors,
        elapsed_s=round(elapsed, 1),
    )

    # Print summary to stdout
    print(f"n_preflop={total_pf}")
    print(f"n_postflop={total_po}")
    print(f"n_equity_miss={total_miss}")
    print(f"n_errors={n_errors}")
    print(f"chunks={chunk_idx}")
    print(f"elapsed_s={elapsed:.1f}")

    return 0


def _inject_scenario_key(dp: dict) -> dict:
    """Derive and inject the equity-table scenario_key from DP fields.

    The equity table scenario format (from enumerate_equity_tuples.py):
        pot_type|hero_rel|nN|POS:role[:target][+POS:role[:target]...]

    We reconstruct the key from DP fields as a best-effort approximation.
    The exact scenario_key is not stored in the DP schema (v1.1), so we
    use a simplified version: pot_type|hero_pos_rel|n{n}|{villain_spec}

    Villain spec:
        - If preflop_aggressor is set and != hero_pos: use as PFR
        - Else: use "vNA" (unknown villain)
    """
    pot_type = dp.get("pot_type", "srp")
    hero_rel = dp.get("hero_pos_rel", "IP")
    n = dp.get("n_players_at_street", 2)
    hero_pos = dp.get("hero_pos", "")
    preflop_aggressor = dp.get("preflop_aggressor", "")

    if preflop_aggressor and preflop_aggressor != hero_pos:
        villain_spec = f"{preflop_aggressor}:pfr"
    else:
        villain_spec = "vNA"

    scenario_key = f"{pot_type}|{hero_rel}|n{n}|{villain_spec}"
    dp = dict(dp)  # shallow copy to avoid mutating caller's dict
    dp["scenario_key"] = scenario_key
    return dp


def _trailing_bet_frac(verb: str) -> float:
    """Pot-fraction for a bet/raise verb, bridging named (bet_half_pot) and frontend
    numeric (bet_50) vocab. Raises map to ~pot-sized for facing-regime purposes."""
    from src.solver.nav_line import _SIZE_FRAC

    suffix = verb.split("_", 1)[1] if "_" in verb else ""
    if suffix in _SIZE_FRAC:
        return _SIZE_FRAC[suffix]
    if verb.startswith("bet"):
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return min(int(digits) / 100.0, 2.0)
    return 1.0


def _trailing_facing_bet_bb(spot: dict, pot_size_bb: float) -> float:
    """Bet hero faces (bb), derived from the action_sequence trailing token.

    The frontend sends action_sequence but not hero_facing_bet_bb; without this the
    encoder defaults facing to 0 and every spot is embedded as checked-to. Only the
    facing fraction (bet/pot) reaches the embedding, so the pot baseline cancels.
    """
    explicit = spot.get("hero_facing_bet_bb")
    if explicit is not None:
        return float(explicit)
    seq = spot.get("action_sequence") or ()
    if not seq:
        return 0.0
    last = seq[-1]
    verb = (last.split(":", 1)[1] if ":" in last else last).lower()
    if verb == "allin":
        return pot_size_bb
    is_wager = verb.startswith(("bet", "raise", "3b", "4b", "5b"))
    return _trailing_bet_frac(verb) * pot_size_bb if is_wager else 0.0


def _gamestate_from_spot(spot: dict) -> GameState:
    """Build a GameState from a partial spot dict using the Mode B default matrix."""
    from src._errors import CanonicalizeError
    from src.protocols.game_state import GameState

    street = spot.get("street")
    if street not in {"preflop", "flop", "turn", "river"}:
        raise CanonicalizeError(f"unsupported street: {street!r}")

    pot_bb = float(spot.get("pot_size_bb", 3.0 if street == "preflop" else 6.0))
    return GameState(
        street=street,
        hero_position=spot.get("hero_position", "BTN"),
        hero_hole_cards=tuple(spot["hero_hole"]),
        board_cards=tuple(spot.get("board", ())),
        pot_size_bb=pot_bb,
        effective_stack_bb=float(spot.get("effective_stack_bb", 97.0)),
        hero_facing_bet_bb=_trailing_facing_bet_bb(spot, pot_bb),
        hero_bet_size_bb=float(spot.get("hero_bet_size_bb", 0.0)),
        action_sequence=tuple(spot.get("action_sequence", ())),
        opponents_remaining=int(spot.get("opponents_remaining", 1)),
        prior_street_aggressor=spot.get("prior_street_aggressor"),
    )


def encode_spot_to_embedding(spot: dict) -> np.ndarray:
    """Return the raw float64 embedding vector for a spot dict (Mode B). long-ok

    Delegates to Canonicalizer.default().encode(gs).embedding. The same
    default-value matrix as encode_spot_to_cluster_key applies — see that
    function for field details.

    Raises:
        CanonicalizeError: when ``spot["street"]`` is invalid.
    """
    import numpy as np

    from src.canonicalizer import Canonicalizer

    gs = _gamestate_from_spot(spot)
    enc = Canonicalizer.default().encode(gs)
    return np.asarray(enc.embedding, dtype=np.float64)


def encode_spot_to_cluster_key(spot: dict) -> str:
    """Encode a partial spot dict to a canonical cluster_key. long-ok

    Used by Probe Mode B (POST /api/probe/spot, src/study/probe.py:91).

    Delegates to ``Canonicalizer.default().encode(gs)`` and
    ``src.sim.harness._cluster_key_from_filter(enc.hard_filter)`` —
    never hand-rolls the cluster_key string. Per D-07-11b, this function
    does NOT extend ``hard_filter`` with ``street`` or ``spr_x100``; the
    cluster_key format must stay byte-identical to the 282k existing PC
    rows.

    Default-value matrix (RESEARCH §Focus Area 5):

    | GameState field        | Source            | Default                                |
    |------------------------|-------------------|----------------------------------------|
    | street                 | spot["street"]    | -                                      |
    | hero_position          | -                 | "BTN" (hero_pos_rel is IP/OOP only)    |
    | hero_hole_cards        | spot["hero_hole"] | -                                      |
    | board_cards            | spot["board"]     | () (empty for preflop)                 |
    | pot_size_bb            | -                 | 3.0 (preflop) / 6.0 (postflop)         |
    | effective_stack_bb     | -                 | 97.0 (100bb - ~3bb invested)           |
    | hero_facing_bet_bb     | -                 | 0.0 (check_to branch)                  |
    | hero_bet_size_bb       | -                 | 0.0 (not yet acted)                    |
    | action_sequence        | -                 | () (extractor: 0 raises -> pot_type=srp)|
    | opponents_remaining    | -                 | 1 (HU; n_players_active = 2)           |
    | prior_street_aggressor | -                 | None                                   |

    Mode B caveat: defaults DO encode into the cluster_key (they affect
    pot_type inference and hero_pos_rel). A spot encoded via Mode B may
    bucket to a DIFFERENT cluster_key than the same spot encoded with
    full context. Acceptable for v1 ("guess the canonical bucket"
    feature); not a precise replay tool.

    Raises:
        CanonicalizeError: when ``spot["street"]`` is not one of
            {"preflop", "flop", "turn", "river"}. Validated BEFORE
            GameState construction (closes ERR-02 exercise).
    """
    from src.canonicalizer import Canonicalizer
    from src.sim.harness import _cluster_key_from_filter

    gs = _gamestate_from_spot(spot)
    enc = Canonicalizer.default().encode(gs)
    return _cluster_key_from_filter(enc.hard_filter)


if __name__ == "__main__":
    sys.exit(main())
