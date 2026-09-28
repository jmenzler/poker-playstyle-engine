"""ANN p99 latency benchmark for ROADMAP Phase 2 §Success Criteria 3 (Plan 09).

Loads collection, warms cache via 1 throwaway query, then runs N kNN queries with
optional hard filter and records per-query elapsed time. Reports p50/p90/p99 in ms
and exits non-zero if p99 > 5ms.

Usage:
    python tools/bench_knn_p99.py --collection postflop_decisions --n-queries 1000 \\
        --filter "street_class == 'postflop' and pot_type == 'srp'"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src._log import configure_logging, get_logger
from tools.upsert_milvus import _milvus_client_from_env

log = get_logger("tools.bench_knn_p99")

P99_BUDGET_MS = 5.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ANN p99 latency benchmark for ROADMAP Phase 2 §Success Criteria 3"
    )
    ap.add_argument(
        "--collection",
        required=True,
        choices=["preflop_decisions", "postflop_decisions"],
    )
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument(
        "--filter",
        default="",
        help="Scalar filter expression (e.g. \"street_class == 'postflop' and pot_type == 'srp'\")",
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

    rng = np.random.default_rng(args.seed)
    dim = 32 if args.collection == "preflop_decisions" else 80

    client, _uri, _token = _milvus_client_from_env()
    client.load_collection(args.collection)

    filter_expr = args.filter or None

    # Sample real embeddings as query vectors — random vectors land far from
    # any cluster in 32/80-dim space and force HNSW to do more graph traversal,
    # which is not what production sees.
    log.info("bench_knn_p99.sampling_real_queries", collection=args.collection)
    sample = client.query(
        collection_name=args.collection,
        filter=filter_expr or "",
        output_fields=["embedding"],
        limit=args.n_queries + 1,
    )
    query_vecs = [r["embedding"] for r in sample]
    if len(query_vecs) < args.n_queries + 1:
        # Fall back to random for the shortfall, but warn
        shortfall = args.n_queries + 1 - len(query_vecs)
        log.warning("bench_knn_p99.real_query_shortfall", shortfall=shortfall)
        for _ in range(shortfall):
            query_vecs.append(rng.random(dim).astype(np.float32).tolist())

    # Warm cache: 1 throwaway query excluded from latency stats
    client.search(
        collection_name=args.collection,
        data=[query_vecs[0]],
        limit=args.k,
        filter=filter_expr,
        output_fields=["decision_id"],
    )
    log.info("bench_knn_p99.warmup_done", collection=args.collection)

    # Bench loop using real query vectors
    latencies_ms: list[float] = []
    for i in range(args.n_queries):
        qv = query_vecs[i + 1]
        t0 = time.perf_counter()
        client.search(
            collection_name=args.collection,
            data=[qv],
            limit=args.k,
            filter=filter_expr,
            output_fields=["decision_id"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        if (i + 1) % 100 == 0:
            log.info("bench_knn_p99.progress", n=i + 1, total=args.n_queries)

    arr = np.array(latencies_ms)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    mean = float(arr.mean())

    log.info(
        "bench_knn_p99.result",
        collection=args.collection,
        n=args.n_queries,
        p50_ms=round(p50, 3),
        p90_ms=round(p90, 3),
        p99_ms=round(p99, 3),
        mean_ms=round(mean, 3),
        filter=filter_expr,
    )

    print(f"\nResults ({args.collection}, n={args.n_queries}, filter={filter_expr or 'none'}):")
    print(f"  p50  = {p50:.3f} ms")
    print(f"  p90  = {p90:.3f} ms")
    print(f"  p99  = {p99:.3f} ms  (budget: {P99_BUDGET_MS} ms)")
    print(f"  mean = {mean:.3f} ms")

    if p99 > P99_BUDGET_MS:
        print(f"\nFAIL: p99 {p99:.3f}ms exceeds {P99_BUDGET_MS}ms budget", file=sys.stderr)
        return 1

    print("\nPASS: p99 within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
