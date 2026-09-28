"""Build equity decile table by dispatching tuples to equity-decile Rust binary.

Reads tools/equity_tuples.json (from enumerate_equity_tuples.py), looks up
villain range string per scenario from research/preflop-ranges, sends NDJSON
requests to the Rust binary via stdin/stdout, writes Parquet output.

Run:
    uv run python3 tools/build_equity_table.py \\
        --tuples tools/equity_tuples.json \\
        --binary ./target/release/equity-decile \\
        --out tools/equity_table.parquet \\
        --threads 12

The binary is invoked as a subprocess and fed all requests via stdin. Internal
rayon parallelism handles per-runout work; sequential dispatch keeps memory
bounded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Map palette scenario_key → list of files to read villain ranges from.
# Scenario key format: pot_type|hero_rel|nN|vVILLAIN(+VILLAIN...)
PALETTE_DIR = Path("research/preflop-ranges/outputs")


_WEIGHT_PAREN = __import__("re").compile(r"\(([\d.]+)\)")


def _normalize_range_syntax(s: str) -> str:
    """postflop-solver expects 'TT:0.5' weight syntax, not 'TT(0.5)'."""
    return _WEIGHT_PAREN.sub(r":\1", s)


def _role_to_filename_candidates(role: str, target: str | None, multiway: bool) -> list[str]:
    """Map a (role, target) to ordered list of palette filename candidates.

    multiway=True appends '_multiway' suffix as preferred variant; falls back
    to non-multiway if multiway variant absent.
    """
    mw = "_multiway" if multiway else ""

    if role == "pfr":
        # Opening raise — no target
        return [f"open{mw}.json", "open.json"]
    if role == "limper":
        return [f"limp{mw}.json", "limp.json"]
    if role == "checker":
        # BB check option (no one raised) → defending air; use limp range as proxy
        return [f"limp{mw}.json", "limp.json"]
    if role == "caller":
        # Called a single raise from `target`
        t = target or "BTN"
        return [f"defend_vs_{t}{mw}.json", f"defend_vs_{t}.json"]
    if role == "cold_caller":
        # Called a 3bet (or larger) from `target`
        t = target or "BTN"
        return [
            f"defend_3bet_vs_{t}{mw}.json",
            f"defend_3bet_vs_{t}.json",
            f"defend_vs_{t}{mw}.json",
            f"defend_vs_{t}.json",
        ]
    if role == "3bettor":
        t = target or "BTN"
        return [f"3bet_vs_{t}{mw}.json", f"3bet_vs_{t}.json"]
    if role == "cold_4bettor":
        t = target or "BTN"
        return [f"cold_4bet_vs_{t}{mw}.json", f"cold_4bet_vs_{t}.json"]
    if role == "4bettor":
        t = target or "BTN"
        return [f"4bet_vs_{t}{mw}.json", f"4bet_vs_{t}.json"]
    if role == "5bettor":
        t = target or "BTN"
        return [f"5bet_vs_{t}{mw}.json", f"5bet_vs_{t}.json"]
    return []


def lookup_villain_range(scenario_key: str) -> str | None:
    """Map scenario key with role-encoded villains to merged range string.

    Format: pot_type|hero_rel|nN|POS:role[:target][+POS:role[:target]...]
    Returns ',' joined range (postflop-solver Range parser merges duplicates),
    or None if any villain's role can't be resolved.
    """
    parts = scenario_key.split("|")
    if len(parts) != 4:
        return None
    _pot_type, _hero_rel, nstr, vstr = parts
    n = int(nstr[1:])
    villain_specs = vstr.split("+")

    if villain_specs == ["vNA"] or vstr == "vNA":
        return None

    multiway = n >= 3
    ranges = []
    for spec in villain_specs:
        bits = spec.split(":")
        if len(bits) < 2:
            return None
        pos = bits[0]
        role = bits[1]
        target = bits[2] if len(bits) > 2 else None

        candidates = _role_to_filename_candidates(role, target, multiway)
        found = None
        for c in candidates:
            p = PALETTE_DIR / pos / c
            if p.exists():
                with p.open() as f:
                    data = json.load(f)
                solver_str = data.get("solver_range_string")
                if solver_str:
                    found = _normalize_range_syntax(solver_str)
                    break
        if found is None:
            return None
        ranges.append(found)

    if not ranges:
        return None
    return ",".join(ranges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples", required=True, type=Path)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="cap requests for testing")
    args = ap.parse_args()

    tuples = json.loads(args.tuples.read_text())
    requests = []
    skipped_no_range = 0
    range_cache: dict[str, str | None] = {}

    for street in ("flop", "turn", "river"):
        for hole, board, scen in tuples["tuples"].get(street, []):
            if scen not in range_cache:
                range_cache[scen] = lookup_villain_range(scen)
            vr = range_cache[scen]
            if vr is None:
                skipped_no_range += 1
                continue
            requests.append(
                {
                    "id": f"{street}|{hole}|{board}|{scen}",
                    "hole": hole,
                    "board": board,
                    "villain_range": vr,
                }
            )
            if args.limit and len(requests) >= args.limit:
                break
        if args.limit and len(requests) >= args.limit:
            break

    print(f"Prepared {len(requests):,} requests", file=sys.stderr)
    print(f"  scenarios with no palette range: {skipped_no_range:,}", file=sys.stderr)
    print(f"  unique scenarios resolved: {sum(1 for v in range_cache.values() if v)}", file=sys.stderr)

    # Dispatch to Rust binary
    env = {"RAYON_NUM_THREADS": str(args.threads)}
    proc = subprocess.Popen(
        [str(args.binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    ids = []
    streets = []
    holes = []
    boards = []
    scens = []
    means = []
    deciles_cols = [[] for _ in range(9)]
    n_runouts = []
    errors = []

    t0 = time.time()

    # Writer thread streams requests one line at a time. Without this, large
    # payloads deadlock: Python blocks on stdin.write because Rust's stdout
    # pipe fills up, but Python isn't reading stdout yet because it's still
    # writing.
    def _writer():
        try:
            for r in requests:
                proc.stdin.write(json.dumps(r) + "\n")
            proc.stdin.flush()
        finally:
            proc.stdin.close()

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()

    n_ok = 0
    n_err = 0
    for _i, line in enumerate(proc.stdout):
        line = line.strip()
        if not line:
            continue
        try:
            resp = json.loads(line)
        except Exception as e:
            print(f"  bad response line: {line[:200]!r} ({e})", file=sys.stderr)
            n_err += 1
            continue
        rid = resp["id"]
        parts = rid.split("|")
        if len(parts) < 4:
            n_err += 1
            continue
        street, hole, board, scen = parts[0], parts[1], parts[2], "|".join(parts[3:])
        if resp.get("error"):
            errors.append((rid, resp["error"]))
            n_err += 1
            continue
        ids.append(rid)
        streets.append(street)
        holes.append(hole)
        boards.append(board)
        scens.append(scen)
        means.append(resp["mean"])
        for j in range(9):
            deciles_cols[j].append(resp["deciles"][j])
        n_runouts.append(resp["n_runouts"])
        n_ok += 1
        if n_ok % 1000 == 0:
            elapsed = time.time() - t0
            rate = n_ok / elapsed
            eta = (len(requests) - n_ok) / rate
            print(
                f"  {n_ok:>7,} / {len(requests):,}  rate={rate:.0f}/s  eta={eta / 60:.1f}min",
                file=sys.stderr,
            )

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        print(f"\n  binary exited {proc.returncode}: {stderr[:500]}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok:,} ok, {n_err:,} errors in {elapsed / 60:.1f} min", file=sys.stderr)

    if errors[:5]:
        print("  first 5 errors:", file=sys.stderr)
        for rid, msg in errors[:5]:
            print(f"    {rid}: {msg}", file=sys.stderr)

    table = pa.table(
        {
            "id": ids,
            "street": streets,
            "hole_canonical": holes,
            "board_canonical": boards,
            "scenario": scens,
            "mean_equity": means,
            "p10": deciles_cols[0],
            "p20": deciles_cols[1],
            "p30": deciles_cols[2],
            "p40": deciles_cols[3],
            "p50": deciles_cols[4],
            "p60": deciles_cols[5],
            "p70": deciles_cols[6],
            "p80": deciles_cols[7],
            "p90": deciles_cols[8],
            "n_runouts": n_runouts,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB, {n_ok:,} rows)")


if __name__ == "__main__":
    main()
