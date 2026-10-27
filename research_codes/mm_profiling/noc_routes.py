#!/usr/bin/env python3
"""Dimension-order NoC routes on the Blackhole physical torus (gemma4-12b record §9.2e, plan v3 H4 / G5).

Physical NoC0 coordinates (x 0-16, y 0-11), as tt-npe and the device log use them. NoC0 routes east (+x) first, then
south (+y); NoC1 routes north (-y) first, then west (-x); both wrap. (The NoC1 order was found by the validation below:
plan v3 G5 had written "west then north".) A link is (y, x, kind): the link leaving router (x, y) in
direction kind, in tt-npe's naming (NOC0_EAST, NOC0_SOUTH, NOC1_WEST, NOC1_NORTH).

A read's request travels core -> endpoint and its response endpoint -> core, both on the issuing NoC. On a
unidirectional ring the two cover the ring once, so request + response hops = 17 in x when the x coordinates differ and
12 in y when the y coordinates differ (plan v3 G5).

    python research_codes/mm_profiling/noc_routes.py validate   -> compares response routes with tt-npe's recorded
                                                                    routes of DRAM reads (T2 W cell on NoC0, the
                                                                    phase2_nocswap cell on NoC1)
"""

import json
import random
import subprocess
import sys
from pathlib import Path

COLS, ROWS = 17, 12
# bank -> NoC0 endpoint, physical (x, y), from the T2 down_proj reads (plan v3 G1)
ENDPOINTS_NOC0 = {0: (0, 11), 1: (0, 2), 2: (0, 9), 3: (0, 5), 4: (9, 11), 5: (9, 3), 6: (9, 8), 7: (9, 6)}


def route(src, dst, noc=0):
    """Directed links from src to dst (physical (x, y)), dimension order with wrap."""
    (x, y), (tx, ty) = src, dst
    links = []
    if noc == 0:
        while x != tx:
            links.append((y, x, "NOC0_EAST"))
            x = (x + 1) % COLS
        while y != ty:
            links.append((y, x, "NOC0_SOUTH"))
            y = (y + 1) % ROWS
    else:
        while y != ty:
            links.append((y, x, "NOC1_NORTH"))
            y = (y - 1) % ROWS
        while x != tx:
            links.append((y, x, "NOC1_WEST"))
            x = (x - 1) % COLS
    return links


def request_route(core, endpoint, noc=0):
    return route(core, endpoint, noc)


def response_route(core, endpoint, noc=0):
    return route(endpoint, core, noc)


def hops(core, endpoint, noc=0):
    """(request hops, response hops)."""
    return len(request_route(core, endpoint, noc)), len(response_route(core, endpoint, noc))


def validate(n=1000, seed=0):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib

    matplotlib.use("Agg")
    from plot_dram_read_grid import T2, CELLS, npeviz

    cells = [
        ("W", T2 / "00_noc_B+logitsl1+gridwide+vc1", "gate_proj (sliding)"),
        ("W", T2 / "00_noc_B+logitsl1+gridwide+vc1", "down_proj (sliding)"),
        ("NoC swap", CELLS.parent / "phase2_nocswap/05_noc_B+logitsl1+nocswap", "gate_proj (sliding)"),
    ]
    rng = random.Random(seed)
    ok = total = 0
    for name, cell, label in cells:
        d = npeviz(cell, label, 0)
        reads = [t for t in d["noc_transfers"] if t["noc_event_type"] == "READ"]
        dram = {tuple(t["src"]) for t in reads} - {tuple(x) for t in reads for x in t["dst"]}
        reads = [t for t in reads if tuple(t["src"]) in dram]
        sample = rng.sample(reads, min(n, len(reads)))
        bad = 0
        per_noc = {0: [0, 0], 1: [0, 0]}
        for t in sample:
            noc = 0 if t["route"][0]["noc_type"] == "NOC0" else 1
            sy, sx = t["src"][1:]
            dy, dx = t["dst"][0][1:]
            ours = response_route((dx, dy), (sx, sy), noc)
            theirs = [(r, c, k) for rt in t["route"] for _, r, c, k in rt["links"] if not k.endswith(("_IN", "_OUT"))]
            per_noc[noc][1] += 1
            if ours == theirs:
                ok += 1
                per_noc[noc][0] += 1
            else:
                bad += 1
                if bad <= 3:
                    print(
                        f"  mismatch {name} {label}: src {(sx, sy)} dst {(dx, dy)}\n    ours   {ours}\n    tt-npe {theirs}"
                    )
            total += 1
        print(
            f"{name} {cell.name} {label}: {len(sample) - bad} / {len(sample)} routes match"
            f" (NoC0 {per_noc[0][0]} / {per_noc[0][1]}, NoC1 {per_noc[1][0]} / {per_noc[1][1]})"
        )
    print(f"total {ok} / {total} ({100 * ok / total:.1f}%)")
    return ok == total


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        sys.exit(0 if validate() else 1)
    print(json.dumps({b: dict(endpoint=e) for b, e in ENDPOINTS_NOC0.items()}))
