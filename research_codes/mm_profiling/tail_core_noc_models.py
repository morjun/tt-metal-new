#!/usr/bin/env python3
"""Which NoC property of a core's position predicts its read time in the drafter-tail matmul?

Input: generated/lmhead_l1_profile/tail_zones/per_core.json (map_tail_cores.py). Model: NoC0
(DRAM reads, preferred_noc_for_dram_read), dimension order +x then +y on the 17 x 12 Blackhole
torus. NoC0 DRAM endpoints from soc_descriptors/blackhole_140_arch.yaml (dram[ch][worker_endpoint[0]]).
Prints Spearman correlation of each candidate metric with per-core own read time, per width.
The last core (15,11), which owns 6 columns, is excluded.
"""

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

D = json.loads(
    (Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/tail_zones/per_core.json").read_text()
)
X, Y = 17, 12
EP = [(0, 11), (0, 2), (0, 9), (0, 5), (9, 11), (9, 3), (9, 8), (9, 6)]  # NoC0 endpoint of channels 0..7
CORES = [tuple(map(int, k.split(","))) for k in D["8"] if k != "15,11"]
WIDTHS = ["1", "2", "4", "8", "16", "32", "64"]


def route(s, d):
    (x, y), hops, inp = s, [], "local"
    while x != d[0]:
        hops.append(((x, y), "E", inp))
        x = (x + 1) % X
        inp = "W"
    while y != d[1]:
        hops.append(((x, y), "S", inp))
        y = (y + 1) % Y
        inp = "N"
    hops.append(((x, y), "eject", inp))
    return hops


def link_loads(direction):
    load, paths = defaultdict(int), {}
    for c in CORES:
        for e in EP:
            p = [(r, o) for r, o, _ in (route(c, e) if direction == "req" else route(e, c)) if o != "eject"]
            paths[(c, e)] = p
            for l in p:
                load[l] += 1
    return load, paths


def parking_lot(direction):
    flows = {(c, e): (route(c, e) if direction == "req" else route(e, c)) for c in CORES for e in EP}
    ports = defaultdict(set)
    for h in flows.values():
        for r, o, i in h:
            ports[(r, o)].add(i)
    return {f: math.prod(1 / len(ports[(r, o)]) for r, o, _ in h) for f, h in flows.items()}


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
            j += 1
        for k in range(i, j + 1):
            r[s[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    a, b = rank(a), rank(b)
    ma, mb = st.mean(a), st.mean(b)
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return float("nan") if den == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den  # nan: metric constant


def main():
    load, paths = link_loads("resp")
    pl_req, pl_resp = parking_lot("req"), parking_lot("resp")
    metrics = {
        "row (physical y)": lambda c: c[1],
        "mean response hops": lambda c: st.mean(len(paths[(c, e)]) for e in EP),
        "mean vertical hops": lambda c: st.mean(sum(1 for (r, o) in paths[(c, e)] if o == "S") for e in EP),
        "max link load on path": lambda c: max(max((load[l] for l in paths[(c, e)]), default=0) for e in EP),
        "sum link load on paths": lambda c: sum(load[l] for e in EP for l in paths[(c, e)]),
        "parking lot, request": lambda c: sum(1 / pl_req[(c, e)] for e in EP),
        "parking lot, response": lambda c: max(1 / pl_resp[(c, e)] for e in EP),
        "response y-wraps (rows 0-1)": lambda c: sum(1 for _, ey in EP if ey > c[1]),
        "response x-wraps (16->0)": lambda c: sum(1 for ex, _ in EP if ex > c[0]),
    }
    print(f"{'metric':>30}" + "".join(f"{'w' + w:>7}" for w in WIDTHS))
    for name, f in metrics.items():
        m = [f(c) for c in CORES]
        print(
            f"{name:>30}"
            + "".join(f"{spearman(m, [D[w][f'{c[0]},{c[1]}']['own_read'] for c in CORES]):+7.2f}" for w in WIDTHS)
        )
    for w in ("8", "64"):
        res = []
        for y in range(2, 12):
            cs = [c for c in CORES if c[1] == y]
            med = st.median(D[w][f"{c[0]},{c[1]}"]["own_read"] for c in cs)
            res += [(c, D[w][f"{c[0]},{c[1]}"]["own_read"] - med) for c in cs]
        print(
            f"width {w}: within-row residual vs response x-wraps: "
            f"{spearman([metrics['response x-wraps (16->0)'](c) for c, _ in res], [v for _, v in res]):+.2f}"
        )


if __name__ == "__main__":
    main()
