#!/usr/bin/env python3
"""Per-core output segments of up_proj L0 in the M3 / M3b cells (gemma4-12b record §9.2b, "The output write").

For every call and core: output segment = all output writes complete (BRISC 0x17) - last block's pack end (TRISC2
0x34, block 3). Concurrent writers of a core = other cores whose [last pack, writes complete] interval overlaps its own.
In the DRAM-fed arm also counts the other cores whose last weight read barrier (0x13, block 3) returns after this
core's last pack. With the output-wait exits (0x16, sets 4 / 64) it splits the segment per output subblock (each core
writes one tile per subblock, then waits on a write barrier).

    python research_codes/mm_profiling/analyze_output_segments.py generated/lmhead_l1_profile/m3_1103/{02..17}_* \
        generated/lmhead_l1_profile/m3b_1103/{02..17}_* --json generated/lmhead_l1_profile/m3b_1103/output_segments.json
"""

import argparse
import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_timeline import MHZ, read_markers, up_calls

LAST = 3  # up L0: 4 K-blocks at iw8
BINS = (0, 20, 40, 60, 80)


def calls(cell):
    _, ups = up_calls(cell)
    starts = [u[0] for u in ups]
    per = defaultdict(lambda: defaultdict(dict))
    for t, core, _, e, b in read_markers(cell / "timeline_samples.csv"):
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t > ups[i][1]:
            continue
        per[i][core][(e, b)] = t / MHZ
    return [m for _, m in sorted(per.items())]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def binned(pairs):
    out = defaultdict(list)
    for k, v in pairs:
        out[max(b for b in BINS if k >= b)].append(v)
    return {f"{b}-{b + 19}" if b < 80 else "80+": (statistics.median(v), len(v)) for b, v in sorted(out.items())}


def cell(path):
    l1 = "pinup" in path.name
    seg, wr, rd, slowest, spread = [], [], [], [], []
    sub = defaultdict(list)
    per_core = defaultdict(list)
    for marks in calls(path):
        ok = {c: v for c, v in marks.items() if (0x17, 0) in v and (0x34, LAST) in v}
        if len(ok) < 80:
            continue
        P = {c: v[(0x34, LAST)] for c, v in ok.items()}
        E = {c: v[(0x17, 0)] for c, v in ok.items()}
        last = max(E, key=E.get)
        ps = sorted(P.values())
        spread.append((ps[-1] - ps[0], ps[int(0.95 * len(ps))] - ps[int(0.05 * len(ps))]))
        slowest.append(E[last] - P[last])
        for c, v in ok.items():
            o = E[c] - P[c]
            seg.append(o)
            per_core[f"{c[0]},{c[1]}"].append(o)
            wr.append((sum(1 for c2 in ok if c2 != c and P[c2] < E[c] and E[c2] > P[c]), o))
            if not l1:
                rd.append((sum(1 for c2, v2 in ok.items() if c2 != c and v2.get((0x13, LAST), 0) > P[c]), o))
            ex = sorted(t for (e, _), t in v.items() if e == 0x16)
            if len(ex) == 3:
                sub["first wait exit - last pack"].append(ex[0] - P[c])
                sub["tile 0 write (exit 0 -> exit 1)"].append(ex[1] - ex[0])
                sub["tile 1 write (exit 1 -> exit 2)"].append(ex[2] - ex[1])
                sub["tile 2 write (exit 2 -> complete)"].append(E[c] - ex[2])
    pc = {c: statistics.median(v) for c, v in per_core.items()}
    return {
        "cell": path.name,
        "feed": "l1" if l1 else "dram",
        "core_calls": len(seg),
        "segment": {"median": statistics.median(seg), "p90": q(seg, 0.9), "max": max(seg)},
        "slowest_core_segment_median": statistics.median(slowest),
        "last_pack_spread_median": statistics.median(x for x, _ in spread),
        "last_pack_spread_5_95_median": statistics.median(y for _, y in spread),
        "by_concurrent_writers": binned(wr),
        "by_cores_still_reading_last_block": binned(rd) if rd else None,
        "per_subblock_median_p90": {k: (statistics.median(v), q(v, 0.9)) for k, v in sub.items()} or None,
        "slowest_cores_by_median": sorted(pc.items(), key=lambda kv: -kv[1])[:6],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cells", nargs="+", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    res = [cell(c) for c in a.cells]
    for r in res:
        s = r["segment"]
        print(
            f"{r['cell'][:2]} {r['feed']:4s} n={r['core_calls']} segment median {s['median']:.2f} p90 {s['p90']:.2f} "
            f"max {s['max']:.2f}; slowest core's {r['slowest_core_segment_median']:.2f}; last-pack spread "
            f"{r['last_pack_spread_median']:.2f} (5-95% {r['last_pack_spread_5_95_median']:.2f})"
        )
        print(
            "   by concurrent writers:",
            ", ".join(f"{k}: {m:.2f} ({n})" for k, (m, n) in r["by_concurrent_writers"].items()),
        )
        if r["by_cores_still_reading_last_block"]:
            print(
                "   by cores still reading:",
                ", ".join(f"{k}: {m:.2f} ({n})" for k, (m, n) in r["by_cores_still_reading_last_block"].items()),
            )
        if r["per_subblock_median_p90"]:
            print(
                "   per subblock:",
                ", ".join(f"{k} {m:.2f}/{x:.2f}" for k, (m, x) in r["per_subblock_median_p90"].items()),
            )
        print("   slowest cores:", ", ".join(f"({c}) {m:.2f}" for c, m in r["slowest_cores_by_median"]))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
