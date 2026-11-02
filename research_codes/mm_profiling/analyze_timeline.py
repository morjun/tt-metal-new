#!/usr/bin/env python3
"""M3 of gemma4-12b record §9.2b: per-core cross-RISC timelines of up_proj L0 in arms c (DRAM-fed) and d (L1-fed).

Reads run_traced_opprof_arms.py cells with "+timeline<mask>" (timeline_samples.csv: device-log rows MM_TL, data =
(event << 8) | block) and their ops_perf.csv. Markers (MM_TIMELINE in the matmul kernels):
  BRISC  0x10 start, 0x11 CB-space wait entry*, 0x12 block's weight reads issued (after CB space), 0x13 block's read
         barrier returned, 0x14 whole L1 shard pushed (once), 0x15 / 0x16 output wait entry / exit* (per subblock),
         0x17 all output writes complete
  NCRISC sender 0x20 activation read complete, 0x21 / 0x22 receivers-ready wait entry / exit, 0x23 multicast issued;
         receivers 0x28 / 0x29 VALID wait entry / exit (activation block delivered)
  TRISC0 0x30 / 0x31 in0 wait entry / exit*, 0x32 both inputs available; TRISC1 0x33 block math done*;
  TRISC2 0x34 block packed
(* = sets 4 / 8; 0x10, 0x12-0x14, 0x17, 0x32, 0x34 = set 1, the minimal set in every cell; 0x2x = set 2.)

Per call of up_proj L0 (the 4th matmul of each step): times in µs from the earliest 0x10; the slowest core = the one
with the latest 0x17. Reports per (arm, mask), median over calls:
  - checks: markers inside the op's firmware window; marker counts per core against the expected counts
  - the slowest core's path: start -> each block's inputs available -> packed -> output complete, the binding input
    of each block (weight 0x13 / 0x14 vs activation 0x29 / 0x23 when set 2 is on), compute per block (0x32 -> 0x34),
    waiting before each block (previous 0x34 -> 0x32)
  - observed concurrency in c: per core, the time compute intervals [0x32, 0x34] of block b overlap the read
    intervals [0x12, 0x13] of later blocks. It is concurrency, not the latency an overlap saves
  - perturbation: up L0 kernel time of the marked cells vs the unmarked cells of the same arm

    python research_codes/mm_profiling/analyze_timeline.py generated/lmhead_l1_profile/m3_1103 -> stdout + m3.json
"""

import argparse
import bisect
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_resource_profile import load  # noqa: E402

MHZ = 1350.0


def read_markers(path):
    out = []
    with open(path) as f:
        next(f)
        for r in csv.DictReader(f, skipinitialspace=True):
            if (r.get("zone name") or "").strip() == "MM_TL":
                d = int(r["data"])
                out.append(
                    (
                        int(r["time[cycles since reset]"]),
                        (int(r["core_x"]), int(r["core_y"])),
                        r["RISC processor type"].strip(),
                        d >> 8,
                        d & 0xFF,
                    )
                )
    return sorted(out)


def up_calls(cell):
    """[(fw_start, fw_end, dur, session)] of up_proj L0 and every op window (for assignment)."""
    allops, ups = [], []
    for sid, ops in load(cell / "ops_perf.csv").items():
        step = []
        for o in ops:
            step.append(o)
            if o["op"] == "ArgMax":
                mms = [x for x in step if x["op"] == "Matmul"]
                ups.append((mms[4]["fw_start"], mms[4]["fw_end"], mms[4]["dur"], sid))
                step = []
        for o in ops:
            allops.append((o["fw_start"], o["fw_end"]))
    return sorted(allops), sorted(ups)


def call_metrics(marks, l1_fed):
    """marks: {core: {(event, block): t_us}} for one call."""
    t0 = min(v[(0x10, 0)] for v in marks.values() if (0x10, 0) in v)
    ends = {c: v[(0x17, 0)] for c, v in marks.items() if (0x17, 0) in v}
    slow = max(ends, key=ends.get)
    ev = marks[slow]
    nb = 1 + max(b for (e, b) in ev if e == 0x32)
    m = {"marker_span": max(ends.values()) - t0, "slow_core": slow, "slow_end": ends[slow] - t0}
    act = {}
    for b in range(nb):
        act[b] = ev.get((0x29, b), ev.get((0x23, b)))  # receiver: VALID arrived; sender core: multicast issued
    w = {b: ev.get((0x14, 0)) if l1_fed else ev.get((0x13, b)) for b in range(nb)}
    prev = t0
    for b in range(nb):
        r, pk = ev.get((0x32, b)), ev.get((0x34, b))
        m[f"b{b}_inputs_ready"] = r - t0
        m[f"b{b}_wait_before"] = r - prev  # from the previous block packed (or start) to these inputs available
        m[f"b{b}_compute"] = pk - r
        m[f"b{b}_weight_avail"] = (w[b] - t0) if w[b] is not None else None
        m[f"b{b}_act_avail"] = (act[b] - t0) if act[b] is not None else None
        if w[b] is not None and act[b] is not None:
            m[f"b{b}_binding"] = "weight" if w[b] >= act[b] else "activation"
        prev = pk
    m["output_after_last_pack"] = ends[slow] - ev[(0x34, nb - 1)]
    if not l1_fed and (0x13, nb - 1) in ev:
        m["compute_after_last_weight"] = ev[(0x34, nb - 1)] - ev[(0x13, nb - 1)]
    if l1_fed and (0x14, 0) in ev:
        m["shard_push"] = ev[(0x14, 0)] - ev[(0x10, 0)]
    # observed concurrency (c): compute of block b vs reads of later blocks, per core
    if not l1_fed:
        conc = []
        for c, v in marks.items():
            tot = 0.0
            for b in range(nb):
                if (0x32, b) not in v or (0x34, b) not in v:
                    continue
                for b2 in range(b + 1, nb):
                    if (0x12, b2) in v and (0x13, b2) in v:
                        lo, hi = max(v[(0x32, b)], v[(0x12, b2)]), min(v[(0x34, b)], v[(0x13, b2)])
                        tot += max(0.0, hi - lo)
            conc.append(tot)
        m["concurrency_median_core"] = statistics.median(conc)
        m["concurrency_slow_core"] = conc[list(marks).index(slow)]
    # per-core compute total (sum of block compute) median over cores
    comp = [
        sum(v[(0x34, b)] - v[(0x32, b)] for b in range(nb) if (0x32, b) in v and (0x34, b) in v) for v in marks.values()
    ]
    m["compute_sum_median_core"] = statistics.median(comp)
    return m


def cell_analysis(cell, l1_fed, mask):
    allops, ups = up_calls(cell)
    markers = read_markers(cell / "timeline_samples.csv")
    starts = [u[0] for u in ups]
    per = defaultdict(lambda: defaultdict(dict))
    counts = defaultdict(lambda: defaultdict(int))
    outside = inside = 0
    for t, core, risc, e, b in markers:
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t > ups[i][1]:
            continue  # another 32x256 matmul (gate / up of other layers)
        per[i][core][(e, b)] = t / MHZ
        counts[i][(core, risc)] += 1
        inside += 1
    # clock alignment: markers that fall in no op's firmware window at all
    ostarts = [a[0] for a in allops]
    for t, *_ in markers:
        j = bisect.bisect_right(ostarts, t) - 1
        if j < 0 or t > allops[j][1]:
            outside += 1
    calls = []
    for i, marks in sorted(per.items()):
        try:
            calls.append(call_metrics(marks, l1_fed))
        except (KeyError, ValueError, TypeError):
            continue
    cnt = [sum(v.values()) for v in counts.values()]
    per_core = defaultdict(list)
    for i, v in counts.items():
        for (core, risc), n in v.items():
            per_core[risc].append(n)
    return {
        "cell": cell.name,
        "mask": mask,
        "l1_fed": l1_fed,
        "up_dur": [u[2] for u in ups],
        "calls_with_markers": len(per),
        "calls_analysed": len(calls),
        "markers_per_call": statistics.median(cnt) if cnt else 0,
        "markers_per_core_by_risc": {k: (min(v), statistics.median(v), max(v)) for k, v in per_core.items()},
        "cores_per_call": statistics.median(len(v) for v in per.values()) if per else 0,
        "markers_outside_any_op": outside,
        "calls": calls,
    }


def summarize(calls, key):
    xs = [c[key] for c in calls if c.get(key) is not None and not isinstance(c[key], (str, tuple))]
    return statistics.median(xs) if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    cells = sorted(d for d in args.root.iterdir() if (d / "ops_perf.csv").exists())
    base_dur = defaultdict(list)
    res = []
    for d in cells:
        arm = re.match(r"\d+_time_(.*)", d.name).group(1)
        l1_fed = "+pinup_proj" in arm
        m = re.search(r"\+timeline(\d+)", arm)
        if not m:
            _, ups = up_calls(d)
            base_dur["d" if l1_fed else "c"] += [u[2] for u in ups]
            continue
        if not (d / "timeline_samples.csv").exists():
            continue
        res.append(cell_analysis(d, l1_fed, int(m.group(1))))
    out = {
        "unmarked": {
            k: {"mean": statistics.mean(v), "sd": statistics.stdev(v), "n": len(v)} for k, v in base_dur.items()
        },
        "cells": res,
    }
    groups = defaultdict(list)
    for r in res:
        groups[("d" if r["l1_fed"] else "c", r["mask"])].append(r)
    print("## perturbation and capture (up L0, µs per call)")
    print(
        "| arm | mask | marked mean (n) | unmarked mean | Δ | 1% limit | markers / call | cores | by RISC (min / median / max per core) | outside any op |"
    )
    print("|---|--:|--:|--:|--:|--:|--:|--:|---|--:|")
    summary = {}
    for (arm, mask), rs in sorted(groups.items()):
        durs = [x for r in rs for x in r["up_dur"]]
        um = out["unmarked"].get(arm, {}).get("mean")
        dm = statistics.mean(durs)
        calls = [c for r in rs for c in r["calls"]]
        keys = sorted({k for c in calls for k in c if k not in ("slow_core",) and not k.endswith("binding")})
        s = {k: summarize(calls, k) for k in keys}
        s["binding"] = {f"b{b}": dict(Counter(c.get(f"b{b}_binding") for c in calls)) for b in range(4)}
        s["perturbation"] = {
            "marked": dm,
            "unmarked": um,
            "delta": (dm - um) if um else None,
            "limit": 0.01 * um if um else None,
        }
        summary[f"{arm} mask {mask}"] = s
        print(
            f"| {arm} | {mask} | {dm:.3f} ({len(durs)}) | {um:.3f} | {dm - um:+.3f} | {0.01 * um:.3f} | {statistics.median(r['markers_per_call'] for r in rs)} | "
            f"{statistics.median(r['cores_per_call'] for r in rs)} | {rs[0]['markers_per_core_by_risc']} | {sum(r['markers_outside_any_op'] for r in rs)} |"
        )
    print("\n## slowest-core path, median over calls (µs from the earliest start marker)")
    for k, s in summary.items():
        print(f"\n### {k}")
        print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in s.items()}, indent=None))
    out["summary"] = summary
    (args.root / "m3.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
