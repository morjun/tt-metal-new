#!/usr/bin/env python3
"""Per-arm op times over repeated traced time cells, and per-row NIU shares (gemma4-12b record §9.2a C1 / C2, §9.2b C3).

    python item6/campaign_tables.py time <root>               -> per (label, layer): µs per iteration, mean and
                                                                  per-cell values for every arm (cells = rounds)
    python item6/campaign_tables.py rows <cell> <label>       -> per physical row: mean per-core delivered GB/s and
                                                                  last-sample time (µs from the op's first sample)

Arm = the cell directory name without its "NN_time_" prefix. Op labels come from
analyze_resource_profile.label_session (lm_head calls carry "(whole, DRAM)", "(L1 slice)", "(DRAM tail)").
"""

import bisect
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_niu_samples import MHZ, read_samples  # noqa: E402  (same directory)
from analyze_resource_profile import label_session, load  # noqa: E402
from compare_opprof_cells import cell_times  # noqa: E402

LAYER_TYPES = ["sliding", "sliding", "sliding", "global"]


def time_tables(root):
    arms = defaultdict(list)
    for d in sorted(root.iterdir()):
        if (d / "ops_perf.csv").exists():
            arms[d.name.split("_", 2)[2]].append(cell_times(d))
    out = {}
    for arm, cells in arms.items():
        keys = set().union(*(c[0] for c in cells))
        out[arm] = {
            "cells": len(cells),
            "span": [c[1] for c in cells],
            "ops": {f"{n} | {l}": [c[0].get((n, l)) for c in cells] for n, l in sorted(keys, key=str)},
        }
    (root / "campaign_times.json").write_text(json.dumps(out, indent=1) + "\n")
    for arm, r in out.items():
        print(
            f"## {arm}: {r['cells']} cells, span µs/iter {statistics.mean(r['span']):.1f} ({', '.join(f'{x:.1f}' for x in r['span'])})"
        )
    return out


def rows(cell, label):
    samples = read_samples(cell / "niu_samples.csv")
    ops = []
    for session in load(cell / "ops_perf.csv").values():
        for o, (_, name, _) in zip(session, label_session(session, LAYER_TYPES)):
            if o["op"] == "Matmul":
                ops.append((o["fw_start"], o["fw_end"], name))
    ops.sort()
    starts = [o[0] for o in ops]
    tol = 2.0 * MHZ
    per = defaultdict(lambda: defaultdict(list))
    for t, core, w in samples:
        i = bisect.bisect_right(starts, t + tol) - 1
        if i > 0 and t <= ops[i - 1][1]:
            i -= 1
        if i >= 0 and t <= ops[i][1] and ops[i][2] == label:
            per[i][core].append((t / MHZ, w))
    by_row = defaultdict(lambda: {"gbs": [], "end": []})
    for cores in per.values():
        t0 = min(v[0][0] for v in cores.values() if v)
        for (x, y), v in cores.items():
            if len(v) < 2:
                continue
            words = (v[-1][1] - v[0][1]) % (1 << 32)
            by_row[y]["gbs"].append(words * 64 / (v[-1][0] - v[0][0]) / 1e3)
            by_row[y]["end"].append(v[-1][0] - t0)
    print(f"{label}: {len(per)} calls")
    print("| physical row | cores × calls | per-core delivered GB/s | last sample µs |")
    print("|--:|--:|--:|--:|")
    for y in sorted(by_row):
        r = by_row[y]
        print(f"| {y} | {len(r['gbs'])} | {statistics.mean(r['gbs']):.2f} | {statistics.mean(r['end']):.1f} |")


if __name__ == "__main__":
    if sys.argv[1] == "time":
        time_tables(Path(sys.argv[2]))
    else:
        rows(Path(sys.argv[2]), sys.argv[3])
