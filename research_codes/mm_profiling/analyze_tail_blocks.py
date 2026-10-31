#!/usr/bin/env python3
"""Per-block progress of the lm_head DRAM tail from NIU samples (gemma4-12b record §9.4, tail slowdown profiling).

For each niusample time cell (run_traced_opprof_arms.py, "+niusample": every core's NIU read-word counter after every
k-th K-block of the in1 reader), per call of the DRAM tail matmul (weight wider than 100,000 columns, in DRAM):
  - completion time of each sampled block on each core, from the earliest first sample of the call
  - spread per sampled block (max - min across cores) and which core is last
  - per core: mean lag behind the median core over the blocks
  - per core: median interval between consecutive samples (the per-block pace)
With a reference cell (the first argument), cores absent from it are reported as "added" and their share of last
places is given.

    python research_codes/mm_profiling/analyze_tail_blocks.py <ref cell> <cell> [...]
"""

import bisect
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_resource_profile import load  # noqa: E402
from item6.analyze_niu_samples import MHZ, read_samples  # noqa: E402


def width(o):
    x = o["row"].get("INPUT_1_X_PAD[LOGICAL]", "")
    return int(x.split("[")[1].rstrip("]")) if "[" in x else 0


def tail_calls(cell):
    """[{core: [(cycle, words), ...]}] per call of the DRAM tail."""
    samples = read_samples(cell / "niu_samples.csv")
    ops = []
    for s in load(cell / "ops_perf.csv").values():
        for o in s:
            if o["op"] == "Matmul":
                tail = width(o) > 100_000 and "DRAM" in o["row"].get("INPUT_1_MEMORY", "")
                ops.append((o["fw_start"], o["fw_end"], tail))
    ops.sort()
    starts = [o[0] for o in ops]
    calls = defaultdict(lambda: defaultdict(list))
    for t, core, w in samples:
        i = bisect.bisect_right(starts, t + 2 * MHZ) - 1
        if i > 0 and t <= ops[i - 1][1]:
            i -= 1
        if i >= 0 and t <= ops[i][1] and ops[i][2]:
            calls[i][core].append((t, w))
    return [dict(c) for _, c in sorted(calls.items())]


def analyze(cell, ref_cores=None):
    calls = tail_calls(cell)
    spreads, lags, paces, last, ops = [], defaultdict(list), defaultdict(list), Counter(), []
    full = None
    for cores in calls:
        n = statistics.median(len(s) for s in cores.values())
        full = [k for k, s in cores.items() if len(s) == n]  # the last core (fewer columns) may sample differently
        t0 = min(s[0][0] for s in cores.values())
        ops.append(max(s[-1][0] - t0 for s in cores.values()) / MHZ)
        for j in range(1, int(n)):
            ts = {k: (cores[k][j][0] - t0) / MHZ for k in full}
            med = statistics.median(ts.values())
            spreads.append(max(ts.values()) - min(ts.values()))
            last[max(ts, key=ts.get)] += 1
            for k, t in ts.items():
                lags[k].append(t - med)
        for k in full:
            s = cores[k]
            paces[k].extend((b[0] - a[0]) / MHZ for a, b in zip(s[1:], s[2:]))
    nblocks = sum(last.values())
    added = [k for k in full if ref_cores is not None and k not in ref_cores]
    mean_lag = {k: statistics.mean(v) for k, v in lags.items()}
    slow = sorted(mean_lag, key=mean_lag.get, reverse=True)[:5]
    res = {
        "cell": cell.name,
        "calls": len(calls),
        "cores": len(calls[0]),
        "samples_per_core": int(statistics.median(len(s) for s in calls[0].values())),
        "op_us_from_samples": statistics.mean(ops),
        "spread_median_us": statistics.median(spreads),
        "spread_mean_us": statistics.mean(spreads),
        "pace_median_us": statistics.median(p for v in paces.values() for p in v),
        "pace_per_core_max_over_min": max(statistics.median(v) for v in paces.values())
        / min(statistics.median(v) for v in paces.values()),
        "slowest_cores_mean_lag_us": {f"{k[0]},{k[1]}": round(mean_lag[k], 2) for k in slow},
        "last_place_top": {f"{k[0]},{k[1]}": round(c / nblocks, 3) for k, c in last.most_common(5)},
        "added_cores": [f"{k[0]},{k[1]}" for k in added],
        "added_last_share": sum(last[k] for k in added) / nblocks if added else None,
        "added_mean_lag_us": {f"{k[0]},{k[1]}": round(mean_lag[k], 2) for k in added},
    }
    return res, set(calls[0])


if __name__ == "__main__":
    ref = None
    for i, c in enumerate(sys.argv[1:]):
        res, cores = analyze(Path(c), ref)
        if i == 0:
            ref = cores
        for k, v in res.items():
            print(f"  {k}: {round(v, 3) if isinstance(v, float) else v}")
        print()
