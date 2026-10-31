#!/usr/bin/env python3
"""Per-op times by position in the drafter step, for comparing layouts (gemma4-12b record §9.4, pin-value standard).

Reads time-mode cells of run_traced_opprof_arms.py (ops_perf.csv, trace replays only). Each op is keyed by its position:
matmuls as ``L<layer> <role>`` (qkv, o, gate, up, down), ``pre_proj``, or ``head/post mm<index>``; other ops as
``L<layer> <op>#<n>`` (the n-th such op after that layer's first matmul), ``head <op>#<n>``; ``gap:<key>`` is the op-to-op
gap before it. Values are µs per 3-step draft, mean over replays. Prints every key whose difference between any pair of
cells (first cell minus each other) is at least --min µs.

    python research_codes/mm_profiling/analyze_p4_perop.py <cell dir> <cell dir> [...] [--min 1.5]
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_resource_profile import load  # noqa: E402

ROLES = ["qkv", "o", "gate", "up", "down"]


def perop(path):
    tot = defaultdict(list)
    for ops in load(path / "ops_perf.csv").values():
        acc, step = defaultdict(float), []
        for o in ops:
            step.append(o)
            if o["op"] != "ArgMax":
                continue
            k, seen = 0, defaultdict(int)
            for x in step:
                if x["op"] == "Matmul":
                    key = (
                        "pre_proj"
                        if k == 0
                        else f"L{(k - 1) // 5} {ROLES[(k - 1) % 5]}"
                        if k <= 20
                        else f"head/post mm{k}"
                    )
                    k += 1
                else:
                    where = f"L{(k - 1) // 5}" if 1 <= k <= 20 else "pre" if k == 0 else "head"
                    seen[(where, x["op"])] += 1
                    key = f"{where} {x['op']}#{seen[(where, x['op'])]}"
                acc[key] += x["dur"]
                if x is not step[0]:
                    acc["gap:" + key] += x["gap"]
            step = []
        for key, v in acc.items():
            tot[key].append(v)
    return {key: statistics.mean(v) for key, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", type=Path)
    ap.add_argument("--min", type=float, default=1.5)
    args = ap.parse_args()
    v = [perop(c) for c in args.cells]
    keys = sorted(set().union(*v))
    print(
        "key".ljust(34)
        + "".join(f"{i:>9d}" for i in range(len(v)))
        + "".join(f"  0-{i}".rjust(9) for i in range(1, len(v)))
    )
    for key in keys:
        d = [v[0].get(key, 0) - x.get(key, 0) for x in v[1:]]
        if max(abs(y) for y in d) >= args.min:
            print(key[:34].ljust(34) + "".join(f"{x.get(key, 0):9.1f}" for x in v) + "".join(f"{y:9.1f}" for y in d))
    for i, c in enumerate(args.cells):
        print(f"{i}: {c}")


if __name__ == "__main__":
    main()
