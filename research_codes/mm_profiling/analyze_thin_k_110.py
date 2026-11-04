#!/usr/bin/env python3
"""S2 of gemma4-12b record §9.5: per-op time of the spread thin-K pin vs the 86-core prefix pin.

Reads the traced time-mode cells of run_traced_opprof_arms.py (<root>/<index>_time_<arm>/ops_perf.csv). Every value is
µs per 3-step draft (one replay), as mean ± SE over the replays of an arm (cells of one arm pooled; arms are different
processes, so a difference's SE = sqrt(SE_a^2 + SE_b^2)). Prints the drafted tokens, the span, and every matmul
instance / op type whose difference between any two arms is >= 0.5 µs.

    python research_codes/mm_profiling/analyze_thin_k_110.py generated/lmhead_l1_profile/thin_k_110/opprof_s2
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_m12 import instances, steps  # noqa: E402
from analyze_resource_profile import load  # noqa: E402


def stats(xs):
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")
    return m, se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    data = defaultdict(list)  # arm -> [{label: µs per draft}]
    tokens = defaultdict(set)
    for d in sorted(x for x in args.root.iterdir() if (x / "ops_perf.csv").exists()):
        arm = re.match(r"\d+_time_(.*)", d.name).group(1)
        tokens[arm].update(re.findall(r"K=3 output: token=\[(\d+)\] hidden_sha256=(\w+)", (d / "run.log").read_text()))
        for ops in load(d / "ops_perf.csv").values():
            rec = defaultdict(float)
            for st in steps(ops):
                for k, v in instances(st).items():
                    rec[k] += v
            rec["span"] = ops[0]["dur"] + sum(o["dur"] + o["gap"] for o in ops[1:])
            data[arm].append(rec)
    arms = list(data)
    out = {"tokens": {a: sorted(t) for a, t in tokens.items()}, "replays": {a: len(v) for a, v in data.items()}}
    print("tokens:", out["tokens"])
    print("replays:", out["replays"])
    keys = sorted({k for v in data.values() for r in v for k in r})
    table = {k: {a: stats([r.get(k, 0.0) for r in data[a]]) for a in arms} for k in keys}
    out["table"] = table
    print("\n| item | " + " | ".join(arms) + " |\n|---|" + "--:|" * len(arms))
    for k in ["span"] + [k for k in keys if k != "span"]:
        means = [table[k][a][0] for a in arms]
        if k != "span" and max(means) - min(means) < 0.5:
            continue
        print(f"| {k} | " + " | ".join(f"{m:.1f} ± {s:.1f}" for m, s in (table[k][a] for a in arms)) + " |")
    (args.root / "s2.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
