#!/usr/bin/env python3
"""Per-op kernel-time difference of traced `time` cells against one baseline cell (gemma4-12b record §9.2).

Each cell is a run_traced_opprof_arms.py output directory (ops_perf.csv). Ops are labelled with
analyze_resource_profile.label_session and summed per (label, layer) over the 3 steps, mean over replays. Prints,
per cell, the iteration-span difference and every (label, layer) whose time moved by more than --min-us.

    python compare_opprof_cells.py <root> [--baseline 00_time_B+logitsl1] [--min-us 1.5]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_resource_profile import load, label_session

LAYER_TYPES = ["sliding", "sliding", "sliding", "global"]


def cell_times(path):
    per_label, spans = defaultdict(list), []
    for ops in load(path / "ops_perf.csv").values():
        acc = defaultdict(float)
        for o, (_, name, layer) in zip(ops, label_session(ops, LAYER_TYPES)):
            acc[(name, layer)] += o["dur"]
        for k, v in acc.items():
            per_label[k].append(v)
        spans.append(ops[0]["dur"] + sum(o["dur"] + o["gap"] for o in ops[1:]))
    return {k: statistics.mean(v) for k, v in per_label.items()}, statistics.mean(spans)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--baseline", default="00_time_B+logitsl1")
    p.add_argument("--min-us", type=float, default=1.5)
    args = p.parse_args()
    cells = {d.name: cell_times(d) for d in sorted(args.root.iterdir()) if (d / "ops_perf.csv").exists()}
    base, base_span = cells[args.baseline]
    out = {}
    for name, (times, span) in cells.items():
        if name == args.baseline:
            continue
        moved = []
        for key in sorted(set(times) | set(base), key=str):
            d = times.get(key, 0.0) - base.get(key, 0.0)
            if abs(d) > args.min_us:
                moved.append(
                    {
                        "label": key[0],
                        "layer": key[1],
                        "base_us": base.get(key, 0.0),
                        "arm_us": times.get(key, 0.0),
                        "delta_us": d,
                    }
                )
        out[name] = {"span_delta_us": span - base_span, "moved": moved}
        print(f"{name}: span {span - base_span:+.1f} us per 3-step iteration")
        for m in sorted(moved, key=lambda m: m["delta_us"]):
            layer = "" if m["layer"] is None else f" L{m['layer']}"
            print(f"    {m['delta_us']:+8.1f}  {m['label']}{layer}  ({m['base_us']:.1f} -> {m['arm_us']:.1f})")
    (args.root / "opprof_vs_baseline.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
