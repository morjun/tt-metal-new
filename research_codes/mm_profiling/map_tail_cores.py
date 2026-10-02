#!/usr/bin/env python3
"""Per-core own read time of the drafter-tail matmul, on the physical chip layout.

Reads the zone profiles from test_lmhead_tail_zones.py (generated/lmhead_l1_profile/tail_zones/w*/)
and writes per_core.json: for each width and core (physical NoC x, y as the profiler reports
them), the median over the profiled calls of
  own read  = in1 reader kernel - waiting for an in1 slot - output-write zones  (issue + barrier)
  barrier   = the DRAM read barrier sum zone
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_tail_zones import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/tail_zones"


def per_core(width):
    freq, recs = load(ROOT / f"w{width}/.logs/profile_log_device.csv")
    runs = sorted({r["run"] for r in recs if r["zone"] == "IN1_RD_BARRIER"})[1:]
    acc = defaultdict(lambda: defaultdict(list))
    for run in runs:
        starts, wr, m = {}, defaultdict(float), defaultdict(dict)
        for r in recs:
            if r["run"] != run or r["risc"] != "BRISC":
                continue
            key = (r["core"], r["zone"])
            if r["type"] == "ZONE_START":
                starts[key] = r["t"]
            elif r["type"] == "ZONE_END" and key in starts:
                d = (r["t"] - starts.pop(key)) / freq
                if r["zone"] == "BRISC-KERNEL":
                    m[r["core"]]["kernel"] = d
                elif r["zone"] == "WR_PASS":
                    wr[r["core"]] += d
            elif r["type"] == "ZONE_TOTAL":
                m[r["core"]][r["zone"]] = r["data"] / freq
        for c, v in m.items():
            acc[c]["own_read"].append(v["kernel"] - v.get("IN1_SLOT_WAIT", 0) - wr[c])
            acc[c]["barrier"].append(v.get("IN1_RD_BARRIER", 0))
            acc[c]["slot_wait"].append(v.get("IN1_SLOT_WAIT", 0))
    return {f"{c[0]},{c[1]}": {k: statistics.median(v) for k, v in d.items()} for c, d in acc.items()}


def main():
    out = {w: per_core(w) for w in (1, 2, 4, 8, 16, 32, 64)}
    (ROOT / "per_core.json").write_text(json.dumps(out, indent=1) + "\n")
    xs = sorted({int(k.split(",")[0]) for k in out[8]})
    ys = sorted({int(k.split(",")[1]) for k in out[8]})
    for w, cores in out.items():
        vals = sorted(v["own_read"] for v in cores.values())
        print(
            f"\nwidth {w}: own read per core (us); physical NoC x across, y down; "
            f"min {vals[0]:.0f}  median {statistics.median(vals):.0f}  max {vals[-1]:.0f}"
        )
        print("   y\\x " + "".join(f"{x:>6}" for x in xs))
        for y in ys:
            print(
                f"   {y:>3} " + "".join(f"{cores.get(f'{x},{y}', {}).get('own_read', float('nan')):6.0f}" for x in xs)
            )


if __name__ == "__main__":
    main()
