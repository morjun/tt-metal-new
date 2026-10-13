#!/usr/bin/env python3
"""Join an eager Tracy device profile to the captured iteration graph (gemma4-12b §5.5.14, reload hiding).

test_assistant_standalone_l1.py::test_lm_head_l1_address_capture runs the fused K=3 body once eagerly, then
captures it in a trace. Tracy's merge stops on the captured ops (no device data), so this joins
cpp_device_perf_report.csv to tracy_ops_data.csv by global call count and takes the N consecutive device rows,
N = the device ops of capture/iteration.graph.json (the eager iteration just before the capture), whose names
match the graph's best: the last N, unless ops ran after the capture (the DRISC reload's stop trigger at close).

    python align_opprof_iteration.py <dir with cpp_device_perf_report.csv, tracy_ops_data.csv, capture/>

Writes <dir>/aligned_iteration.json: per op, graph node, name, call, kernel ns, cores, kernel start / end cycle.
"""

import csv
import json
import re
import sys
from pathlib import Path


def device_ops(graph):
    """Outermost device operations of a ttnn graph capture, in order."""
    stack, seq = [], []
    for i, n in enumerate(graph):
        if n["node_type"] == "function_start":
            name = n["params"].get("name", "")
            if name.endswith("DeviceOperation") and not any(s.endswith("DeviceOperation") for s in stack):
                seq.append((i, name))
            stack.append(name)
        elif n["node_type"] == "function_end" and stack:
            stack.pop()
    return seq


def main(out):
    names = {}
    for line in open(out / "tracy_ops_data.csv", errors="ignore"):
        m = re.search(r'TT_DNN_DEVICE_OP: "(\w+)", \d+, \d+, (\w+), (\d+)', line)
        if m:
            names[int(m.group(3))] = m.group(1)

    def num(r, k):
        return int(r[k]) if r.get(k) else 0

    rows = sorted(
        (
            num(r, "GLOBAL CALL COUNT"),
            names.get(num(r, "GLOBAL CALL COUNT"), "?"),
            num(r, "DEVICE KERNEL DURATION [ns]"),
            num(r, "CORE COUNT"),
            num(r, "DEVICE KERNEL START CYCLE"),
            num(r, "DEVICE KERNEL END CYCLE"),
        )
        for r in csv.DictReader(open(out / "cpp_device_perf_report.csv"))
    )
    seq = device_ops(json.load(open(out / "capture/iteration.graph.json")))

    def miss(o):
        return sum(1 for (_, g), row in zip(seq, rows[o : o + len(seq)]) if g != row[1])

    offset = min(range(len(rows) - len(seq), -1, -1), key=miss)  # ties: the latest
    tail = rows[offset : offset + len(seq)]
    mismatches = miss(offset)
    aligned = [
        {"graph_node": gi, "op": g, "call": c, "dev_name": dn, "kernel_ns": d, "cores": cc, "start": s, "end": e}
        for (gi, g), (c, dn, d, cc, s, e) in zip(seq, tail)
    ]
    (out / "aligned_iteration.json").write_text(json.dumps(aligned, indent=1) + "\n")
    print(
        f"{out}: {len(rows)} device rows, {len(seq)} graph ops at rows [{offset}, {offset + len(seq)}), "
        f"{mismatches} name mismatches, {len(rows) - offset - len(seq)} rows after"
    )
    return mismatches


if __name__ == "__main__":
    sys.exit(1 if main(Path(sys.argv[1])) else 0)
