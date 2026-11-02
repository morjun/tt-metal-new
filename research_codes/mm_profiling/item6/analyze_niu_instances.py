#!/usr/bin/env python3
"""Per-instance NIU read windows (gemma4-12b record §9.2b, "Read windows", 2026-11-02).

analyze_niu_samples.py averages each op family over its layers (e.g. "up_proj (sliding)" = layers 0-2). This script
keeps every matmul instance (family, layer) apart and reports, per call and over calls:
  - NIU bytes: 64 B read-data words counted by each in1-reader core's NIU master counter between its first sample
    (taken at reader-kernel start, before the first block's reads are issued) and its last (after the last block's
    read barrier); this includes phantom reads of a partial last core. Tensor bytes: the in1 DRAM tensor (padded shape)
  - read window: first core's first sample -> last core's last sample (device cycles at 1,350 MHz)
  - op duration: DEVICE KERNEL DURATION of the same op (ops_perf.csv)
  - alignment: first sample - op FW start, op FW end - last sample (both should be >= 0 up to clock skew)
  - mean GB/s = NIU bytes / window
Statistics over calls: mean, SD, min, max.

    python item6/analyze_niu_instances.py <cell> [<cell> ...] [--only up_proj,lm_head] -> stdout + <cell>/niu_instances.json
"""

import argparse
import bisect
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_niu_samples import LAYER_TYPES, MHZ, WORD_BYTES, read_samples  # noqa: E402
from analyze_resource_profile import label_session, load, tensor_bytes  # noqa: E402


def instances(cell):
    samples = read_samples(cell / "niu_samples.csv")
    ops = []  # (fw_start, fw_end, label, in1 DRAM bytes, kernel duration us)
    for session in load(cell / "ops_perf.csv").values():
        for o, (_, name, layer) in zip(session, label_session(session, LAYER_TYPES)):
            if o["op"] == "Matmul":
                label = name if layer is None else f"{name} L{layer}"
                ops.append((o["fw_start"], o["fw_end"], label, tensor_bytes(o["row"], "INPUT_1"), o["dur"]))
    ops.sort()
    starts = [o[0] for o in ops]
    per_call = defaultdict(lambda: defaultdict(list))
    tol = 2.0 * MHZ
    for t, core, words in samples:
        i = bisect.bisect_right(starts, t + tol) - 1
        if i > 0 and t <= ops[i - 1][1]:
            i -= 1
        if i >= 0 and t <= ops[i][1] + tol:
            per_call[i][core].append((t, words))
    out = defaultdict(list)
    for i, cores in sorted(per_call.items()):
        fw0, fw1, label, in1_bytes, dur = ops[i]
        if not in1_bytes:
            continue
        cores = {c: s for c, s in cores.items() if len(s) >= 2}
        if not cores:
            continue
        words = sum((b - a) % (1 << 32) for s in cores.values() for (_, a), (_, b) in zip(s, s[1:]))
        first = min(s[0][0] for s in cores.values())
        last = max(s[-1][0] for s in cores.values())
        first_ends = min(s[-1][0] for s in cores.values())
        out[label].append(
            {
                "cores": len(cores),
                "samples_per_core": statistics.median(len(s) for s in cores.values()),
                "niu_bytes": words * WORD_BYTES,
                "tensor_bytes": in1_bytes,
                "window_us": (last - first) / MHZ,
                "op_us": dur,
                "start_off_us": (first - fw0) / MHZ,
                "end_off_us": (fw1 - last) / MHZ,
                "tail_us": (last - first_ends) / MHZ,
                "gbs": words * WORD_BYTES / ((last - first) / MHZ) / 1e3,
            }
        )
    return out


def stats(xs):
    return {
        "mean": statistics.mean(xs),
        "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cells", type=Path, nargs="+")
    p.add_argument("--only", default="", help="comma-separated substrings of instance labels to print")
    args = p.parse_args()
    only = [x for x in args.only.split(",") if x]
    for cell in args.cells:
        res = instances(cell)
        summary = {label: {"calls": len(c), **{k: stats([x[k] for x in c]) for k in c[0]}} for label, c in res.items()}
        (cell / "niu_instances.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"\n## {cell.name}")
        print(
            "| instance | calls | cores | samples/core | NIU MB | tensor MB | window us mean ± SD (min-max) | op us mean ± SD"
            " | window / op | GB/s mean ± SD | start off us (min) | end off us (min) | tail us |"
        )
        print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for label, s in sorted(summary.items(), key=lambda x: -x[1]["window_us"]["mean"]):
            if only and not any(o in label for o in only):
                continue
            w, o = s["window_us"], s["op_us"]
            print(
                f"| {label} | {s['calls']} | {s['cores']['mean']:.0f} | {s['samples_per_core']['mean']:.0f}"
                f" | {s['niu_bytes']['mean'] / 1e6:.3f} | {s['tensor_bytes']['mean'] / 1e6:.3f}"
                f" | {w['mean']:.2f} ± {w['sd']:.2f} ({w['min']:.2f}-{w['max']:.2f}) | {o['mean']:.2f} ± {o['sd']:.2f}"
                f" | {w['mean'] / o['mean']:.3f} | {s['gbs']['mean']:.0f} ± {s['gbs']['sd']:.0f}"
                f" | {s['start_off_us']['mean']:.2f} ({s['start_off_us']['min']:.2f})"
                f" | {s['end_off_us']['mean']:.2f} ({s['end_off_us']['min']:.2f}) | {s['tail_us']['mean']:.2f} |"
            )


if __name__ == "__main__":
    main()
