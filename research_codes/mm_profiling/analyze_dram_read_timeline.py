#!/usr/bin/env python3
"""Instantaneous DRAM read rate inside each traced matmul of a noc cell (gemma4-12b record §9.2).

A noc cell of run_traced_opprof_arms.py has one profiled replay (ops_perf.csv) and one tt-npe file per op
(npe_viz/<Op>_ID<n>_traceID<t>.npeviz.zst, n = GLOBAL CALL COUNT // 1024). Each file lists every NoC transfer of
the op with its MEASURED issue cycle (start_cycle, from the device NoC trace) and size; end_cycle is tt-npe's
congestion model and is not used. A DRAM read = a READ whose source is never the destination of a READ.

Per op: DRAM read bytes binned by issue time. Each core's reader waits on a barrier per block, so over a bin longer
than one read round trip the issue rate is the delivered rate. Reported: the read window (first to last DRAM read
issue), its mean rate, the rate in each tenth of the window, and the time to first reach 90% of the steady rate
(median of tenths 3-8). Per core: the time of its last DRAM read issue; "GB/s while all cores read" is the rate up to
the first core's last read, and the tail is what is issued after it. Ops are labelled with analyze_resource_profile.label_session and averaged over the steps and
layers sharing a label.

Trace guard: per op and core, the profiler pushes inside the core's kernels and the overrun (last DRAM read issue minus
that core's own kernel end). An op is INVALID when any call's max overrun exceeds --overrun-threshold-us (default 10,
chosen from the distribution over all cells: backbone -4..+6 us, DRAM-fed lm_head +20..+111 us). Invalid ops' rates
are reported but must not be used.

    python analyze_dram_read_timeline.py <noc_cell_dir> [--bin-us 1.0] -> <cell>/dram_timeline.{json,md}
"""

import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from analyze_resource_profile import label_session, load

LAYER_TYPES = ["sliding", "sliding", "sliding", "global"]
MHZ = 1350.0  # P150 AICLK under load; checked against FW cycles / ns in ops_perf


def load_npeviz(npeviz):
    return json.loads(subprocess.run(["zstd", "-dc", str(npeviz)], capture_output=True, check=True).stdout)


def dram_reads(d):
    reads = [t for t in d["noc_transfers"] if t["noc_event_type"] == "READ"]
    dsts = {tuple(x) for t in reads for x in t["dst"]}
    dram = {tuple(t["src"]) for t in reads} - dsts
    return [(t["start_cycle"] / MHZ, t["total_bytes"], tuple(t["dst"][0])) for t in reads if tuple(t["src"]) in dram]


def trace_guard(d, reads):
    """Per core (dev, row, col): (profiler pushes inside its kernels, overrun us).

    overrun = the core's last DRAM read issue minus the end of its own kernel (latest of BRISC-KERNEL and
    NCRISC-KERNEL, from the same trace). A read cannot be issued after its kernel ends, so a positive overrun means
    the trace's timestamps for that core are not trustworthy. Pushes = PROFILER-NOC-QUICK-PUSH zones nested in the
    core's kernels (mid-kernel flushes of the profiler buffer), reported to check, not assume, a link to overruns.
    """
    kernel_end, pushes = defaultdict(float), defaultdict(int)
    for z in d["zones"]:
        core = tuple(z["core"])
        for k in z["zones"]:
            if "-KERNEL[" in k["id"]:
                kernel_end[core] = max(kernel_end[core], k["end"] / MHZ)
                pushes[core] += sum(1 for x in k["zones"] if x["id"].startswith("PROFILER-NOC-QUICK-PUSH"))
    last = defaultdict(lambda: float("-inf"))
    for t, _, core in reads:
        last[core] = max(last[core], t)
    return {c: (pushes[c], last[c] - kernel_end[c]) for c in last if c in kernel_end}


def guard_summary(g):
    """Distribution over cores of the per-core overrun and push count for one op call."""
    ov = sorted(v for _, v in g.values())
    pu = sorted(p for p, _ in g.values())
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]
    return {
        "cores": len(ov),
        "overrun_min_us": ov[0],
        "overrun_median_us": q(ov, 0.5),
        "overrun_p90_us": q(ov, 0.9),
        "overrun_max_us": ov[-1],
        "cores_overrun_gt0": sum(1 for v in ov if v > 0),
        "pushes_median": q(pu, 0.5),
        "pushes_max": pu[-1],
    }


def profile(reads, bin_us):
    t0 = min(t for t, _, _ in reads)
    t1 = max(t for t, _, _ in reads)
    total = sum(b for _, b, _ in reads)
    span = max(t1 - t0, bin_us)
    tenths = [0.0] * 10
    for t, b, _ in reads:
        tenths[min(9, int((t - t0) / span * 10))] += b
    tenths = [b / (span / 10) / 1e3 for b in tenths]  # GB/s
    bins = defaultdict(float)
    for t, b, _ in reads:
        bins[int((t - t0) / bin_us)] += b
    series = [bins[i] / bin_us / 1e3 for i in range(int(span / bin_us) + 1)]
    steady = statistics.median(tenths[2:8])
    ramp = next((i * bin_us for i, r in enumerate(series) if r >= 0.9 * steady), None)
    first_per_core = defaultdict(lambda: float("inf"))
    last_per_core = defaultdict(float)
    for t, _, core in reads:
        first_per_core[core] = min(first_per_core[core], t)
        last_per_core[core] = max(last_per_core[core], t)
    ends = sorted(t - t0 for t in last_per_core.values())
    # rate while every core is still reading vs after the first core has issued its last read
    t_all = ends[0] + t0
    all_b = sum(b for t, b, _ in reads if t <= t_all)
    return {
        "core_end_min_us": ends[0],
        "core_end_median_us": statistics.median(ends),
        "core_end_max_us": ends[-1],
        "all_cores_gbs": all_b / max(t_all - t0, bin_us) / 1e3,
        "tail_us": ends[-1] - ends[0],
        "tail_mb": (total - all_b) / 1e6,
        "t_first_us": t0,
        "window_us": t1 - t0,
        "mb": total / 1e6,
        "mean_gbs": total / span / 1e3,
        "tenths_gbs": tenths,
        "steady_gbs": steady,
        "ramp_us": ramp,
        "core_start_skew_us": max(first_per_core.values()) - min(first_per_core.values()),
        "cores": len(first_per_core),
        "series_gbs": series,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cell", type=Path)
    p.add_argument("--bin-us", type=float, default=1.0)
    # From the distribution over every noc cell to date (record §9.2, trace validity): every backbone matmul's per-op
    # max overrun is -4.0 .. +6.0 us; every DRAM-fed lm_head call is +20.1 .. +111.4 us. 10 us sits between them.
    p.add_argument("--overrun-threshold-us", type=float, default=10.0)
    args = p.parse_args()
    (session,) = load(args.cell / "ops_perf.csv").values()
    files = {
        int(f.name.split("_ID")[1].split("_")[0]): f
        for f in (args.cell / "npe_viz").glob("Matmul*_traceID*.npeviz.zst")
        if "_split_" not in f.name  # tt-npe also writes long ops in _split_<k> chunks; the unsplit file is complete
    }
    per_label = defaultdict(list)
    for o, (_, name, layer) in zip(session, label_session(session, LAYER_TYPES)):
        if o["op"] != "Matmul":
            continue
        f = files.get(int(o["row"]["GLOBAL CALL COUNT"]) // 1024)
        if not f:
            continue
        d = load_npeviz(f)
        reads = dram_reads(d)
        if reads:
            guard = guard_summary(trace_guard(d, reads))
            per_label[name].append(
                {"layer": layer, "kernel_us": o["dur"], **profile(reads, args.bin_us), "guard": guard}
            )
    out = {}
    for name, calls in per_label.items():
        mean = lambda k: statistics.mean(c[k] for c in calls)
        out[name] = {
            "calls": len(calls),
            **{
                k: mean(k)
                for k in (
                    "kernel_us",
                    "t_first_us",
                    "window_us",
                    "mb",
                    "mean_gbs",
                    "steady_gbs",
                    "core_start_skew_us",
                    "cores",
                    "core_end_min_us",
                    "core_end_median_us",
                    "core_end_max_us",
                    "all_cores_gbs",
                    "tail_us",
                    "tail_mb",
                )
            },
            "ramp_us": statistics.mean(c["ramp_us"] for c in calls if c["ramp_us"] is not None),
            "tenths_gbs": [statistics.mean(c["tenths_gbs"][i] for c in calls) for i in range(10)],
            "series_gbs": calls[0]["series_gbs"],
            # per-op max overrun over the calls; invalid = any call above the threshold
            "overrun_max_us": max(c["guard"]["overrun_max_us"] for c in calls),
            "overrun_median_us": statistics.median(c["guard"]["overrun_median_us"] for c in calls),
            "cores_overrun_gt0": max(c["guard"]["cores_overrun_gt0"] for c in calls),
            "pushes_max": max(c["guard"]["pushes_max"] for c in calls),
            "valid": all(c["guard"]["overrun_max_us"] <= args.overrun_threshold_us for c in calls),
        }
    (args.cell / "dram_timeline.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [
        f"DRAM read rate inside each matmul, {args.cell.name} (noc mode, issue-time bins of {args.bin_us} us)",
        "",
        f"Trace guard: overrun = a core's last DRAM read minus its own kernel end; an op is INVALID if any call's max"
        f" overrun > {args.overrun_threshold_us:g} us. Invalid ops' rates are shown but must not be used.",
        "",
        "| op | calls | cores | kernel us (noc mode) | first read us | read window us | MB | mean GB/s | steady GB/s | ramp to 90% us"
        " | GB/s while all cores read | cores' last read: first / median / last us | tail us (MB) | tenths of the window, GB/s"
        " | guard: max / median overrun us, cores > 0, pushes/core max |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|",
    ]
    for name, r in sorted(out.items(), key=lambda x: -x[1]["mb"]):
        lines.append(
            f"| {name} | {r['calls']} | {r['cores']:.0f} | {r['kernel_us']:.1f} | {r['t_first_us']:.2f} | {r['window_us']:.1f}"
            f" | {r['mb']:.2f} | {r['mean_gbs']:.0f} | {r['steady_gbs']:.0f} | {r['ramp_us']:.1f} | {r['all_cores_gbs']:.0f}"
            f" | {r['core_end_min_us']:.1f} / {r['core_end_median_us']:.1f} / {r['core_end_max_us']:.1f}"
            f" | {r['tail_us']:.1f} ({r['tail_mb']:.2f}) | "
            + " ".join(f"{x:.0f}" for x in r["tenths_gbs"])
            + f" | {'' if r['valid'] else '**INVALID** '}{r['overrun_max_us']:.1f} / {r['overrun_median_us']:.1f},"
            f" {r['cores_overrun_gt0']}, {r['pushes_max']} |"
        )
    (args.cell / "dram_timeline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
