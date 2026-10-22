#!/usr/bin/env python3
"""Delivered DRAM read rate per op from the in1 reader's NIU counter samples (gemma4-12b record §9.2, item 6b).

Input: a time-mode cell of run_traced_opprof_arms.py run with the "+niusample" arm suffix
(MM_IN1_NIU_SAMPLE in reader_bmm_tile_layout_in1_sender_writer_padding.cpp): ops_perf.csv and niu_samples.csv (the
device-log rows of DeviceTimestampedData("IN1_NIU_RD_WORDS", NIU_MST_RD_DATA_WORD_RECEIVED)).

Each 1D-mcast matmul core samples its NIU master "read data words received" counter on the in1 reader's NoC once
before its first block and after every block's read barrier. Unlike the NoC trace (issue times), a sample is taken
after the block's data has ARRIVED, so consecutive samples give the delivered bytes per core over time:
  - word size: 64 B, measured (the DRAM-side probe item6/test_drisc_niu_probe.py counts exactly the weight bytes in
    64 B words). Coverage = counted bytes / the op's DRAM in1 bytes is reported: below 100% the core's profiler
    buffer filled before the op ended (lm_head: 125 samples per core, ~49% of its reads), so that op's rates cover
    only its first part
  - per core: delivered GB/s = its bytes / (last sample - first sample)
  - per op: delivered window (first core's first sample to last core's last sample), its mean rate, the tail after
    the first core's last sample, and the rate while all cores deliver (up to that sample) as lower / upper bounds:
    sample intervals that end by then / that start before then. With per-block samples only (4 per core on thin-K)
    the bounds are far apart; "+niusample<ticks>" also samples every <ticks> wall-clock ticks and closes them
  - peak aggregate rate: all cores' delivered bytes in BIN_US bins (each interval's bytes spread evenly over it),
    the highest bin; meaningful only with timed samples
The counter counts every read on that NoC at that core; the in1 reader is the only NoC-0 reader of these kernels
(in0 is on NoC 1 unless the NoC swap is set).

    python item6/analyze_niu_samples.py <cell> -> <cell>/niu_delivered.{json,md}
"""

import argparse
import bisect
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_resource_profile import label_session, load, tensor_bytes  # noqa: E402

LAYER_TYPES = ["sliding", "sliding", "sliding", "global"]
MHZ = 1350.0
WORD_BYTES = 64
BIN_US = 2.0


def read_samples(path):
    """[(cycle, (core_x, core_y), counter)] from the device-log rows of IN1_NIU_RD_WORDS."""
    out = []
    with open(path) as f:
        next(f)  # "ARCH: ..., CHIP_FREQ[MHz]: ..." line
        for row in csv.DictReader(f, skipinitialspace=True):
            if (row.get("zone name") or "").strip() == "IN1_NIU_RD_WORDS":
                out.append(
                    (int(row["time[cycles since reset]"]), (int(row["core_x"]), int(row["core_y"])), int(row["data"]))
                )
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cell", type=Path)
    args = p.parse_args()
    samples = read_samples(args.cell / "niu_samples.csv")
    ops = []  # (fw_start, fw_end, label, in1 DRAM bytes)
    for session in load(args.cell / "ops_perf.csv").values():
        for o, (_, name, layer) in zip(session, label_session(session, LAYER_TYPES)):
            if o["op"] == "Matmul":
                ops.append((o["fw_start"], o["fw_end"], name, tensor_bytes(o["row"], "INPUT_1")))
    ops.sort()
    starts = [o[0] for o in ops]
    per_call = defaultdict(lambda: defaultdict(list))  # op index -> core -> [(cycle, counter)]
    for t, core, words in samples:
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and t <= ops[i][1]:
            per_call[i][core].append((t, words))
    per_label = defaultdict(list)
    for i, cores in per_call.items():
        _, _, name, in1_bytes = ops[i]
        spans, totals = {}, {}
        for core, s in cores.items():
            if len(s) < 2:
                continue
            words = sum((b - a) % (1 << 32) for (_, a), (_, b) in zip(s, s[1:]))
            spans[core] = (s[0][0] / MHZ, s[-1][0] / MHZ)
            totals[core] = words
        tw = sum(totals.values())
        if not tw or not in1_bytes:
            continue  # L1-fed op (no DRAM in1) or no samples
        bpw = WORD_BYTES
        t0 = min(a for a, _ in spans.values())
        ends = sorted(b for _, b in spans.values())
        core_gbs = [totals[c] * bpw / max(spans[c][1] - spans[c][0], 1e-3) / 1e3 for c in totals]
        # bytes delivered by the first core's last sample, bounded by whole sample intervals; aggregate rate per bin
        t_all = ends[0]
        lo = hi = 0.0
        gaps = []
        bins = defaultdict(float)
        for core, s in cores.items():
            for (ta, a), (tb, b) in zip(s, s[1:]):
                ta, tb = ta / MHZ, tb / MHZ
                d = ((b - a) % (1 << 32)) * bpw
                gaps.append(tb - ta)
                if tb <= t_all:
                    lo += d
                    hi += d
                elif ta < t_all:
                    hi += d
                k0, k1 = int((ta - t0) // BIN_US), int((tb - t0) // BIN_US)
                for k in range(k0, k1 + 1):
                    overlap = min(tb, t0 + (k + 1) * BIN_US) - max(ta, t0 + k * BIN_US)
                    if overlap > 0:
                        bins[k] += d * overlap / max(tb - ta, 1e-6)
        per_label[name].append(
            {
                "cores": len(totals),
                "coverage": tw * bpw / in1_bytes,
                "window_us": ends[-1] - t0,
                "mean_gbs": tw * bpw / (ends[-1] - t0) / 1e3,
                "all_cores_gbs_lo": lo / max(t_all - t0, 1e-3) / 1e3,
                "all_cores_gbs_hi": hi / max(t_all - t0, 1e-3) / 1e3,
                "sample_gap_us_median": statistics.median(gaps),
                "peak_bin_gbs": max(bins.values()) / BIN_US / 1e3,
                "tail_us": ends[-1] - ends[0],
                "core_gbs_min": min(core_gbs),
                "core_gbs_median": statistics.median(core_gbs),
                "core_gbs_max": max(core_gbs),
            }
        )
    out = {
        name: {"calls": len(c), **{k: statistics.mean(x[k] for x in c) for k in c[0]}} for name, c in per_label.items()
    }
    (args.cell / "niu_delivered.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [
        f"Delivered DRAM read rate from NIU counter samples, {args.cell.name} (completion times, 64 B words; rates"
        " cover the sampled bytes only)",
        "",
        "| op | calls | cores | coverage % | median sample gap us | delivered window us | mean GB/s"
        " | GB/s while all cores deliver, lower / upper | tail us | peak 2-us aggregate GB/s"
        " | per-core GB/s min / median / max |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name, r in sorted(out.items(), key=lambda x: -x[1]["window_us"]):
        lines.append(
            f"| {name} | {r['calls']} | {r['cores']:.0f} | {100 * r['coverage']:.1f} | {r['sample_gap_us_median']:.2f}"
            f" | {r['window_us']:.1f} | {r['mean_gbs']:.0f} | {r['all_cores_gbs_lo']:.0f} / {r['all_cores_gbs_hi']:.0f}"
            f" | {r['tail_us']:.1f} | {r['peak_bin_gbs']:.0f}"
            f" | {r['core_gbs_min']:.2f} / {r['core_gbs_median']:.2f} / {r['core_gbs_max']:.2f} |"
        )
    (args.cell / "niu_delivered.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
