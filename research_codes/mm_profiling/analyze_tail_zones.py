#!/usr/bin/env python3
"""Per-engine time split of the drafter-tail matmul from test_lmhead_tail_zones.py profiles.

    python research_codes/mm_profiling/analyze_tail_zones.py generated/lmhead_l1_profile/tail_zones/w{1,2,4,8,16,32,64}

For each width directory: the matmul runs are the run host IDs that carry the in1 reader's
zones; the first is the warm-up and is dropped. Per profiled call and core it takes each
RISC's kernel duration (KERNEL zone start->end), the two accumulated SUM zones per RISC
(ZONE_TOTAL rows, value in cycles), and the per-pass WR_PASS zones. It prints the median over
the 108 "plain" cores (not the in0 sender at the first core, not the last core, which owns
fewer columns) and the max over all cores, in us, and writes summary.json in each directory.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SUMS = {
    "BRISC": ("IN1_RD_BARRIER", "IN1_SLOT_WAIT"),
    "NCRISC": ("IN0_RX_WAIT", "IN0_RX_SLOT_WAIT", "IN0_SND_DRAM", "IN0_SND_RX_READY"),
    # Compute: the firmware's own sum zones (llk_io_unpack.h / llk_io_pack.h), which own both
    # TRISC sum slots: every unpack cb_wait_front, and every pack cb_reserve_back.
    "TRISC_0": ("CB-COMPUTE-WAIT-FRONT",),
    "TRISC_2": ("CB-COMPUTE-RESERVE-BACK",),
}


def load(path):
    rows = list(csv.reader(open(path)))
    freq_mhz = float(rows[0][1].split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    header = [h.strip() for h in rows[1]]
    col = {h: i for i, h in enumerate(header)}
    recs = []
    for r in rows[2:]:
        if len(r) < len(header):
            continue
        recs.append(
            {
                "core": (int(r[col["core_x"]]), int(r[col["core_y"]])),
                "risc": r[col["RISC processor type"]].strip(),
                "t": int(r[col["time[cycles since reset]"]]),
                "data": int(r[col["data"]] or 0),
                "run": int(r[col["run host ID"]]),
                "zone": r[col["zone name"]].strip(),
                "type": r[col["type"]].strip(),
            }
        )
    return freq_mhz, recs


def analyze(d):
    freq, recs = load(Path(d) / ".logs/profile_log_device.csv")
    us = lambda cyc: cyc / freq  # noqa: E731
    runs = sorted({r["run"] for r in recs if r["zone"] == "IN1_RD_BARRIER"})
    profiled = runs[1:]  # drop the warm-up call
    per = defaultdict(lambda: defaultdict(dict))  # run -> core -> metric -> us
    for run in profiled:
        rr = [r for r in recs if r["run"] == run]
        starts = {}
        wr = defaultdict(list)
        for r in rr:
            key = (r["core"], r["risc"], r["zone"])
            if r["type"] == "ZONE_START":
                starts[key] = r["t"]
            elif r["type"] == "ZONE_END" and key in starts:
                dur = us(r["t"] - starts.pop(key))
                if r["zone"].endswith("-KERNEL"):
                    per[run][r["core"]][f'{r["risc"]}:kernel'] = dur
                elif r["zone"] == "WR_PASS":
                    wr[r["core"]].append(dur)
            elif r["type"] == "ZONE_TOTAL":
                per[run][r["core"]][f'{r["risc"]}:{r["zone"]}'] = us(r["data"])
        for core, v in wr.items():
            per[run][core]["BRISC:WR_PASS"] = sum(v)
            per[run][core]["BRISC:WR_PASS_count"] = len(v)
    cores = sorted({c for run in per for c in per[run]})
    # The in0 sender is the core with sender zones; the last core owns fewer columns.
    sender = {c for run in per for c, m in per[run].items() if "NCRISC:IN0_SND_RX_READY" in m}
    last = max(cores, key=lambda c: (c[1], c[0]))
    plain = [c for c in cores if c not in sender and c != last]
    metrics = sorted({m for run in per for c in per[run] for m in per[run][c]})
    out = {
        "dir": str(d),
        "freq_mhz": freq,
        "profiled_calls": len(profiled),
        "plain_cores": len(plain),
        "sender": sorted(sender),
        "last_core": last,
        "metrics": {},
    }
    for m in metrics:
        vals = [per[run][c][m] for run in per for c in plain if m in per[run][c]]
        allv = [per[run][c][m] for run in per for c in cores if m in per[run][c]]
        if not allv:
            continue
        out["metrics"][m] = {
            "median_plain": statistics.median(vals) if vals else None,
            "max_all": max(allv),
            "min_all": min(allv),
        }
    # Sender core values, separately.
    out["sender_metrics"] = {
        m: statistics.median([per[run][c][m] for run in per for c in sender if m in per[run][c]])
        for m in metrics
        if any(m in per[run][c] for run in per for c in sender)
    }
    (Path(d) / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    for d in sys.argv[1:]:
        o = analyze(d)
        print(f'== {o["dir"]}  ({o["profiled_calls"]} calls, {o["plain_cores"]} plain cores, {o["freq_mhz"]:.0f} MHz)')
        for m, v in o["metrics"].items():
            med = v["median_plain"]
            print(f'  {m:28s} median {med if med is None else round(med, 2)!s:>10}  max {v["max_all"]:10.2f}')
        for m, v in o["sender_metrics"].items():
            print(f"  [sender] {m:19s} {v:10.2f}")


if __name__ == "__main__":
    main()
