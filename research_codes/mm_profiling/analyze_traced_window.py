#!/usr/bin/env python3
"""Window ops of the traced fused K=3 drafter, per replay (gemma4-12b §5.5.14, reload hiding).

Reads the cpp_device_perf_report.csv of test_drisc_traced_opprof.py runs. Per trace replay and drafter step:
the ops from the drafter's global SDPA (the 32-core SdpaDecode) to the lm_head's L1-slice matmul, with kernel
durations, the 4 window matmuls (the 32/64-core matmuls in that span), and the reload op (Tensix arm) or the
trigger / wait ops (DRISC arm: 1-core GenericOps).

    python analyze_traced_window.py <run dir> [<run dir> ...]   -> <run dir>/traced_window.json + a summary
"""

import csv
import json
import statistics
import sys
from pathlib import Path

NS_PER_CYCLE = 1 / 1.35  # AICLK 1350 MHz


def replays(path):
    rows = [r for r in csv.DictReader(open(path)) if r["METAL TRACE REPLAY SESSION ID"]]
    by = {}
    for r in rows:
        by.setdefault(int(r["METAL TRACE REPLAY SESSION ID"]), []).append(
            {
                "op": r["OP NAME"].replace("DeviceOperation", ""),
                "cores": int(r["CORE COUNT"]),
                "us": int(r["DEVICE KERNEL DURATION [ns]"]) / 1e3,
                "start": int(r["DEVICE KERNEL START CYCLE"]),
                "end": int(r["DEVICE KERNEL END CYCLE"]),
            }
        )
    return {k: sorted(v, key=lambda o: o["start"]) for k, v in sorted(by.items())}


def steps(ops):
    sdpa = [i for i, o in enumerate(ops) if o["op"] == "SdpaDecode" and o["cores"] == 32]
    out = []
    for i in sdpa:
        # the lm_head L1 slice: the first 110-core matmul after the SDPA
        j = next(k for k in range(i + 1, len(ops)) if ops[k]["op"] == "Matmul" and ops[k]["cores"] == 110)
        win = ops[i + 1 : j]
        generic = [o for o in win if o["op"] == "GenericOp"]
        mm = [o for o in win if o["op"] == "Matmul"]
        out.append(
            {
                "sdpa_us": ops[i]["us"],
                "matmul_us": [o["us"] for o in mm],
                "generic": [(o["cores"], o["us"]) for o in generic],
                "other_us": sum(o["us"] for o in win if o["op"] not in ("Matmul", "GenericOp")),
                "span_us": (ops[j - 1]["end"] - ops[i]["end"]) * NS_PER_CYCLE / 1e3,
                "l1_slice_us": ops[j]["us"],
                "ops": [(o["op"], o["cores"], round(o["us"], 3)) for o in win],
            }
        )
    return out


def main(dirs):
    for d in map(Path, dirs):
        reps = replays(d / "cpp_device_perf_report.csv")
        res = {
            s: {"iteration_us": (ops[-1]["end"] - ops[0]["start"]) * NS_PER_CYCLE / 1e3, "steps": steps(ops)}
            for s, ops in reps.items()
        }
        (d / "traced_window.json").write_text(json.dumps(res, indent=1) + "\n")
        for s, r in res.items():
            print(f"{d.name} replay {s}: iteration {r['iteration_us']:.1f} us")
            for k, st in enumerate(r["steps"]):
                print(
                    f"  step {k}: matmuls {[round(x, 1) for x in st['matmul_us']]} = {sum(st['matmul_us']):.1f}, "
                    f"other {st['other_us']:.1f}, generic {[(c, round(u, 1)) for c, u in st['generic']]}, "
                    f"SDPA-end..last-window-op-end {st['span_us']:.1f}, L1 slice {st['l1_slice_us']:.1f}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
