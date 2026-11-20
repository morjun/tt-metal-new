"""Per-op DRAM read / write bytes of the standalone drafter on this branch, from recorded NoC traces (record §8).

Port of l1w analyze_dram_bytes_noc.py to run_opprof.py cells (<root>/<index>_noc_<arm>/: ops_perf.csv + npe_viz/).
Each npe_viz file (<OP CODE>_ID<GLOBAL CALL COUNT / 1024>_traceID1.npeviz.zst) lists every NoC transfer the op's
kernels issued in the traced replay (tt-npe frame [device, y, x] for both NoCs). Blackhole DRAM cores: x = 0 and 9.
  DRAM read  = READ* transfers whose source is a DRAM core.
  DRAM write = WRITE* transfers with a DRAM core among the destinations.
Ops are labelled per drafter step (a step ends at its ArgMax): matmuls by weight shape K x N and layer, SDPA by layer
type, every other op by op code and the matmul it follows. The SDPA ops are also broken down by transfer size.

    python analyze_noc_dram.py <cell> [<cell> ...] [--json out.json]
"""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

DRAM_X = {0, 9}
LAYER_TYPES = ["sliding", "sliding", "sliding", "global"]
DTYPE_BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "BFLOAT8_B": 1088 / 1024}


def tensor_bytes(r, p):
    if "DRAM" not in (r.get(f"{p}_MEMORY") or ""):
        return 0
    n = 1
    for d in "WZYX":
        m = re.match(r"(\d+)", r.get(f"{p}_{d}_PAD[LOGICAL]") or "")
        if not m:
            return 0
        n *= int(m.group(1))
    return n * DTYPE_BYTES.get(r[f"{p}_DATATYPE"], 2)


def shape(r, p):
    return "x".join(re.match(r"(\d+)", r.get(f"{p}_{d}_PAD[LOGICAL]") or "0").group(1) for d in "WZYX")


def transfers(path):
    j = json.loads(subprocess.run(["zstd", "-dc", str(path)], capture_output=True, check=True).stdout)
    for t in j["noc_transfers"]:
        ty = t["noc_event_type"]
        if ty.startswith("READ") and t["src"][2] in DRAM_X:
            yield "read", t["total_bytes"], tuple(t["dst"][0][1:])
        elif ty.startswith("WRITE") and any(d[2] in DRAM_X for d in t["dst"]):
            yield "write", t["total_bytes"], tuple(t["src"][1:])


def label(ops):
    """(step, name) per op; matmuls named by K x N of their weight, SDPA by layer type."""
    out, step, mm_seen, prev, occ, sdpa_i = [], 0, 0, "start", Counter(), 0
    for r in ops:
        code = r["OP CODE"].replace("DeviceOperation", "")
        if code == "Matmul":
            k, n = shape(r, "INPUT_1").split("x")[2:]
            name = f"Matmul {k}x{n}"
            prev, occ = name, Counter()
        elif code == "SdpaDecode":
            name = f"SDPA ({LAYER_TYPES[sdpa_i % 4]})"
            sdpa_i += 1
        else:
            name = f"{code} after {prev} #{occ[code]}"
            occ[code] += 1
        out.append((step, name))
        if code == "ArgMax":
            step, prev, occ = step + 1, "start", Counter()
    return out


def cell(d):
    rows = [
        r for r in csv.DictReader(open(d / "ops_perf.csv")) if (r.get("METAL TRACE REPLAY SESSION ID") or "").strip()
    ]
    rows.sort(key=lambda r: int(r["DEVICE FW START CYCLE"]))
    agg = defaultdict(lambda: defaultdict(float))
    sdpa = defaultdict(lambda: defaultdict(float))  # (name) -> (dir, size) -> bytes
    sdpa_cores = defaultdict(Counter)
    missing = []
    for r, (step, name) in zip(rows, label(rows)):
        gcc = int(r["GLOBAL CALL COUNT"])
        f = d / "npe_viz" / f"{r['OP CODE']}_ID{gcc // 1024}_traceID1.npeviz.zst"
        a = agg[name]
        a["count"] += 1
        a["us"] += float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3
        a["cores"] = max(a["cores"], int(r["CORE COUNT"]))
        a["tensor_bytes"] += sum(tensor_bytes(r, f"INPUT_{i}") for i in range(6)) + tensor_bytes(r, "OUTPUT_0")
        if name.startswith("SDPA"):
            for i in range(6):
                if r.get(f"INPUT_{i}_MEMORY"):
                    sdpa[name][f"in{i} {shape(r, f'INPUT_{i}')} {r[f'INPUT_{i}_DATATYPE']}"] = tensor_bytes(
                        r, f"INPUT_{i}"
                    )
        if not f.exists():
            missing.append(f.name)
            continue
        for kind, b, core in transfers(f):
            a[kind] += b
            if name.startswith("SDPA"):
                sdpa[name][(kind, b)] += b
                sdpa_cores[name][(kind, core)] += b
    return {"ops": len(rows), "missing": missing, "agg": agg, "sdpa": sdpa, "sdpa_cores": sdpa_cores}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cells", nargs="+", type=Path)
    p.add_argument("--min-mb", type=float, default=1.0)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    dump = {}
    for d in args.cells:
        c = cell(d)
        tot_r = sum(a["read"] for a in c["agg"].values())
        tot_w = sum(a["write"] for a in c["agg"].values())
        print(
            f"\n## {d.parent.name}/{d.name}: {c['ops']} ops, traces missing {len(c['missing'])}; read {tot_r / 1e6:,.2f} MB, write {tot_w / 1e6:,.2f} MB"
        )
        print("| op | n | cores | read MB | write MB | tensor MB | µs |")
        rows = sorted(c["agg"].items(), key=lambda kv: -(kv[1]["read"] + kv[1]["write"]))
        for name, a in rows:
            if (a["read"] + a["write"]) / 1e6 >= args.min_mb:
                print(
                    f"| {name} | {a['count']:.0f} | {a['cores']:.0f} | {a['read'] / 1e6:,.3f} | {a['write'] / 1e6:,.3f} | {a['tensor_bytes'] / 1e6:,.3f} | {a['us']:,.1f} |"
                )
        for name, br in c["sdpa"].items():
            n = c["agg"][name]["count"]
            print(
                f"  {name} per call:",
                {
                    str(k): (round(v / n / 1e6, 4) if isinstance(k, tuple) else round(v / 1e6, 4))
                    for k, v in sorted(br.items(), key=str)
                },
            )
            cores = c["sdpa_cores"][name]
            for kind in ("read", "write"):
                per = sorted(v / n for (k, _), v in cores.items() if k == kind)
                if per:
                    print(
                        f"    {kind}: {len(per)} cores, per core min / median / max {per[0]:,.0f} / {per[len(per) // 2]:,.0f} / {per[-1]:,.0f} B"
                    )
        dump[f"{d.parent.name}/{d.name}"] = {
            "ops": c["ops"],
            "missing": c["missing"],
            "agg": c["agg"],
            "sdpa": {k: {str(kk): vv for kk, vv in v.items()} for k, v in c["sdpa"].items()},
        }
    if args.json:
        args.json.write_text(json.dumps(dump, indent=1) + "\n")


if __name__ == "__main__":
    main()
