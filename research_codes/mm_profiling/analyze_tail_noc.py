#!/usr/bin/env python3
"""NoC trace of the lm_head DRAM tail at 98 vs 99 cores (gemma4-12b record §9.4, stage 5).

Cells from run_1031_tail_noc.sh (generated/lmhead_l1_profile/tail_noc/): time_<w> (no NoC trace) and noc_<w> (tt-npe
files per op). Per profiled tail call (the warm-up call is dropped):
  - op duration (DEVICE KERNEL DURATION) traced vs untraced
  - DRAM reads (READ transfers whose source is never a destination): per core bytes per endpoint, issue span, block
    cadence (median interval between the issue of every 12th read = one in1 block of 2 K rows x 6 columns)
  - per link: bytes routed over it (from each transfer's measured route) / op time = GB/s ("route load"), and tt-npe's
    modelled mean demand (% of the link's bandwidth; a model, not a measurement)
Compares the 99-core width against the 98-core width: the added core, its routes, the links whose load rises most,
and the cores whose cadence slows most.

    python research_codes/mm_profiling/analyze_tail_noc.py generated/lmhead_l1_profile/tail_noc 6467 6533
"""

import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

MHZ = 1350.0
NPE_LINK_BW = 60.9
LINK_KINDS = ("NOC0_EAST", "NOC0_SOUTH", "NOC1_NORTH", "NOC1_WEST")


def tail_calls(cell):
    """[(global call id, device kernel us)] of the DRAM tail matmuls, warm-up dropped."""
    rows = list(csv.DictReader(open(cell / "ops_perf.csv")))
    out = []
    for r in rows:
        if "Matmul" in r["OP CODE"] and "DRAM" in r.get("INPUT_1_MEMORY", ""):
            x = r.get("INPUT_1_X_PAD[LOGICAL]", "")
            if "[" in x and int(x.split("[")[1].rstrip("]")) > 100_000:
                out.append((int(r["GLOBAL CALL COUNT"]), float(r["DEVICE KERNEL DURATION [ns]"]) / 1e3))
    return out[1:]


def npe(cell, call_id):
    fs = []
    for n in (call_id // 1024, call_id):
        for pat in (f"MatmulDeviceOperation_ID{n}.npeviz.zst", f"MatmulDeviceOperation_ID{n}_*.npeviz.zst"):
            fs = fs or [f for f in (cell / "npe_viz").glob(pat) if "split" not in f.name]
    (f,) = fs
    return json.loads(subprocess.run(["zstd", "-dc", str(f)], capture_output=True, check=True).stdout)


def analyze_call(d, op_us):
    reads = [t for t in d["noc_transfers"] if t["noc_event_type"] == "READ"]
    dsts = {tuple(x) for t in reads for x in t["dst"]}
    dram = [t for t in reads if tuple(t["src"]) not in dsts]
    per_core = defaultdict(list)
    ep_bytes = defaultdict(lambda: defaultdict(int))
    link_bytes = defaultdict(int)
    core_links = defaultdict(lambda: defaultdict(int))
    for t in dram:
        core = tuple(t["dst"][0][1:])
        per_core[core].append(t["start_cycle"] / MHZ)
        ep_bytes[core][tuple(t["src"][1:])] += t["total_bytes"]
        for r in t["route"]:
            for _, row, col, kind in r["links"]:
                if kind in LINK_KINDS:
                    link_bytes[(row, col, kind)] += t["total_bytes"]
                    core_links[core][(row, col, kind)] += t["total_bytes"]
    t0 = min(min(v) for v in per_core.values())
    cad, span = {}, {}
    for c, ts in per_core.items():
        ts.sort()
        starts = ts[::12]
        cad[c] = statistics.median(b - a for a, b in zip(starts, starts[1:]))
        span[c] = (ts[0] - t0, ts[-1] - t0)
    load = {k: v / op_us / 1e3 for k, v in link_bytes.items()}  # GB/s
    steps = d["timestep_data"]
    model = defaultdict(float)
    for s in steps:
        for _, row, col, kind, dem in s["link_demand"]:
            if kind in LINK_KINDS:
                model[(row, col, kind)] += dem / NPE_LINK_BW * 100 / len(steps)
    return {"cad": cad, "span": span, "ep": ep_bytes, "load": load, "model": model, "core_links": core_links}


def mean_dicts(ds):
    keys = set().union(*ds)
    return {k: statistics.mean(d.get(k, 0.0) for d in ds) for k in keys}


def cell_summary(root, w):
    calls = tail_calls(root / f"noc_{w}")
    untraced = [us for _, us in tail_calls(root / f"time_{w}")]
    res = [analyze_call(npe(root / f"noc_{w}", cid), us) for cid, us in calls]
    return {
        "traced_us": [us for _, us in calls],
        "untraced_us": untraced,
        "cad": mean_dicts([r["cad"] for r in res]),
        "load": mean_dicts([r["load"] for r in res]),
        "model": mean_dicts([r["model"] for r in res]),
        "ep": res[0]["ep"],
        "core_links": res[0]["core_links"],
        "span": res[0]["span"],
    }


def main(root, w98, w99):
    root = Path(root)
    a, b = cell_summary(root, w98), cell_summary(root, w99)
    for w, s in ((w98, a), (w99, b)):
        print(
            f"width {w}: cores {len(s['cad'])}; tail us traced {[round(x, 1) for x in s['traced_us']]} "
            f"untraced {[round(x, 1) for x in s['untraced_us']]}; cadence median {statistics.median(s['cad'].values()):.3f} us/block"
        )
    added = sorted(set(b["cad"]) - set(a["cad"]))
    print(f"added core(s) (row, col physical): {added}")
    for c in added:
        eps = {k: round(v / 1e6, 2) for k, v in sorted(b["ep"][c].items())}
        print(
            f"  {c}: MB per endpoint {eps}; issue span {b['span'][c][0]:.1f}-{b['span'][c][1]:.1f} us; cadence {b['cad'][c]:.3f}"
        )
        links = sorted(b["core_links"][c], key=b["core_links"][c].get, reverse=True)
        print(f"  its route links ({len(links)}): {links}")
    # link loads
    keys = set(a["load"]) | set(b["load"])
    dl = {k: b["load"].get(k, 0) - a["load"].get(k, 0) for k in keys}
    dm = {k: b["model"].get(k, 0) - a["model"].get(k, 0) for k in set(a["model"]) | set(b["model"])}
    added_links = set().union(*(b["core_links"][c] for c in added)) if added else set()
    print(
        f"median route-load change over links {statistics.median(dl.values()):+.2f} GB/s; "
        f"median modelled change {statistics.median(dm.values()):+.2f} pts"
    )
    print("top route-load links at 99 cores (GB/s at 98 -> 99, modelled % 98 -> 99, on the added core's route):")
    for k in sorted(b["load"], key=b["load"].get, reverse=True)[:12]:
        print(
            f"  {k}: {a['load'].get(k, 0):6.1f} -> {b['load'][k]:6.1f}   model {a['model'].get(k, 0):6.1f} -> {b['model'].get(k, 0):6.1f}   {'YES' if k in added_links else ''}"
        )
    print("largest route-load increases:")
    for k in sorted(dl, key=dl.get, reverse=True)[:10]:
        print(
            f"  {k}: {dl[k]:+6.1f} GB/s ({a['load'].get(k, 0):.1f} -> {b['load'].get(k, 0):.1f}); model {dm.get(k, 0):+5.1f} pts   {'YES' if k in added_links else ''}"
        )
    print("largest modelled-demand increases:")
    for k in sorted(dm, key=dm.get, reverse=True)[:10]:
        print(
            f"  {k}: {dm[k]:+5.1f} pts ({a['model'].get(k, 0):.1f} -> {b['model'].get(k, 0):.1f})   {'YES' if k in added_links else ''}"
        )
    # per-core cadence change
    common = set(a["cad"]) & set(b["cad"])
    ratio = {c: b["cad"][c] / a["cad"][c] for c in common}
    print(
        f"cadence ratio 99/98 over the {len(common)} common cores: median {statistics.median(ratio.values()):.4f}, "
        f"min {min(ratio.values()):.4f}, max {max(ratio.values()):.4f}"
    )
    top = sorted(ratio, key=ratio.get, reverse=True)[:8]
    for c in top:
        share = sum(v for k, v in b["core_links"][c].items() if k in added_links) / sum(b["core_links"][c].values())
        print(
            f"  slowest-growing {c}: x{ratio[c]:.4f}; share of its bytes on links the added core also uses {share:.2f}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:4])
