#!/usr/bin/env python3
"""Re-run tt-npe on DRAM-sharded fat-K matmul traces with their true transfer sizes (gemma4-12b record §9.2a / §9.2e).

The device NoC trace stores a transfer's size as a uint8 count of 32 B chunks (LocalNocEvent::payload_chunks,
tt_metal/tools/profiler/event_metadata.hpp:69-89), so any transfer above 8,160 B is recorded as 8,160 B. The
DRAM-sharded reader's in1 reads are 16 KB bursts (NOC_MAX_BURST_SIZE), its in0 multicasts are 16 KB blocks
(in0_block_w 8 x 2 KB) and its output writes are 8 KB (4 tiles); all are recorded as 8,160 B. This script copies the
listed ops' merged traces from the profiler's .logs (only the last noc cell's traces are there) and runs tt-npe's
per-op step (process_trace) on them twice: <out>/orig/npe_viz with the recorded sizes (a check that this path
reproduces the cell's npe_viz) and <out>/fixed/npe_viz with the 8,160 B events restored by type (READ -> 16,384,
WRITE_MULTICAST -> 16,384, WRITE_ -> 8,192) after checking that the restored DRAM read bytes equal the weight bytes.

    python research_codes/mm_profiling/npe_sizefix.py --ids 562,529 --weight-bytes 16777216,15728640 \\
        --out generated/lmhead_l1_profile/t2_noc/02_noc_B+logitsl1+gridwide+vc1+dramshard/sizefix
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from run_lmhead_e2e import REPO

LOGS = REPO / "generated/profiler/.logs"
CAP = 8160
TRUE = {"READ": 16384, "WRITE_MULTICAST": 16384, "WRITE_": 8192}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True)
    p.add_argument("--weight-bytes", required=True, help="per id, the op's weight bytes (check)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ids = [int(v) for v in a.ids.split(",")]
    wbytes = [int(v) for v in a.weight_bytes.split(",")]
    npe = Path(os.getenv("NPE_ROOT", Path.home() / "codes/tt-npe")) / "install"
    sys.path[:0] = [str(npe / "lib"), str(npe / "bin")]
    os.environ["TT_NPE_DEVICE_MODEL_CONFIG_DIR"] = str(npe / "share/tt_npe/device/models")
    from npe_analyze_noc_trace_dir import OpUID, TopologyGraph, process_trace

    # "orig": the recorded sizes, through this same path (must reproduce the cell's npe_viz); "fixed": restored sizes
    for variant in ("orig", "fixed"):
        d = a.out / variant
        (d / "npe_viz").mkdir(parents=True, exist_ok=True)
        for name in ("cluster_coordinates.json", "topology.json"):
            shutil.copy(LOGS / name, d / name)
        device = TopologyGraph(str(d / "topology.json")).cluster_type
        for i, wb in zip(ids, wbytes):
            (src,) = LOGS.glob(f"noc_trace_ID{i}_*merged.json")
            ev = json.loads(src.read_text())
            if variant == "fixed":
                issuers = {(e["sx"], e["sy"]) for e in ev if "type" in e}
                dram = [e for e in ev if e.get("type") == "READ" and (e["dx"], e["dy"]) not in issuers]
                fixed = 0
                for e in ev:
                    t = e.get("type")
                    if t in TRUE and e.get("num_bytes") == CAP:
                        e["num_bytes"] = TRUE[t]
                        fixed += 1
                got = sum(e["num_bytes"] for e in dram)
                print(f"ID{i}: {fixed} events restored; DRAM read bytes {got} vs weight {wb} ({got / wb:.4f})")
                if got != wb:
                    sys.exit(f"ID{i}: restored DRAM read bytes differ from the weight bytes")
            f = d / src.name
            f.write_text(json.dumps(ev))
            r = process_trace(
                (str(f), "MatmulDeviceOperation", OpUID(i, 1)),
                device,
                str(d / "topology.json"),
                True,
                str(d / "npe_viz"),
                True,
                10000,
            )
            print(variant, f"ID{i}", "ok" if r else "FAILED")
            f.unlink()


if __name__ == "__main__":
    main()
