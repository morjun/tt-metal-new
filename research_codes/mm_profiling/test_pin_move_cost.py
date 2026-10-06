# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gate 2 of the "moving pinned weights" plan: what does an L1 -> L1 move of the head's overlap cost?

For each candidate pin size, the part of the head that overlaps the global SDPA's CB stack is
`801,024 - S` bytes on every SDPA core (S = head start). That part must move to the other cores'
free L1 before the SDPA and back after it. This times exactly those moves, traced:

  target case : 64 source cores (logical 8x8, the re-gridded target global SDPA) -> 46 parking cores
  drafter case: 32 source cores (logical 8x4, the drafter's global SDPA)        -> 78 parking cores

Each parking core pulls an equal contiguous share of the source bytes (out); each source core pulls
its bytes back (back). Movers: `l1_pull_chunks.cpp` on one RISC (NoC0), or split over both
data-movement RISCs (NoC0 + NoC1). Data are checked bit-exactly after out and after back. Writes
generated/lmhead_l1_profile/pin_move_cost/move_cost.json; pin_move_feasibility.py has the gain side.

    pytest research_codes/mm_profiling/test_pin_move_cost.py -s
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

sys.path.insert(0, str(Path(__file__).parent))

from l1_mover import core_set, l1_sharded, l1_sharded_spec, round_trip_programs, share_bytes  # noqa: E402

SDPA_CEILING, TOP = 801_024, 1_572_864
# overlap bytes per SDPA core for each candidate pin size (head start S = TOP - k * 65,536)
PIN_OVERLAP = {
    cols: SDPA_CEILING - (TOP - (cols // 3520) * 65_536) for cols in (42_240, 45_760, 49_280, 52_800, 56_320)
}
SIZES = [64] + sorted(PIN_OVERLAP.values())  # 64 B measures the fixed per-move cost
INNER, REPLAYS, ROUNDS = int(os.getenv("MOVE_INNER", "20")), int(os.getenv("MOVE_REPLAYS", "20")), 3


def timed(device, fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(INNER):
        fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    for _ in range(3):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / REPLAYS / INNER * 1e6
    ttnn.release_trace(device, tid)
    return us


def run_case(device, name, src_cores, dst_cores, src_bytes, both_riscs):
    n_src, n_dst = len(src_cores), len(dst_cores)
    dst_bytes = share_bytes(n_src, src_bytes, n_dst)
    data = torch.randn(1, 1, n_src, src_bytes // 2, dtype=torch.bfloat16)
    src = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=l1_sharded_spec(src_cores, src_bytes),
    )
    park = l1_sharded(device, dst_cores, dst_bytes)
    back = l1_sharded(device, src_cores, src_bytes)  # restore target, separate so "back" is verifiable
    out_prog, back_prog = round_trip_programs(
        device,
        src_cores,
        dst_cores,
        src_bytes,
        src.buffer_address(),
        park.buffer_address(),
        back.buffer_address(),
        both_riscs,
    )
    ttnn.generic_op([src, park], out_prog)
    ttnn.generic_op([park, back], back_prog)
    flat = data.reshape(-1)
    got_park = ttnn.to_torch(park).reshape(-1)[: flat.numel()]
    got_back = ttnn.to_torch(back).reshape(data.shape)
    exact = bool(torch.equal(got_park, flat)) and bool(torch.equal(got_back, data))
    t_out = [timed(device, lambda: ttnn.generic_op([src, park], out_prog)) for _ in range(ROUNDS)]
    t_back = [timed(device, lambda: ttnn.generic_op([park, back], back_prog)) for _ in range(ROUNDS)]
    for t in (src, park, back):
        t.deallocate(True)
    row = {
        "case": name,
        "riscs": 2 if both_riscs else 1,
        "bytes_per_src_core": src_bytes,
        "n_src": n_src,
        "n_dst": n_dst,
        "total_B": n_src * src_bytes,
        "bit_exact": exact,
        "out_us": statistics.median(t_out),
        "back_us": statistics.median(t_back),
        "out_runs": t_out,
        "back_runs": t_back,
    }
    gbps = row["total_B"] / (row["out_us"] * 1e-6) / 1e9
    logger.info(
        f"[move] {name:12s} riscs={row['riscs']} {src_bytes:>7,} B x {n_src} -> {n_dst}: "
        f"out {row['out_us']:8.2f} us  back {row['back_us']:8.2f} us  ({gbps:7.1f} GB/s out)  exact={exact}"
    )
    return row


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 100_000_000})
def test_pin_move_cost(mesh_device, reset_seeds):
    cases = {
        "target_8x8": (core_set(lambda x, y: x < 8 and y < 8), core_set(lambda x, y: not (x < 8 and y < 8))),
        "drafter_8x4": (core_set(lambda x, y: x < 8 and y < 4), core_set(lambda x, y: not (x < 8 and y < 4))),
    }
    rows = []
    # MOVE_CASE=name, MOVE_RISCS=1|2, MOVE_SIZES=comma list select a subset (one process per subset).
    sel_case, sel_riscs = os.getenv("MOVE_CASE"), os.getenv("MOVE_RISCS")
    sel_sizes = [int(v) for v in os.getenv("MOVE_SIZES", "").split(",") if v] or SIZES
    for both in [r == "2" for r in sel_riscs.split(",")] if sel_riscs else (False, True):
        for name, (src_c, dst_c) in cases.items():
            if sel_case and name != sel_case:
                continue
            for b in sel_sizes:
                rows.append(run_case(mesh_device, name, src_c, dst_c, b, both))
    tag = os.getenv("MOVE_TAG", "all")
    out = Path(__file__).resolve().parents[2] / f"generated/lmhead_l1_profile/pin_move_cost/move_cost_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"pin_overlap_B": PIN_OVERLAP, "rows": rows, "inner": INNER, "replays": REPLAYS, "rounds": ROUNDS}, indent=1
        )
        + "\n"
    )
    logger.info(f"[move] wrote {out}")
    assert all(r["bit_exact"] for r in rows)
