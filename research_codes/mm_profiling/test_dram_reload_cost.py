# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What does reloading the overwritten part of a pinned lm_head from DRAM cost? (gemma4-12b
MEASUREMENT_RECORD §5.5.14, Gate 2e.)

In the DRAM-reload scheme, a global SDPA overwrites the head's overlap `[S, 801,024)` on its own
cores, and the bytes are read back from a DRAM copy before the next drafter lm_head. This times
that read, standalone: every core of a set reads its overlap (rounded up to whole 2,048 B pages)
from a DRAM interleaved tensor into its own L1.

Core sets:
- 110 (full 11x10: the target global SDPA as it runs today);
- 64 (8x8: re-gridded target SDPA);
- 32 (8x4: the drafter global SDPA).

Overlap per core for the 42,240 ... 56,320-col pins: 14,592 / 80,128 / 145,664 / 211,200 / 276,736 B.

Movers:
- `custom_1r`: kernels/dram_reload_pages.cpp on one RISC (NoC0). The DRAM copy is page-interleaved
  with 2,048 B pages, and core i reads its run of consecutive pages, so every core touches all 8
  banks.
- `custom_2r`: the same split over both data-movement RISCs (NoC0 + NoC1).
- `stock`: ttnn.to_memory_config from a DRAM interleaved ROW_MAJOR tensor (one page = one core's
  whole chunk) to the L1 height-sharded layout. It allocates its output on every call.

Timing: traced, as test_pin_move_cost.py. Every arm's L1 result is compared byte-for-byte with the
source. Writes generated/lmhead_l1_profile/dram_reload/reload_cost.json.

    pytest research_codes/mm_profiling/test_dram_reload_cost.py -s
"""

import json
import os
import statistics
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

KERNEL = "research_codes/mm_profiling/kernels/dram_reload_pages.cpp"
PAGE = 2048
OVERLAPS = [14_592, 80_128, 145_664, 211_200, 276_736]
INNER, REPLAYS, ROUNDS = int(os.getenv("RELOAD_INNER", "20")), int(os.getenv("RELOAD_REPLAYS", "20")), 3
SETS = {
    "full_110": [(x, y) for y in range(10) for x in range(11)],
    "8x8_64": [(x, y) for y in range(8) for x in range(8)],
    "8x4_32": [(x, y) for y in range(4) for x in range(8)],
}
SEL_SETS = [s for s in os.getenv("RELOAD_SETS", "").split(",") if s] or list(SETS)
SEL_SIZES = [int(v) for v in os.getenv("RELOAD_SIZES", "").split(",") if v] or OVERLAPS


def crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def l1_spec(cores, row_bytes):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(cores), [1, row_bytes // 2], ttnn.ShardOrientation.ROW_MAJOR),
    )


def reload_program(src, dst, cores, ppc, both):
    """Core i reads pages [i * ppc, (i + 1) * ppc) of `src` into its shard of `dst`."""
    ct = [PAGE] + list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
    halves = [(0, ppc // 2), (ppc // 2, ppc - ppc // 2)] if both else [(0, ppc)]
    kernels = []
    for h, (off, n) in enumerate(halves):
        rt = ttnn.RuntimeArgs()
        for i, (x, y) in enumerate(cores):
            rt[x][y] = [src.buffer_address(), i * ppc + off, n, dst.buffer_address() + off * PAGE]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs(cores),
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor() if h == 0 else ttnn.WriterConfigDescriptor(),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])


def timed(device, fn):
    """Per-call us; `fn` returns a tensor to free after the batch, or None."""

    def batch():
        return [fn() for _ in range(INNER)]

    for o in batch():
        if o is not None:
            o.deallocate(True)
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = batch()
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
    for o in outs:
        if o is not None:
            o.deallocate(True)
    return us


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 100_000_000})
def test_dram_reload_cost(mesh_device, reset_seeds):
    dev = mesh_device
    rows, times, exact, nofit = [], {}, {}, {}
    for r in range(ROUNDS):
        for set_name in SEL_SETS if r % 2 == 0 else list(reversed(SEL_SETS)):
            cores = SETS[set_name]
            n = len(cores)
            for overlap in SEL_SIZES if r % 2 == 0 else list(reversed(SEL_SIZES)):
                ppc = -(-overlap // PAGE)
                chunk = ppc * PAGE
                data = torch.randn(1, 1, n, chunk // 2, dtype=torch.bfloat16)
                # custom source: one 2,048 B page per row, page-interleaved over the 8 DRAM banks
                src_pages = ttnn.from_torch(
                    data.reshape(1, 1, n * ppc, PAGE // 2),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                # stock source: one page = one core's whole chunk
                src_rows = ttnn.from_torch(
                    data,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                dst = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, n, chunk // 2]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev, l1_spec(cores, chunk)
                )
                progs = {
                    m: reload_program(src_pages, dst, cores, ppc, m == "custom_2r") for m in ("custom_1r", "custom_2r")
                }
                arms = {
                    "custom_1r": lambda: (ttnn.generic_op([src_pages, dst], progs["custom_1r"]), None)[1],
                    "custom_2r": lambda: (ttnn.generic_op([src_pages, dst], progs["custom_2r"]), None)[1],
                    "stock": lambda: ttnn.to_memory_config(src_rows, l1_spec(cores, chunk)),
                }
                # timed() frees outputs only after the whole batch; 20 live stock outputs of up to
                # 278 KB/core would not fit, so the timed stock arm frees each output at once.
                timed_fn = {
                    **arms,
                    "stock": lambda: ttnn.to_memory_config(src_rows, l1_spec(cores, chunk)).deallocate(True),
                }
                for method, fn in arms.items():
                    key = (set_name, overlap, method)
                    if key in nofit:
                        continue
                    if key not in exact:
                        if method == "stock":
                            try:  # its CBs hold several whole-row pages; large rows clash with L1 buffers
                                o = fn()
                            except RuntimeError as e:
                                nofit[key] = str(e).splitlines()[0][:300]
                                logger.info(f"[reload] {key}: stock does not fit: {nofit[key]}")
                                continue
                            got = ttnn.to_torch(o)
                            o.deallocate(True)
                        else:
                            fn()
                            got = ttnn.to_torch(dst)
                        exact[key] = bool(torch.equal(got.reshape(data.shape), data))
                    times.setdefault(key, []).append(timed(dev, timed_fn[method]))
                for t in (src_pages, src_rows, dst):
                    t.deallocate(True)
    for set_name in SEL_SETS:
        n = len(SETS[set_name])
        for overlap in SEL_SIZES:
            ppc = -(-overlap // PAGE)
            for method in ("custom_1r", "custom_2r", "stock"):
                key = (set_name, overlap, method)
                if key in nofit:
                    rows.append(
                        {"cores": set_name, "n_cores": n, "overlap_B": overlap, "method": method, "no_fit": nofit[key]}
                    )
                    logger.info(f"[reload] {set_name:8s} overlap {overlap:>7,} B {method:9s} no fit")
                    continue
                us = statistics.median(times[key])
                total = n * ppc * PAGE
                row = {
                    "cores": set_name,
                    "n_cores": n,
                    "overlap_B": overlap,
                    "bytes_per_core": ppc * PAGE,
                    "total_B": total,
                    "method": method,
                    "us": us,
                    "runs": times[key],
                    "GBps": total / us / 1e3,
                    "bit_exact": exact[key],
                }
                rows.append(row)
                logger.info(
                    f"[reload] {set_name:8s} overlap {overlap:>7,} B ({ppc} pages/core, {total / 1e6:6.2f} MB) "
                    f"{method:9s} {us:8.2f} us  {row['GBps']:6.1f} GB/s  exact={row['bit_exact']}  "
                    f"runs={[round(v, 2) for v in times[key]]}"
                )
    out = (
        Path(__file__).resolve().parents[2]
        / "generated/lmhead_l1_profile/dram_reload"
        / os.getenv("RELOAD_OUT", "reload_cost.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "inner": INNER, "replays": REPLAYS, "rounds": ROUNDS}, indent=1) + "\n")
    logger.info(f"[reload] wrote {out}")
    assert all(exact.values())
