# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gate 4 of the "larger pin" study (gemma4-12b MEASUREMENT_RECORD §5.5.14): can a pinned head
that reaches below the global SDPA's CB ceiling survive the SDPA, inside a trace?

Mechanism under test (no per-core allocation; the default LOCKSTEP allocator):
1. Free the whole pinned head tensor right before the SDPA. That is host-side bookkeeping only; it
   does not touch L1. The SDPA's CBs now pass the host-side CB clash check.
2. Run the SDPA. It overwrites [CB base, 801,024) on its own cores only.
3. Re-allocate the head at the same address (asserted).
4. Reload only the overwritten bytes [S, 801,024) (rounded up to 2,048 B pages), on only the
   overwritten cores, from a DRAM snapshot.
5. Run the lm_head slice matmul on the re-allocated head. Its output must equal the reference
   bit-for-bit.

The pinned slice is 52,800 cols: 15 tiles/core on 110 cores, at S = 589,824.

The SDPA stand-in is a generic_op program whose CB c_0 spans exactly [111,488, 801,024), the
captured global-SDPA ceiling, on the SDPA's cores. Its kernel fills the whole CB with junk, so it
overwrites everything a real SDPA could. Core sets: 110 (target global SDPA as it runs today,
11x10) and 32 (drafter global SDPA, 8x4).

Checks:
- A. With the head allocated, the stand-in is rejected (CB clash). This confirms it is a faithful
  collider.
- B. Eager: free -> stand-in(32) -> re-allocate -> lm_head WITHOUT reload must differ from the
  reference (the overwrite is real); with the reload it must match. Then the same with 110 cores.
- C. Traced: one capture of [target step on 110 cores, drafter step on 32, drafter step on 32]
  (free / stand-in / re-allocate / reload / lm_head each), replayed REPLAYS times. Every lm_head
  output must match the reference after every replay, and the head's full contents must equal the
  original after the last replay.

    pytest research_codes/mm_profiling/test_pin_free_realloc_trace.py -s
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

K, COLS, CORES_TOTAL, T, PAGE = 1024, 52_800, 110, 32, 2048
TOP, CB_BASE, CEILING = 1_572_864, 111_488, 801_024
S_EXPECTED = TOP - (COLS // T // CORES_TOTAL) * K * 2 * T  # 589,824
RELOAD_PAGES = -(-(CEILING - S_EXPECTED) // PAGE)  # 104 pages = 212,992 B per core
REPLAYS = int(os.getenv("GATE4_REPLAYS", "5"))
FULL = [(x, y) for y in range(10) for x in range(11)]
DRAFT = [(x, y) for y in range(4) for x in range(8)]
KDIR = "research_codes/mm_profiling/kernels/"


def crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def head_spec():
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(FULL), [K, COLS // CORES_TOTAL], ttnn.ShardOrientation.ROW_MAJOR),
    )


def slice_pc():
    # The model's slice config (_split_lm_head_to_l1).
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=COLS // T // CORES_TOTAL,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def page_program(kernel, tensor_dram, cores, l1_addr):
    """Per core i (index in FULL): move pages [idx * RELOAD_PAGES, +RELOAD_PAGES) <-> L1 at l1_addr."""
    ct = [PAGE] + list(ttnn.TensorAccessorArgs(tensor_dram).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for x, y in cores:
        rt[x][y] = [tensor_dram.buffer_address(), FULL.index((x, y)) * RELOAD_PAGES, RELOAD_PAGES, l1_addr]
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + kernel,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs(cores),
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])


def clobber_program(cores, pattern, page_size=64):
    size = CEILING - CB_BASE  # 689,536 B: the CB stack up to the captured SDPA ceiling
    cb = ttnn.CBDescriptor(
        total_size=size,
        core_ranges=crs(cores),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=page_size)],
    )
    rt = ttnn.RuntimeArgs()
    for x, y in cores:
        rt[x][y] = [size, pattern]
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + "l1_clobber_cb.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs(cores),
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 50_000_000})
def test_pin_free_realloc_trace(mesh_device, reset_seeds):
    dev = mesh_device
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    report = {"pin_cols": COLS, "S_expected": S_EXPECTED, "reload_pages_per_core": RELOAD_PAGES}
    x = ttnn.rand([1, 1, 1, K], device=dev, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    w = ttnn.rand([1, 1, K, COLS], device=dev, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
    head = ttnn.to_memory_config(w, head_spec())
    w.deallocate(True)
    S = head.buffer_address()
    report["S"] = S
    assert S == S_EXPECTED, f"head at {S}, expected {S_EXPECTED}"
    head_ref = ttnn.to_torch(head)

    def lm_head(h):
        return ttnn.linear(x, h, program_config=slice_pc(), compute_kernel_config=ckc)

    ref = ttnn.to_torch(lm_head(head))
    # DRAM snapshot of every core's first RELOAD_PAGES pages of the head shard (taken once, at load).
    snap = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, CORES_TOTAL * RELOAD_PAGES, PAGE // 2]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        dev,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.generic_op([head, snap], page_program("dram_snapshot_pages.cpp", snap, FULL, S))
    ttnn.synchronize_device(dev)

    # A. The stand-in must collide with the allocated head.
    try:
        # page_size 32: a different program hash, so this rejected program never shares a cache entry
        ttnn.generic_op([x, x], clobber_program(DRAFT, 0x5A5A0000, page_size=32))
        report["A_clash_with_head_allocated"] = False
    except RuntimeError as e:
        report["A_clash_with_head_allocated"] = True
        report["A_message"] = str(e).splitlines()[0][:300]
    logger.info(f"[gate4] A: stand-in rejected while the head is allocated: {report['A_clash_with_head_allocated']}")
    assert report["A_clash_with_head_allocated"], "stand-in did not collide: not a faithful SDPA stand-in"

    def step(h, cores, pattern, reload=True):
        """free -> stand-in on `cores` -> re-allocate (same address) -> reload -> lm_head."""
        h.deallocate(True)
        ttnn.generic_op([x, x], clobber_program(cores, pattern))
        h = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, K, COLS]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, head_spec()
        )
        assert h.buffer_address() == S, f"re-allocated at {h.buffer_address()}, expected {S}"
        if reload:
            ttnn.generic_op([snap, h], page_program("dram_reload_pages.cpp", snap, cores, S))
        return h, lm_head(h)

    # B. Eager.
    for name, cores in (("drafter_32", DRAFT), ("target_110", FULL)):
        head, out = step(head, cores, 0x11110000, reload=False)
        no_reload_equal = bool(torch.equal(ttnn.to_torch(out), ref))
        out.deallocate(True)
        head, out = step(head, cores, 0x22220000, reload=True)
        reload_equal = bool(torch.equal(ttnn.to_torch(out), ref))
        out.deallocate(True)
        report[f"B_{name}"] = {"no_reload_equals_ref": no_reload_equal, "reload_equals_ref": reload_equal}
        logger.info(
            f"[gate4] B {name}: without reload == ref: {no_reload_equal} (must be False); "
            f"with reload == ref: {reload_equal} (must be True)"
        )
    head_after_b = bool(torch.equal(ttnn.to_torch(head), head_ref))
    report["B_head_intact"] = head_after_b
    logger.info(f"[gate4] B: head contents == original after eager steps: {head_after_b}")

    # C. Traced: target step (110) + 2 drafter steps (32), captured once, replayed.
    def iteration(h):
        outs = []
        for cores, pat in ((FULL, 0x33330000), (DRAFT, 0x44440000), (DRAFT, 0x55550000)):
            h, o = step(h, cores, pat)
            outs.append(o)
        return h, outs

    head, warm = iteration(head)  # compile everything outside the capture
    for o in warm:
        o.deallocate(True)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    head, outs = iteration(head)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    replays = []
    for r in range(REPLAYS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        replays.append([bool(torch.equal(ttnn.to_torch(o), ref)) for o in outs])
        logger.info(f"[gate4] C replay {r}: lm_head outputs == ref: {replays[-1]}")
    ttnn.release_trace(dev, tid)
    report["C_replays"] = replays
    report["C_head_intact"] = bool(torch.equal(ttnn.to_torch(head), head_ref))
    report["C_head_addr"] = head.buffer_address()
    logger.info(f"[gate4] C: head contents == original after {REPLAYS} replays: {report['C_head_intact']}")

    out_path = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/gate4/gate4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    logger.info(f"[gate4] wrote {out_path}")
    for o in outs:
        o.deallocate(True)
    for t in (head, snap, x):
        t.deallocate(True)
    assert not report["B_drafter_32"]["no_reload_equals_ref"] and not report["B_target_110"]["no_reload_equals_ref"]
    assert report["B_drafter_32"]["reload_equals_ref"] and report["B_target_110"]["reload_equals_ref"]
    assert head_after_b and report["C_head_intact"] and all(all(r) for r in replays)
