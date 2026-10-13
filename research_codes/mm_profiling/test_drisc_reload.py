# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hidden lm_head reload on the DRAM cores' RISC-Vs (gemma4-12b MEASUREMENT_RECORD §5.5.14, reload hiding).

Drives lm_head_reload.py with GEMMA4_LMHEAD_L1_RELOAD_DRISC on a stand-alone pinned head: 59,840 cols
(17 tiles/core on 110 cores, S = 458,752; the standalone-optimum pin), 168 reload pages per core.

One "step" = the drafter's sequence around its global SDPA:
  drafter_step(); bracket("global SDPA"){ SDPA stand-in }  -> the DRISC trigger fires here
  window: WINDOW DRAM matmuls ([32, 1024] x [1024, 8192], 16.8 MB of weights each, like o_proj/gate/up/down)
  reload()                                                  -> waits for the DRISCs
The SDPA stand-in is Gate 4's generic_op whose CB c_0 spans [111,488, 801,024) and is filled with junk.

Checks:
- A. Eager, 32 cores (8x4) and 110 cores: after the step the head equals the original; the same step with
  the reload skipped does not (the overwrite is real).
- B. Traced: one capture of [110-core step, 32-core step, 32-core step], replayed REPLAYS times; the head
  equals the original after every replay.
- C. Timing, traced, TIMED_REPLAYS replays of STEPS 32-core steps per arm: DRISC (trigger + wait),
  Tensix (the reload op as today), none (no reload; wrong data, the lower bound).

    pytest research_codes/mm_profiling/test_drisc_reload.py -s
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
from models.demos.gemma4.tt import lm_head_reload as R

K, COLS, CORES_TOTAL, T = 1024, 59_840, 110, 32
TOP, CB_BASE, CEILING = 1_572_864, 111_488, 801_024
S_EXPECTED = TOP - (COLS // T // CORES_TOTAL) * K * 2 * T  # 458,752
FULL = [(x, y) for y in range(10) for x in range(11)]
DRAFT = [(x, y) for y in range(4) for x in range(8)]
REPLAYS = int(os.getenv("DRISC_REPLAYS", "5"))
TIMED_REPLAYS = int(os.getenv("DRISC_TIMED_REPLAYS", "30"))
STEPS = 3
WINDOW = int(os.getenv("DRISC_WINDOW", "4"))
WINDOW_N = int(os.getenv("DRISC_WINDOW_N", "8192"))  # window matmul output width (DRAM weight bytes)
KDIR = "research_codes/mm_profiling/kernels/"


class Owner:
    pass


def crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def head_spec():
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(FULL), [K, COLS // CORES_TOTAL], ttnn.ShardOrientation.ROW_MAJOR),
    )


def clobber_program(cores, pattern):
    size = CEILING - CB_BASE
    cb = ttnn.CBDescriptor(
        total_size=size,
        core_ranges=crs(cores),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=64)],
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
def test_drisc_reload(mesh_device, reset_seeds):
    dev = mesh_device
    R.ENABLED, R.DRISC = True, True
    report = {"pin_cols": COLS, "window_matmuls": WINDOW, "window_n": WINDOW_N}
    x = ttnn.rand([1, 1, 1, K], device=dev, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    a = ttnn.rand([1, 1, 32, K], device=dev, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=3)
    ws = [
        ttnn.rand([1, 1, K, WINDOW_N], device=dev, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=10 + i)
        for i in range(WINDOW)
    ]
    w = ttnn.rand([1, 1, K, COLS], device=dev, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
    owner = Owner()
    owner.head = ttnn.to_memory_config(w, head_spec())
    w.deallocate(True)
    assert owner.head.buffer_address() == S_EXPECTED, owner.head.buffer_address()
    head_ref = ttnn.to_torch(owner.head)
    R.register(owner, "head", dev)
    st = R._st
    report.update(S=st.S, pages=st.pages, waiter=list(st.waiter), driscs=len(st.drisc_xy))

    def window():
        for wt in ws:
            ttnn.linear(a, wt).deallocate(True)

    def step(cores, pattern, mode):
        """mode: 'drisc' | 'tensix' | 'none'."""
        drisc = st.drisc
        if mode != "drisc":
            st.drisc = None  # no trigger: reload() falls back to the Tensix op
        R.drafter_step()
        with R.bracket("global SDPA", x, *([a] if cores is FULL else [])):
            ttnn.generic_op([x, x], clobber_program(cores, pattern))
        window()
        if mode == "none":
            st.dirty.clear()
            st.in_drafter = False
        else:
            R.reload()
        st.drisc = drisc

    def head_ok():
        return bool(torch.equal(ttnn.to_torch(owner.head), head_ref))

    W, TC = COLS // CORES_TOTAL, COLS // CORES_TOTAL // T  # shard width, tiles per shard row

    def tile(t, c, i):  # page i of core c = tile (i // TC, i % TC) of its shard
        r, q = divmod(i, TC)
        return t[0, 0, r * T : (r + 1) * T, c * W + q * T : c * W + (q + 1) * T].contiguous().view(torch.int16)

    ref_where = {
        hash(tile(head_ref, c, i).numpy().tobytes()): (c, i) for c in range(CORES_TOTAL) for i in range(st.pages)
    }

    def drisc_status():
        """Per DRISC: commands (taken, done), from its L1."""
        ex = ttnn._ttnn.operations.experimental
        return {f"{c.x},{c.y}": ex.read_drisc_l1(dev, c, st.drisc_sync_addr, 2) for c in st.drisc_cores}

    logger.info(f"[drisc] status after launch + warm-up: {drisc_status()}")

    def diag():
        """Per wrong page in [S, ceiling): which reference page it holds, if any."""
        got = ttnn.to_torch(owner.head)
        bad, kinds, examples = {}, {}, []
        for c in range(CORES_TOTAL):
            for i in range(st.pages):
                g = tile(got, c, i)
                if torch.equal(g, tile(head_ref, c, i)):
                    continue
                bad[c] = bad.get(c, 0) + 1
                src = ref_where.get(hash(g.numpy().tobytes()))
                kind = "other" if src is None else f"ref page shift {src[0] - c} cores, {src[1] - i} pages"
                kinds[kind] = kinds.get(kind, 0) + 1
                if len(examples) < 12:
                    examples.append({"core": c, "page": i, "holds": src, "first_words": g.flatten()[:4].tolist()})
        return {"bad_pages_per_core": bad, "kinds": kinds, "examples": examples}

    # A. Eager.
    for name, cores in (("drafter_32", DRAFT), ("all_110", FULL)):
        step(cores, 0x11110000, "none")
        no_reload = head_ok()
        step(cores, 0x22220000, "drisc")
        ttnn.synchronize_device(dev)
        with_drisc = head_ok()
        report[f"A_{name}"] = {"no_reload_equals_ref": no_reload, "drisc_reload_equals_ref": with_drisc}
        logger.info(f"[drisc] A {name}: no reload == ref {no_reload} (must be False); DRISC reload == ref {with_drisc}")
        if not with_drisc:
            report[f"A_{name}"]["diag_after_step"] = diag()
            # One more push with nothing clobbered, timed alone: does the done count arrive, and what lands?
            R._drisc_trigger(R.CMD_SMALL if cores is DRAFT else R.CMD_ALL)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            R._drisc_wait()
            ttnn.synchronize_device(dev)
            report[f"A_{name}"]["isolated_wait_s"] = time.perf_counter() - t0
            report[f"A_{name}"]["diag_after_isolated_push"] = diag()
            report[f"A_{name}"]["drisc_status"] = drisc_status()
            logger.info(f"[drisc] A {name} diag: {report[f'A_{name}']}")
            break

    # B. Traced, mixed 110 / 32.
    def iteration():
        for cores, pat in ((FULL, 0x33330000), (DRAFT, 0x44440000), (DRAFT, 0x55550000)):
            step(cores, pat, "drisc")

    replays = []
    if all(v["drisc_reload_equals_ref"] for k, v in report.items() if k.startswith("A_")):
        iteration()  # warm up outside the capture
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        iteration()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        for r in range(REPLAYS):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            replays.append(head_ok())
            logger.info(f"[drisc] B replay {r}: head == ref {replays[-1]}")
        ttnn.release_trace(dev, tid)
    report["B_replays"] = replays

    # C. Timing.
    timing = {}
    if replays and all(replays):
        for mode in ("drisc", "tensix", "none", "drisc", "tensix", "none"):

            def body():
                for s in range(STEPS):
                    step(DRAFT, 0x66660000 + s, mode)

            body()
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            body()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts = []
            for _ in range(TIMED_REPLAYS):
                t0 = time.perf_counter()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) * 1e6 / STEPS)
            ttnn.release_trace(dev, tid)
            timing.setdefault(mode, []).append(statistics.median(ts))
            logger.info(f"[drisc] C {mode}: {statistics.median(ts):.1f} us/step (median of {TIMED_REPLAYS} replays)")
        step(DRAFT, 0x77770000, "tensix")  # restore the head after the 'none' arm
        ttnn.synchronize_device(dev)
        report["C_head_after_timing"] = head_ok()
    report["C_us_per_step"] = timing

    R.drisc_stop()
    out = (
        Path(__file__).resolve().parents[2]
        / f"generated/lmhead_l1_profile/drisc_reload/drisc_reload_w{WINDOW}_n{WINDOW_N}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    logger.info(f"[drisc] wrote {out}: {report}")
    assert not report["A_drafter_32"]["no_reload_equals_ref"]
    assert all(v["drisc_reload_equals_ref"] for k, v in report.items() if k.startswith("A_"))
    assert replays and all(replays)
    assert report.get("C_head_after_timing", False)
