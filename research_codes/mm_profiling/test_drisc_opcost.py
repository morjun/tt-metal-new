# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced per-op cost of the DRISC reload's 1-core Tensix ops (gemma4-12b §5.5.14, reload hiding).

With the resident kernel running and nothing to push (empty core sets), times traced chains of N ops:
- noop: the trigger kernel with no DRISC targets (a bare 1-core program);
- trigger: +1 (CMD_SMALL, empty set) to all 16 DRISCs;
- wait: the wait kernel when every DRISC is idle (go == 0, taken == done);
- trigger+wait pairs.
"""

import statistics
import time

from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

KDIR = "research_codes/mm_profiling/kernels/"
GO, PAGE, CHUNK, N, REPS = 15, 2048, 16, 50, 20
CORE = (10, 9)


def crs(c):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c))])


def prog(kernel, ct, rt_words, cbs=()):
    rt = ttnn.RuntimeArgs()
    rt[CORE[0]][CORE[1]] = rt_words
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + kernel,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs(CORE),
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=list(cbs))


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 20_000_000})
def test_drisc_opcost(mesh_device):
    dev = mesh_device
    ex = ttnn._ttnn.operations.experimental
    senders = ex.drisc_senders(dev)
    base, _ = ex.drisc_l1_unreserved(dev)
    sync_addr = base + 2 * CHUNK * PAGE
    cores = [l for _, l, _ in senders]
    xys = [(v.x << 16) | v.y for _, _, v in senders]
    kern = ex.launch_resident_drisc_kernel(
        dev,
        KDIR + "drisc_reload_resident.cpp",
        cores,
        [PAGE, CHUNK, GO, 0],
        [[0, 8, b, 168, 458752, base, 0, 0] for b, _, _ in senders],
    )
    x = ttnn.rand([1, 1, 32, 32], device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    scratch = ttnn.CBDescriptor(
        total_size=32 * len(xys),
        core_ranges=crs(CORE),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=32)],
    )
    noop = lambda: ttnn.generic_op([x, x], prog("drisc_reload_trigger.cpp", [1, GO, 0], []))  # noqa: E731
    trig = lambda: ttnn.generic_op([x, x], prog("drisc_reload_trigger.cpp", [1, GO, len(xys)], xys))  # noqa: E731
    wait = lambda: ttnn.generic_op(  # noqa: E731
        [x, x], prog("drisc_reload_wait.cpp", [len(xys), GO, sync_addr, 1_000_000], xys, [scratch])
    )
    arms = {"noop": [noop], "trigger": [trig, wait], "wait": [wait], "trigger+wait": [trig, wait], "noop again": [noop]}
    res = {}
    for name, ops in arms.items():
        chain = (lambda: [ops[0]() for _ in range(N)]) if name in ("noop", "wait", "noop again") else None
        if name == "trigger":  # N triggers, one wait at the end (it is needed to finish the pushes)
            chain = lambda: ([trig() for _ in range(N)], wait())  # noqa: E731
        if name == "trigger+wait":
            chain = lambda: [(trig(), wait()) for _ in range(N)]  # noqa: E731
        chain()
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        chain()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e6 / N)
        ttnn.release_trace(dev, tid)
        res[name] = statistics.median(ts)
        logger.info(f"[opcost] {name}: {res[name]:.2f} us per op (or pair), median of {REPS} replays of {N}")
    st = [ex.read_drisc_l1(dev, c, sync_addr, 2) for c in cores]
    logger.info(f"[opcost] DRISC (taken, done): {st}")
    ttnn.generic_op([x, x], prog("drisc_reload_trigger.cpp", [3, GO, len(xys)], xys))
    ttnn.synchronize_device(dev)
    ex.wait_resident_drisc_kernel(kern)
    logger.info(f"[opcost] {res}")
