# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DIAG: Tensix -> DRISC go-counter signalling alone (drisc_reload_resident.cpp built with DIAG_SIGNAL).

The resident kernel only records the counts it reads (sync[2..9] past its DMA buffers) and quits on a
count >= 3. test_drisc_signal sends single increments over NOC1 and NOC0, to all DRISCs and to one, and reads
every DRISC's history after each.

test_drisc_signal_ep_noc1 (gemma4-12b record §9.2e, plan v4 step 1b): the DRISC on bank 0's (0, 0) endpoint
runs on NoC1 (stream mode on its NIU1; its NIU0 stays NOC2AXI for TT_MM1D_IN1_BANK0_XY reads), the others on
NoC0. The production trigger and wait kernels reach it over NoC1 (bit 31 of its word). Checks: every DRISC
counts each command once, and the wait op returns without timing out.
"""

import time

from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

KDIR = "research_codes/mm_profiling/kernels/"
GO, PAGE, CHUNK = 15, 2048, 16
TRIG_CORE = (10, 9)


def crs(c):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c))])


def trigger(dev, x, cmd, xys, noc):
    rt = ttnn.RuntimeArgs()
    rt[TRIG_CORE[0]][TRIG_CORE[1]] = xys
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + "drisc_reload_trigger.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs(TRIG_CORE),
        compile_time_args=[cmd, GO, len(xys)],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=noc),
    )
    ttnn.generic_op([x, x], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))
    ttnn.synchronize_device(dev)
    time.sleep(0.05)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_drisc_signal(mesh_device):
    dev = mesh_device
    ex = ttnn._ttnn.operations.experimental
    senders = ex.drisc_senders(dev)
    base, size = ex.drisc_l1_unreserved(dev)
    sync_addr = base + 2 * CHUNK * PAGE
    logger.info(f"[sig] senders (bank, logical, virtual): {[(b, (l.x, l.y), (v.x, v.y)) for b, l, v in senders]}")
    cores = [l for _, l, _ in senders]
    rts = [[0, 8, b, 168, 458752, base, 0, 0, 0] for b, _, _ in senders]  # last: the DRISC writes on NoC0
    x = ttnn.rand([1, 1, 32, 32], device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    xys = [(v.x << 16) | v.y for _, _, v in senders]
    kern = ex.launch_resident_drisc_kernel(
        dev, KDIR + "drisc_reload_resident.cpp", cores, [PAGE, CHUNK, GO, 0], rts, {"DIAG_SIGNAL": "1"}
    )
    time.sleep(0.2)

    def status(tag):
        hists = []
        for c in cores:
            w = ex.read_drisc_l1(dev, c, sync_addr, 10)
            hists.append(w[2 : 2 + min(w[0], 8)])
        logger.info(f"[sig] {tag}:\n  " + "\n  ".join(f"({c.x},{c.y}) {h}" for c, h in zip(cores, hists)))
        return hists

    status("after launch")
    trigger(dev, x, 1, xys, ttnn.NOC.RISCV_1_default)
    status("after +1 to all over NOC1")
    trigger(dev, x, 1, xys, ttnn.NOC.RISCV_0_default)
    status("after +1 to all over NOC0")
    trigger(dev, x, 1, xys[:1], ttnn.NOC.RISCV_1_default)
    status(f"after +1 to the first DRISC only {senders[0][2]} over NOC1")
    trigger(dev, x, 3, xys, ttnn.NOC.RISCV_1_default)

    def all_quit(hists):
        return all(any(v >= 3 for v in h) for h in hists)

    rows = status("after +3 (quit) to all over NOC1")
    if not all_quit(rows):
        trigger(dev, x, 3, xys, ttnn.NOC.RISCV_0_default)
        rows = status("after +3 (quit) over NOC0")
    if all_quit(rows):
        ex.wait_resident_drisc_kernel(kern)
        logger.info("[sig] kernel stopped")
    else:
        logger.error("[sig] not every DRISC quit; not waiting")


EP0 = 785  # TT_MM1D_IN1_BANK0_XY of the +ep0 arms: bank 0's (0, 0) endpoint, packed x | y << 6 (virtual)


def run_one(dev, x, kernel, ct, rt, cbs=()):
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + kernel,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs(TRIG_CORE),
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    t = time.perf_counter()
    ttnn.generic_op([x, x], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=list(cbs)))
    ttnn.synchronize_device(dev)
    return time.perf_counter() - t


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_drisc_signal_ep_noc1(mesh_device):
    dev = mesh_device
    ex = ttnn._ttnn.operations.experimental
    senders = ex.drisc_senders(dev)
    base, size = ex.drisc_l1_unreserved(dev)
    sync_addr = base + 2 * CHUNK * PAGE
    cores = [l for _, l, _ in senders]
    # the sender on the endpoint the +ep0 matmuls read bank 0 through (matched as in lm_head_reload._drisc_start)
    on1 = [(v.x, v.y) == (EP0 & 0x3F, EP0 >> 6) for _, _, v in senders]
    logger.info(f"[sig1] senders: {[(b, (l.x, l.y), (v.x, v.y), n) for (b, l, v), n in zip(senders, on1)]}")
    assert sum(on1) == 1, "expected exactly one DRISC on the +ep0 endpoint"
    rts = [[0, 8, b, 168, 458752, base, 0, 0, int(n)] for (b, _, _), n in zip(senders, on1)]
    xys = [(n << 31) | (v.x << 16) | v.y for (_, _, v), n in zip(senders, on1)]
    x = ttnn.rand([1, 1, 32, 32], device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    kern = ex.launch_resident_drisc_kernel(
        dev, KDIR + "drisc_reload_resident.cpp", cores, [PAGE, CHUNK, GO, 0], rts, {"DIAG_SIGNAL": "1"}
    )
    time.sleep(0.2)

    def hists():
        out = []
        for c in cores:
            w = ex.read_drisc_l1(dev, c, sync_addr, 10)
            out.append(w[2 : 2 + min(w[0], 8)])
        return out

    rt = ttnn.RuntimeArgs()
    rt[TRIG_CORE[0]][TRIG_CORE[1]] = xys
    scratch = ttnn.CBDescriptor(
        total_size=32 * len(xys),
        core_ranges=crs(TRIG_CORE),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=32)],
    )
    results = {}
    for cmd in (1, 2):
        run_one(dev, x, "drisc_reload_trigger.cpp", [cmd, GO, len(xys)], rt)
        time.sleep(0.05)
        h = hists()
        logger.info(
            f"[sig1] after cmd {cmd}: "
            + ", ".join(f"({c.x},{c.y}){'*' if n else ''} {v}" for c, n, v in zip(cores, on1, h))
        )
        results[cmd] = h
        # DIAG: taken == done after every command, so the wait op must return at once (timeout = 1e6 polls)
        dt = run_one(dev, x, "drisc_reload_wait.cpp", [len(xys), GO, sync_addr, 1_000_000], rt, [scratch])
        logger.info(f"[sig1] wait op after cmd {cmd}: {dt * 1e3:.2f} ms host time")
        results[f"wait{cmd}"] = dt
    run_one(dev, x, "drisc_reload_trigger.cpp", [3, GO, len(xys)], rt)
    time.sleep(0.05)
    h = hists()
    logger.info("[sig1] after quit: " + ", ".join(f"({c.x},{c.y}) {v}" for c, v in zip(cores, h)))
    if all(any(v >= 3 for v in r) for r in h):
        ex.wait_resident_drisc_kernel(kern)
        logger.info("[sig1] kernel stopped")
    else:
        logger.error("[sig1] not every DRISC quit; not waiting")
    assert all(r == [1] for r in results[1]), f"cmd 1 not counted once everywhere: {results[1]}"
    assert all(r == [1, 2] for r in results[2]), f"cmd 2 not counted once everywhere: {results[2]}"
    assert results["wait1"] < 0.5 and results["wait2"] < 0.5, "wait op ran into its timeout"
