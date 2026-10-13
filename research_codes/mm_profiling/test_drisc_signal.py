# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DIAG: Tensix -> DRISC go-counter signalling alone (drisc_reload_resident.cpp built with DIAG_SIGNAL).

The resident kernel only records the counts it reads (sync[2..9] past its DMA buffers) and quits on a
count >= 3. Sends single increments over NOC1 and NOC0, to all DRISCs and to one, and reads every DRISC's
history after each.
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
    rts = [[0, 8, b, 168, 458752, base, 0, 0] for b, _, _ in senders]
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
