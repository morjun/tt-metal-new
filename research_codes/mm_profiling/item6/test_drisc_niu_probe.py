# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which DRISC endpoints' NIU counters see Tensix DRAM reads, and at what per-bank rate (gemma4-12b record §9.2, item 6b).

Launches item6/drisc_niu_probe.cpp on every DRISC sender, starts sampling, runs a DRAM-fed 1D-mcast matmul of
gate_proj's shape (M 32, K 1024, N 8192, bf16, the tuner's 64-core config) a few times, quits the probe and reads
every DRISC's records. Reports per DRISC (bank, virtual coordinate) the counter deltas over the run on both NoCs and,
on NOC1, the delivered rate while it serves reads. Run it on the overlay build (setup_overlay in run_lmhead_e2e.py):

    TT_METAL_RUNTIME_ROOT=$PWD TT_METAL_HOME=$PWD PYTHONPATH=/tmp/lmhead_ttnn_overlay_e2e:$PWD TT_VISIBLE_DEVICES=0 \
        pytest -q -s research_codes/mm_profiling/item6/test_drisc_niu_probe.py

Scope: the DRISC senders are each bank's NOC1 endpoint and its unused endpoint; the NOC0 endpoint (syseng-owned, no
DRISC firmware) is not among them, so reads on NOC0 are never observed. Record §9.2 uses no result of this test.
"""

import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.matmul_tuning import derive_decode_1d_config

KDIR = "research_codes/mm_profiling/"
GO = 15
TRIG_CORE = (10, 9)
INTERVAL_TICKS = (
    4050  # 3 us at 1350 MHz: the buffer (~3,680 records) then spans ~11 ms, past the ~5 ms start-to-quit window
)
REPEATS = 5


def crs(c):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c))])


def trigger(dev, x, cmd, xys):
    rt = ttnn.RuntimeArgs()
    rt[TRIG_CORE[0]][TRIG_CORE[1]] = xys
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + "kernels/drisc_reload_trigger.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs(TRIG_CORE),
        compile_time_args=[cmd, GO, len(xys)],
        runtime_args=rt,
        # NOC0: an inline write over NOC1 adds 3 to a DRISC stream counter (lm_head_reload.py)
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    ttnn.generic_op([x, x], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))
    ttnn.synchronize_device(dev)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_drisc_niu_probe(mesh_device):
    dev = mesh_device
    ex = ttnn._ttnn.operations.experimental
    senders = ex.drisc_senders(dev)
    base, size = ex.drisc_l1_unreserved(dev)
    max_records = (size // 4 - 4) // 6
    cores = [l for _, l, _ in senders]
    xys = [(v.x << 16) | v.y for _, _, v in senders]
    logger.info(f"[niu] senders (bank, virtual): {[(b, (v.x, v.y)) for b, _, v in senders]}; {max_records} records")

    x = ttnn.from_torch(torch.randn(1, 1, 32, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(
        torch.randn(1, 1, 1024, 8192),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pc = derive_decode_1d_config(32, 1024, 8192)
    ttnn.linear(x, w, program_config=pc)  # compile outside the sampled window
    ttnn.synchronize_device(dev)

    kern = ex.launch_resident_drisc_kernel(
        dev, KDIR + "item6/drisc_niu_probe.cpp", cores, [GO, max_records, INTERVAL_TICKS], [[base] for _ in cores]
    )
    time.sleep(1.0)  # at 0.2 s some DRISCs had not started: they missed the start command
    logger.info(f"[niu] state before start: {[ex.read_drisc_l1(dev, c, base, 2)[1] for c in cores]}")
    trigger(dev, x, 1, xys)  # start
    for _ in range(REPEATS):
        ttnn.linear(x, w, program_config=pc)
    ttnn.synchronize_device(dev)
    trigger(dev, x, 3, xys)  # quit
    time.sleep(0.05)

    weight_bytes = 1024 * 8192 * 2 * REPEATS
    in0_bytes = 32 * 1024 * 2 * REPEATS
    noc0_total = noc1_total = 0
    for (bank, _, v), c in zip(senders, cores):
        head = ex.read_drisc_l1(dev, c, base, 4)
        n = head[0]
        recs = ex.read_drisc_l1(dev, c, base + 16, 6 * n) if n else []
        rows = [recs[6 * i : 6 * i + 6] for i in range(n)]
        if len(rows) < 2:
            logger.info(f"[niu] bank {bank} ({v.x},{v.y}): {n} records, state {head[1]}")
            continue
        d = [(rows[-1][k] - rows[0][k]) % (1 << 32) for k in range(1, 5)]
        # NOC1 data words per interval: delivered rate while this endpoint serves reads
        steps = [((b[2] - a[2]) % (1 << 32), (b[0] - a[0]) / 1350) for a, b in zip(rows, rows[1:])]
        busy = [(w, us) for w, us in steps if w]
        rate = f"; NOC1 busy {sum(us for _, us in busy):.0f} us" if busy else ""
        if busy:
            rate += (
                f" at {64 * sum(w for w, _ in busy) / sum(us for _, us in busy) / 1e3:.2f} GB/s (64 B words),"
                f" peak {max(64 * w / us for w, us in busy) / 1e3:.2f} GB/s per {INTERVAL_TICKS / 1350:.0f} us"
            )
        noc0_total += d[0]
        noc1_total += d[1]
        logger.info(
            f"[niu] bank {bank} ({v.x},{v.y}): {n} records over {(rows[-1][0] - rows[0][0]) / 1350:.0f} us, state "
            f"{head[1]}; data words sent NOC0 {d[0]} NOC1 {d[1]}; read requests NOC0 {d[2]} NOC1 {d[3]}{rate}"
        )
    logger.info(
        f"[niu] totals over all DRISCs: NOC0 {noc0_total} words, NOC1 {noc1_total} words; weight {weight_bytes} B,"
        f" in0 {in0_bytes} B; NOC1 B/word vs in0 {in0_bytes / max(noc1_total, 1):.1f},"
        f" vs weight + in0 {(weight_bytes + in0_bytes) / max(noc1_total, 1):.1f}"
    )
    if noc0_total:
        logger.info(
            f"[niu] NOC0 data words over all DRISCs {noc0_total}: {weight_bytes / noc0_total:.1f} B/word if they are the weight reads"
        )
    else:
        logger.info(
            "[niu] no DRISC endpoint's NOC0 slave counter moved: the weight reads are served by other endpoints"
        )
    if all(ex.read_drisc_l1(dev, c, base, 2)[1] == 2 for c in cores):
        ex.wait_resident_drisc_kernel(kern)
        logger.info("[niu] probe stopped")
    else:
        logger.error("[niu] not every DRISC finished; not waiting")
