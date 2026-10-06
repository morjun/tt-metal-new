# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""L1 -> L1 mover programs for the "moving pinned weights" study (gemma4-12b MEASUREMENT_RECORD §5.5.14).

Shared by test_pin_move_cost.py (standalone timing) and models/demos/gemma4/tt/pin_move_dummy.py
(in-situ timing inside the demo trace). Kernel: kernels/l1_pull_chunks.cpp.

generic_op hashes kernel core ranges and the NUMBER of per-core runtime-arg entries, not their
length; a cached program re-applies new args truncated to its own length. So every program here
uses one fixed per-core width (MAX_PIECES pieces) - a mismatch made the kernel read garbage NoC
coordinates and hung the board twice.
"""

import ttnn

KERNEL = "research_codes/mm_profiling/kernels/l1_pull_chunks.cpp"
GRID_X, GRID_Y = 11, 10
MAX_PIECES = 8  # fixed runtime-arg capacity per core (1 + 5 * MAX_PIECES args) for every program


def core_set(pred):
    return [(x, y) for y in range(GRID_Y) for x in range(GRID_X) if pred(x, y)]


def crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def l1_sharded_spec(cores, row_bytes):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(cores), [1, row_bytes // 2], ttnn.ShardOrientation.ROW_MAJOR),
    )


def l1_sharded(device, cores, row_bytes):
    """Uninitialised bf16 row-major tensor with `row_bytes` on each of `cores`."""
    shape = ttnn.Shape([1, 1, len(cores), row_bytes // 2])
    return ttnn.allocate_tensor_on_device(
        shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, l1_sharded_spec(cores, row_bytes)
    )


def share_bytes(n_src, src_bytes, n_dst):
    """Equal per-parking-core share of the concatenated source stream, 64 B aligned."""
    return ((n_src * src_bytes + n_dst - 1) // n_dst + 63) // 64 * 64


def plan_chunks(n_src, src_bytes, n_dst, dst_bytes):
    """Split the concatenated source stream into equal contiguous dst shares.
    Returns {dst_index: [(src_index, src_off, dst_off, size)]}."""
    plan = {j: [] for j in range(n_dst)}
    total = n_src * src_bytes
    for j in range(n_dst):
        lo, hi = j * dst_bytes, min((j + 1) * dst_bytes, total)
        pos = lo
        while pos < hi:
            i, off = divmod(pos, src_bytes)
            size = min(src_bytes - off, hi - pos)
            plan[j].append((i, off, pos - lo, size))
            pos += size
    return plan


def mover(puller_cores, pieces_by_core, both_riscs):
    """generic_op program: each puller core pulls its (noc_x, noc_y, src, dst, size) pieces."""
    kernels = []
    halves = [0, 1] if both_riscs else [None]
    for half in halves:
        per_core = {}
        for (x, y), pieces in pieces_by_core.items():
            args = []
            for nx, ny, src, dst, size in pieces:
                if half is None:
                    args.append((nx, ny, src, dst, size))
                else:  # split every piece into two halves (64 B aligned) across the two RISCs / NoCs
                    a = (size // 2) // 64 * 64
                    part = (nx, ny, src, dst, a) if half == 0 else (nx, ny, src + a, dst + a, size - a)
                    if part[4]:
                        args.append(part)
            per_core[(x, y)] = args
        assert max(len(a) for a in per_core.values()) <= MAX_PIECES, "raise MAX_PIECES"
        width = 1 + 5 * MAX_PIECES
        rt = ttnn.RuntimeArgs()
        for (x, y), args in per_core.items():
            flat = [len(args)] + [v for p in args for v in p]
            rt[x][y] = flat + [0] * (width - len(flat))
        cfg = ttnn.ReaderConfigDescriptor() if half in (None, 0) else ttnn.WriterConfigDescriptor()
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs(puller_cores),
                compile_time_args=[MAX_PIECES],
                runtime_args=rt,
                config=cfg,
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])


def round_trip_programs(device, src_cores, dst_cores, src_bytes, s_addr, p_addr, b_addr, both_riscs):
    """(out, back) programs: parking cores pull the source shards (out) into equal shares at p_addr;
    source cores pull them back (back) to b_addr."""
    n_src, n_dst = len(src_cores), len(dst_cores)
    dst_bytes = share_bytes(n_src, src_bytes, n_dst)
    noc = {c: device.worker_core_from_logical_core(ttnn.CoreCoord(*c)) for c in src_cores + dst_cores}
    plan = plan_chunks(n_src, src_bytes, n_dst, dst_bytes)
    out_pieces = {
        dst_cores[j]: [
            (noc[src_cores[i]].x, noc[src_cores[i]].y, s_addr + so, p_addr + do, sz) for i, so, do, sz in pcs
        ]
        for j, pcs in plan.items()
        if pcs
    }
    back_pieces = {c: [] for c in src_cores}
    for j, pcs in plan.items():
        for i, so, do, sz in pcs:
            back_pieces[src_cores[i]].append((noc[dst_cores[j]].x, noc[dst_cores[j]].y, p_addr + do, b_addr + so, sz))
    return mover(list(out_pieces), out_pieces, both_riscs), mover(src_cores, back_pieces, both_riscs)
