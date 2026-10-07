# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Research knob: pin a drafter lm_head slice that reaches below the global-SDPA CB ceiling.

Off unless GEMMA4_LMHEAD_L1_RELOAD=1 (gemma4-12b MEASUREMENT_RECORD §5.5.14, Gate 5).

The pinned slice (GEMMA4_LMHEAD_L1_COLS) starts at S. Some ops' program-local CBs reach above S
(the global SDPAs up to 801,024; the split-logits concat on core (0,0)). Around each such op:
1. `bracket(cores)` frees the head before the op. This is host-side bookkeeping only; L1 is not
   touched, and the op's CBs pass the host-side CB clash check.
2. After the op, it re-allocates the head at the same address (asserted) and marks the op's cores
   dirty.
3. Before the drafter lm_head, `reload()` copies `[S, ceiling)` (whole 2,048 B pages) back from a
   DRAM snapshot taken once at pin time, on the dirty cores only.

The head stays one tensor, so it is still a single slice matmul. Verified standalone and under trace
in research_codes/mm_profiling/test_pin_free_realloc_trace.py (Gate 4).

GEMMA4_LMHEAD_L1_RELOAD_CEIL (default 801,024) is the highest CB end any bracketed op can reach:
the captured global-SDPA ceiling. Everything from S up to it is reloaded.
"""

import os

from loguru import logger

import ttnn

ENABLED = os.getenv("GEMMA4_LMHEAD_L1_RELOAD", "0") == "1"
CEILING = int(os.getenv("GEMMA4_LMHEAD_L1_RELOAD_CEIL", "801024"))
PAGE = 2048
KDIR = "research_codes/mm_profiling/kernels/"

_st = None  # the registered head (one drafter model per process)


class _State:
    pass


def active():
    return _st is not None


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _page_program(kernel, dram, cores, l1_addr):
    ct = [PAGE] + list(ttnn.TensorAccessorArgs(dram).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for x, y in cores:
        rt[x][y] = [dram.buffer_address(), _st.index[(x, y)] * _st.pages, _st.pages, l1_addr]
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + kernel,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_crs(cores),
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])


def register(owner, attr, device):
    """Take over `owner.<attr>` (the pinned, width-sharded L1 head) and snapshot `[S, ceiling)`."""
    global _st
    if not ENABLED:
        return
    head = getattr(owner, attr)
    mc = head.memory_config()
    grid = mc.shard_spec.grid
    cores = [(c.x, c.y) for c in ttnn.corerange_to_cores(grid, row_wise=True)]
    s_addr = head.buffer_address()
    if s_addr >= CEILING:
        logger.info(f"[lm-head-reload] head at {s_addr} >= ceiling {CEILING}: nothing to reload, knob idle")
        return
    st = _State()
    st.owner, st.attr, st.device = owner, attr, device
    st.shape, st.dtype, st.layout, st.mc = head.shape, head.dtype, head.layout, mc
    st.S, st.cores, st.index = s_addr, cores, {c: i for i, c in enumerate(cores)}
    st.pages = -(-(CEILING - s_addr) // PAGE)
    st.dirty = set()
    # Reload core sets are snapped to two fixed sets so the trace sees a fixed set of programs: the
    # drafter global SDPA's 8x4 block (it also covers the split-logits concat on (0,0)), else all cores.
    st.small = [c for c in cores if c[0] < 8 and c[1] < 4]
    st.snap = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, len(cores) * st.pages, PAGE // 2]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    _st = st
    ttnn.generic_op([head, st.snap], _page_program("dram_snapshot_pages.cpp", st.snap, cores, s_addr))
    # Only two reload programs ever run (see reload()); build both now, outside any trace capture
    # (a program first seen during capture is fatal). Reloading from the snapshot is idempotent.
    for reload_set in (st.small, st.cores):
        ttnn.generic_op([st.snap, head], _page_program("dram_reload_pages.cpp", st.snap, reload_set, s_addr))
    logger.info(
        f"[lm-head-reload] head at S={s_addr} on {len(cores)} cores; reloads [S, {CEILING}) = {st.pages} pages "
        f"({st.pages * PAGE} B/core) from a {len(cores) * st.pages * PAGE} B DRAM snapshot"
    )


class bracket:
    """Free the head around an op whose CBs reach above S on `cores`, then re-allocate it."""

    def __init__(self, cores, tag):
        self.cores, self.tag = cores, tag

    def __enter__(self):
        if _st is not None:
            getattr(_st.owner, _st.attr).deallocate(True)
        return self

    def __exit__(self, *exc):
        if _st is None:
            return False
        h = ttnn.allocate_tensor_on_device(_st.shape, _st.dtype, _st.layout, _st.device, _st.mc)
        if h.buffer_address() != _st.S:
            raise RuntimeError(
                f"[lm-head-reload] head re-allocated at {h.buffer_address()} after {self.tag}, expected {_st.S}: "
                "something allocated L1 inside the bracket"
            )
        setattr(_st.owner, _st.attr, h)
        _st.dirty.update(c for c in self.cores if c in _st.index)
        return False


def grid_cores(gx, gy):
    return [(x, y) for y in range(gy) for x in range(gx)]


def reload():
    """Restore `[S, ceiling)` on every dirty core. Call right before the lm_head reads the head."""
    if _st is None or not _st.dirty:
        return
    cores = _st.small if _st.dirty <= set(_st.small) else _st.cores
    head = getattr(_st.owner, _st.attr)
    ttnn.generic_op([_st.snap, head], _page_program("dram_reload_pages.cpp", _st.snap, cores, _st.S))
    if len(cores) not in _st.__dict__.setdefault("logged", set()):
        _st.logged.add(len(cores))
        logger.info(f"[lm-head-reload] reload on {len(cores)} dirty cores")
    _st.dirty.clear()
