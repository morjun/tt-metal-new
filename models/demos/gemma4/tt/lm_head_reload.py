# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Research knob: pin a drafter lm_head slice that reaches below the global-SDPA CB ceiling.

Off unless GEMMA4_LMHEAD_L1_RELOAD=1 (gemma4-12b MEASUREMENT_RECORD §5.5.14, Gate 5).

The pinned slice (GEMMA4_LMHEAD_L1_COLS) starts at S. Some ops' program-local CBs reach above S
(the global SDPAs up to 801,024; the split-logits concat on core (0,0)). Around each such op:
1. `bracket(tag)` frees the head before the op. This is host-side bookkeeping only; L1 is not
   touched, and the op's CBs pass the host-side CB clash check.
2. After the op, it re-allocates the head at the same address (asserted) and marks dirty the cores
   whose CBs, as allocated by the op's program, end above S (read once per input signature from a
   ttnn graph capture of an eager call, then cached; see `bracket`).
3. Before the drafter lm_head, `reload()` copies `[S, ceiling)` (whole 2,048 B pages) back from a
   DRAM snapshot taken once at pin time, on the dirty cores only.

The head stays one tensor, so it is still a single slice matmul. Verified standalone and under trace
in research_codes/mm_profiling/test_pin_free_realloc_trace.py (Gate 4).

GEMMA4_LMHEAD_L1_RELOAD_ROPE=1 also brackets the global layer's batch-1 rotary embedding (CBs to 512,896 on
one core), which a pin of 59,840 cols or more (S <= 458,752) reaches; off by default.

GEMMA4_LMHEAD_L1_RELOAD_CEIL (default 801,024, the captured global-SDPA ceiling) bounds the snapshot
and the reload: everything from S up to it is reloaded. Every bracket asserts that its op's highest
CB end, read from the op's program, does not exceed it.
"""

import os
import re

from loguru import logger

import ttnn

ENABLED = os.getenv("GEMMA4_LMHEAD_L1_RELOAD", "0") == "1"
CEILING = int(os.getenv("GEMMA4_LMHEAD_L1_RELOAD_CEIL", "801024"))
ROPE = os.getenv("GEMMA4_LMHEAD_L1_RELOAD_ROPE", "0") == "1"
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
    st.footprints = {}  # bracket key -> (dirty cores, highest CB end); see bracket
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


_CORE_RANGE = re.compile(r"\[(\d+)-(\d+)\s*-\s*(\d+)-(\d+)\]")  # "[x1-y1 - x2-y2]", x first


def _cb_reach(graph, s_addr):
    """From a ttnn graph capture: the cores whose program-local CBs end above `s_addr`, and the highest CB end."""
    dirty, top = set(), 0
    for node in graph:
        p = node.get("params", {})
        if node.get("node_type") != "circular_buffer_allocate" or p.get("globally_allocated") != "0":
            continue
        end = int(p["address"]) + int(p["size"])
        if end <= s_addr:
            continue
        top = max(top, end)
        for x1, y1, x2, y2 in _CORE_RANGE.findall(p.get("core_range_set", "")):
            dirty.update((x, y) for y in range(int(y1), int(y2) + 1) for x in range(int(x1), int(x2) + 1))
    return dirty, top


class bracket:
    """Free the head around one op whose CBs may reach into it, then re-allocate it at the same address.

    The op's CB footprint is read from its own program, not passed in. On the first call for a given
    `key_tensors` signature (shapes + memory configs, a proxy for "same program"), the op runs inside a
    ttnn graph capture (nesting is supported: GraphTracker keeps a processor stack). From the circular
    buffers the program allocated, the bracket takes the cores whose CBs end above S, which it marks
    dirty, and the highest CB end, which must not exceed the reload ceiling or bytes would be overwritten
    that reload() never restores. The footprint is cached per signature.

    Graph capture cannot run inside a trace capture: it serialises op arguments, and printing a tensor
    list reads device data, which fails with "Reads are not supported during trace capture". So every
    signature must be seen once eagerly first. ttnn imposes the same rule on programs ("Cannot load new
    binaries during trace capture"), and the demo's compile run meets both. A signature first seen
    inside a capture fails loudly; it is never silently wrong. `tag` only names the op in logs, errors
    and the cache key.
    """

    def __init__(self, tag, *key_tensors):
        self.tag = tag
        self.key = (tag,) + tuple((tuple(t.shape), str(t.memory_config())) for t in key_tensors)
        self.capture = False

    def __enter__(self):
        if _st is None:
            return self
        getattr(_st.owner, _st.attr).deallocate(True)
        if self.key not in _st.footprints:
            self.capture = True
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        return self

    def __exit__(self, exc_type, *exc):
        if _st is None:
            return False
        if self.capture:
            graph = ttnn.graph.end_graph_capture()
            if exc_type is not None:
                return False
            dirty, top = _cb_reach(graph, _st.S)
            if top > CEILING:
                raise RuntimeError(
                    f"[lm-head-reload] {self.tag}: CBs reach {top} > reload ceiling {CEILING} "
                    f"(GEMMA4_LMHEAD_L1_RELOAD_CEIL); the reload would leave [{CEILING}, {top}) overwritten"
                )
            _st.footprints[self.key] = (frozenset(c for c in dirty if c in _st.index), top)
            logger.info(
                f"[lm-head-reload] bracket '{self.tag}': CBs reach into the head on {len(dirty)} cores, "
                f"highest CB end {top} (learned from the program; cached for this input signature)"
            )
        elif exc_type is not None:
            return False
        h = ttnn.allocate_tensor_on_device(_st.shape, _st.dtype, _st.layout, _st.device, _st.mc)
        if h.buffer_address() != _st.S:
            raise RuntimeError(
                f"[lm-head-reload] head re-allocated at {h.buffer_address()} after {self.tag}, expected {_st.S}: "
                "something allocated L1 inside the bracket"
            )
        setattr(_st.owner, _st.attr, h)
        _st.dirty.update(_st.footprints[self.key][0])
        return False


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
