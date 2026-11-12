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

GEMMA4_LMHEAD_L1_RELOAD_DRISC=1 moves the reload off the command queue (hidden reload, off by default): a
resident kernel on the DRAM cores' RISC-Vs (kernels/drisc_reload_resident.cpp), launched once at
registration outside the command queue, pushes the snapshot into the dirty cores while the Tensix ops after
the drafter's global SDPA run. A 1-core trigger op right after that SDPA starts it; a 1-core wait op in place
of the reload op blocks until all DRISCs report done. Any bracket entered while a push is in flight waits for
it first, since its op's CBs may overlap the bytes being written. The kernel is stopped before the device
closes (`ttnn.close_mesh_device` is wrapped). GEMMA4_LMHEAD_L1_RELOAD_DRISC_GBPS caps the push's aggregate rate
(default 0 = unpaced), so it spreads over the window instead of crowding the first ops after the trigger;
GEMMA4_LMHEAD_L1_RELOAD_DRISC_CHUNK sets the DRISC DMA chunk in 2 KB pages (default 16).

GEMMA4_LMHEAD_L1_RELOAD_CEIL (default 802,816: the captured global-SDPA CB end 802,112, rounded up to a page) bounds the snapshot
and the reload: everything from S up to it is reloaded. Every bracket asserts that its op's highest
CB end, read from the op's program, does not exceed it.
"""

import os
import re

from loguru import logger

import ttnn

ENABLED = os.getenv("GEMMA4_LMHEAD_L1_RELOAD", "0") == "1"
CEILING = int(os.getenv("GEMMA4_LMHEAD_L1_RELOAD_CEIL", "802816"))
ROPE = os.getenv("GEMMA4_LMHEAD_L1_RELOAD_ROPE", "0") == "1"
DRISC = os.getenv("GEMMA4_LMHEAD_L1_RELOAD_DRISC", "0") == "1"
GO_STREAM = 15  # DRISC stream whose counter the trigger raises (16 DRISC streams)
CMD_SMALL, CMD_ALL, CMD_QUIT = 1, 2, 3
# Paced push (off by default): aggregate rate cap over all DRISCs, GB/s (0 = unpaced), and DMA chunk in pages.
DRISC_GBPS = float(os.getenv("GEMMA4_LMHEAD_L1_RELOAD_DRISC_GBPS", "0"))
DRISC_CHUNK = int(os.getenv("GEMMA4_LMHEAD_L1_RELOAD_DRISC_CHUNK", "16"))
AICLK_HZ = 1.35e9  # DRISC wall-clock ticks per second (the device profiler converts its timestamps at AICLK)
WAIT_TIMEOUT = 1_000_000  # waiter polls (two NoC round trips each) before it gives up instead of hanging the device
PAGE = 2048
# The small reload core set: the block of cores the drafter global SDPA's CBs cover, "<x>x<y>" from (0, 0).
SMALL = tuple(int(v) for v in os.getenv("GEMMA4_LMHEAD_L1_RELOAD_SMALL", "8x4").split("x"))
KDIR = "models/demos/gemma4/tt/assistant/kernels/"

_st = None  # the registered anchor: the head, or the lowest backbone pin (one drafter model per process)


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
    if not ENABLED:
        return
    _register(owner, getattr(owner, attr), [(owner, attr)], device)


def _register(root, head, refs, device):
    """Snapshot `[S, ceiling)` of the anchor `head` (referenced by `refs`) on its shard cores; find the extras."""
    global _st
    mc = head.memory_config()
    grid = mc.shard_spec.grid
    cores = [(c.x, c.y) for c in ttnn.corerange_to_cores(grid, row_wise=True)]
    s_addr = head.buffer_address()
    if s_addr >= CEILING:
        logger.info(f"[lm-head-reload] head at {s_addr} >= ceiling {CEILING}: nothing to reload, knob idle")
        return
    st = _State()
    st.anchor, st.device = {"tensor": head, "refs": refs}, device
    st.shape, st.dtype, st.layout, st.mc = head.shape, head.dtype, head.layout, mc
    st.S, st.cores, st.index = s_addr, cores, {c: i for i, c in enumerate(cores)}
    st.pages = -(-(CEILING - s_addr) // PAGE)
    st.dirty = set()
    st.in_drafter = False
    st.footprints = {}  # bracket key -> (dirty cores, highest CB end); see bracket
    # Reload core sets are snapped to two fixed sets so the trace sees a fixed set of programs: the
    # drafter global SDPA's 8x4 block (it also covers the split-logits concat on (0,0)), else all cores.
    st.small = [c for c in cores if c[0] < SMALL[0] and c[1] < SMALL[1]]
    st.extras = []  # backbone pins are not ported (l1w record 9.4, 9.5)
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
    st.inflight = None
    if DRISC:
        _drisc_start(st, head)
    logger.info(
        f"[lm-head-reload] head at S={s_addr} on {len(cores)} cores; reloads [S, {CEILING}) = {st.pages} pages "
        f"({st.pages * PAGE} B/core) from a {len(cores) * st.pages * PAGE} B DRAM snapshot"
    )


def _one_core_program(kernel, core, compile_args, runtime_args, noc, cbs=()):
    rt = ttnn.RuntimeArgs()
    rt[core[0]][core[1]] = runtime_args
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + kernel,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_crs([core]),
        compile_time_args=compile_args,
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=noc),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=list(cbs))


def _drisc_trigger(cmd):
    head = _st.anchor["tensor"]
    prog = _one_core_program(
        "drisc_reload_trigger.cpp",
        _st.waiter,
        [cmd, GO_STREAM, len(_st.drisc_xy)],
        _st.drisc_xy,
        ttnn.NOC.RISCV_0_default,  # NOC0: measured, one inline write over NOC1 adds 3 to a DRISC stream counter
    )
    ttnn.generic_op([head, head], prog)


def _drisc_wait():
    head = _st.anchor["tensor"]
    scratch = ttnn.CBDescriptor(
        total_size=32 * len(_st.drisc_xy),
        core_ranges=_crs([_st.waiter]),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=32)],
    )
    prog = _one_core_program(
        "drisc_reload_wait.cpp",
        _st.waiter,
        [len(_st.drisc_xy), GO_STREAM, _st.drisc_sync_addr, WAIT_TIMEOUT],
        _st.drisc_xy,
        ttnn.NOC.RISCV_0_default,  # NOC0 reaches the DRISCs' stream-mode NIU0: their L1 and registers
        [scratch],
    )
    ttnn.generic_op([head, head], prog)
    _st.inflight = None


def _drisc_start(st, head):
    """Launch the resident DRISC reload kernel on every DRISC sender and warm up its trigger / wait ops."""
    ex = ttnn._ttnn.operations.experimental
    dev = st.device
    senders = ex.drisc_senders(dev)
    noc1 = set()  # every sender writes on NoC0 (the firmware keeps NIU 0 in stream mode)
    base, size = ex.drisc_l1_unreserved(dev)
    chunk = min(DRISC_CHUNK, size // (2 * PAGE))
    banks = sorted({b for b, _, _ in senders})

    def xy(c):
        v = dev.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        return (v.x << 16) | v.y

    st.waiter = st.cores[-1]  # the core that runs the trigger and wait ops
    # Each bank's two DRISCs push that bank's pages of alternate cores of a set.
    by_bank = {b: [s for s in senders if s[0] == b] for b in banks}
    rts = {}
    for b in banks:
        for e, (_, logical, _) in enumerate(by_bank[b]):
            lists = [
                [c for k, c in enumerate(core_set) if k % len(by_bank[b]) == e] for core_set in (st.small, st.cores)
            ]
            rt = [st.snap.buffer_address(), len(banks), b, st.pages, st.S, base, len(lists[0]), len(lists[1])]
            rt.append(1 if (logical.x, logical.y) in noc1 else 0)  # the NoC this DRISC writes on
            for lst in lists:
                for c in lst:
                    rt += [st.index[c], xy(c)]
            rts[(logical.x, logical.y)] = rt
    # bit 31: the trigger and wait ops reach this DRISC over NoC1 (its NIU0 stays in NOC2AXI mode)
    st.drisc_xy = [((l.x, l.y) in noc1) << 31 | (v.x << 16) | v.y for _, l, v in senders]
    cores = [ttnn.CoreCoord(*k) for k in rts]
    st.drisc_cores, st.drisc_sync_addr = cores, base + 2 * chunk * PAGE  # (taken, done) after the DMA buffers
    # Paced: each DRISC issues one page per `pace` ticks, so the 16 together stay under DRISC_GBPS.
    pace = round(len(cores) * PAGE * AICLK_HZ / (DRISC_GBPS * 1e9)) if DRISC_GBPS > 0 else 0
    st.drisc = ex.launch_resident_drisc_kernel(
        dev,
        KDIR + "drisc_reload_resident.cpp",
        cores,
        [PAGE, chunk, GO_STREAM, pace],
        [rts[(c.x, c.y)] for c in cores],
    )
    # Build the trigger and wait programs now, outside any trace capture; reloading is idempotent.
    for cmd in (CMD_ALL, CMD_SMALL):
        _drisc_trigger(cmd)
        _drisc_wait()
    ttnn.synchronize_device(dev)

    orig_close = ttnn.close_mesh_device

    def close_mesh_device(mesh_device, *args, **kwargs):
        drisc_stop()
        return orig_close(mesh_device, *args, **kwargs)

    ttnn.close_mesh_device = close_mesh_device
    logger.info(
        f"[lm-head-reload] DRISC reload resident on {len(cores)} DRAM cores ({len(banks)} banks, "
        f"{chunk}-page DMA chunks, pace {pace} ticks/page ({DRISC_GBPS or 'unpaced'} GB/s cap), "
        f"DRISC L1 [{base}, {base + size})), waiter core {st.waiter}"
    )


def drisc_stop():
    """Stop the resident DRISC kernel (before the device closes). Idempotent."""
    st = _st
    if st is None or getattr(st, "drisc", None) is None:
        return
    if st.inflight is not None:
        _drisc_wait()
    ex = ttnn._ttnn.operations.experimental
    ticks = [ex.read_drisc_l1(st.device, c, st.drisc_sync_addr, 3)[2] for c in st.drisc_cores]
    logger.info(
        f"[lm-head-reload] DRISC last push: slowest DRISC {max(ticks)} ticks ({max(ticks) / AICLK_HZ * 1e6:.1f} us), "
        f"fastest {min(ticks)} ticks"
    )
    _drisc_trigger(CMD_QUIT)
    ttnn.synchronize_device(st.device)
    ttnn._ttnn.operations.experimental.wait_resident_drisc_kernel(st.drisc)
    st.drisc = None
    logger.info("[lm-head-reload] DRISC reload kernel stopped")


_CORE_RANGE = re.compile(r"\[(\d+)-(\d+)\s*-\s*(\d+)-(\d+)\]")  # "[x1-y1 - x2-y2]", x first


def _cb_reach(graph, s_addr):
    """From a ttnn graph capture: the cores whose program-local CBs end above `s_addr`, the highest CB end, and
    each such core's highest CB end."""
    dirty, top, ends = set(), 0, {}
    for node in graph:
        p = node.get("params", {})
        if node.get("node_type") != "circular_buffer_allocate" or p.get("globally_allocated") != "0":
            continue
        end = int(p["address"]) + int(p["size"])
        if end <= s_addr:
            continue
        top = max(top, end)
        for x1, y1, x2, y2 in _CORE_RANGE.findall(p.get("core_range_set", "")):
            for c in ((x, y) for y in range(int(y1), int(y2) + 1) for x in range(int(x1), int(x2) + 1)):
                dirty.add(c)
                ends[c] = max(ends.get(c, 0), end)
    return dirty, top, ends


def _set_refs(e, t):
    for holder, key in e["refs"]:
        if isinstance(holder, (dict, list)):
            holder[key] = t
        else:
            object.__setattr__(holder, key, t)
    e["tensor"] = t


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
        if _st.inflight is not None:
            # The op's CBs may overlap bytes a DRISC is still writing: finish the push first.
            _drisc_wait()
            logger.warning(f"[lm-head-reload] bracket '{self.tag}' entered during a DRISC reload; waited for it")
        _st.anchor["tensor"].deallocate(True)
        for e in _st.extras:
            e["tensor"].deallocate(True)
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
            dirty, top, ends = _cb_reach(graph, _st.S)
            if top > CEILING:
                for node in graph:
                    if node.get("node_type") == "circular_buffer_allocate":
                        logger.info(f"[lm-head-reload] {self.tag}: static L1 region {node.get('params')}")
                raise RuntimeError(
                    f"[lm-head-reload] {self.tag}: CBs reach {top} > reload ceiling {CEILING} "
                    f"(GEMMA4_LMHEAD_L1_RELOAD_CEIL); the reload would leave [{CEILING}, {top}) overwritten"
                )
            _st.footprints[self.key] = (frozenset(c for c in dirty if c in _st.index), top)
            logger.info(
                f"[lm-head-reload] bracket '{self.tag}': CBs reach into the head on {len(dirty)} cores, "
                f"highest CB end {top} (learned from the program; cached for this input signature)"
            )
            for e in _st.extras:
                over = {c: ends[c] - a for c, a in e["addrs"].items() if ends.get(c, 0) > a}
                logger.info(
                    f"[lm-head-reload] bracket '{self.tag}': CBs overlap extra {e['name']} on {len(over)} cores"
                    + (f", by {min(over.values())}-{max(over.values())} B" if over else "")
                )
        elif exc_type is not None:
            return False
        h = ttnn.allocate_tensor_on_device(_st.shape, _st.dtype, _st.layout, _st.device, _st.mc)
        if h.buffer_address() != _st.S:
            raise RuntimeError(
                f"[lm-head-reload] head re-allocated at {h.buffer_address()} after {self.tag}, expected {_st.S}: "
                "something allocated L1 inside the bracket"
            )
        _set_refs(_st.anchor, h)
        _st.dirty.update(_st.footprints[self.key][0])
        if getattr(_st, "drisc", None) is not None and _st.in_drafter and self.tag == "global SDPA" and _st.dirty:
            # The drafter's last head-overwriting op before its lm_head: start the push now.
            _st.inflight = _st.small if _st.dirty <= set(_st.small) else _st.cores
            _drisc_trigger(CMD_SMALL if _st.inflight is _st.small else CMD_ALL)
            _st.dirty.clear()
        return False


def drafter_step():
    """Mark the start of a drafter step: its global SDPA is the last op before the lm_head that overwrites it."""
    if _st is not None:
        _st.in_drafter = True


def reload():
    """Restore `[S, ceiling)` on every dirty core. Call right before the lm_head reads the head."""
    if _st is None:
        return
    _st.in_drafter = False
    if _st.inflight is not None:
        cores = _st.inflight
        _drisc_wait()
        if len(cores) not in _st.__dict__.setdefault("logged", set()):
            _st.logged.add(len(cores))
            logger.info(f"[lm-head-reload] DRISC reload on {len(cores)} dirty cores (waited)")
        if not _st.dirty:
            return
    if not _st.dirty:
        return
    cores = _st.small if _st.dirty <= set(_st.small) else _st.cores
    head = _st.anchor["tensor"]
    ttnn.generic_op([_st.snap, head], _page_program("dram_reload_pages.cpp", _st.snap, cores, _st.S))
    if len(cores) not in _st.__dict__.setdefault("logged", set()):
        _st.logged.add(len(cores))
        logger.info(f"[lm-head-reload] reload on {len(cores)} dirty cores")
    _st.dirty.clear()
