# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM read probe harness v2 (gemma4-12b record §9.2e, plan v3 H1-H3): kernels/dram_read_probe.cpp on chosen cores,
optionally paced by the in0 multicast sender kernels/dram_read_in0_sender.cpp.

Each probe core issues a DRAM read stream (a matmul's in1 strip of an interleaved bf16 tile tensor, or a direct
per-bank stream) and logs its first issue, first word, last word, and per block its issue / completion time and NIU
master counters. One config group per process (PROBE_GROUP), run under `timeout`:

    TT_METAL_RUNTIME_ROOT=$PWD TT_METAL_HOME=$PWD PYTHONPATH=/tmp/lmhead_ttnn_overlay_e2e:$PWD TT_VISIBLE_DEVICES=0 \\
        PROBE_GROUP=gate2 PROBE_OUT=generated/dram_read_probe timeout 600 \\
        pytest -q -s research_codes/mm_profiling/test_dram_read_probe.py

Repeats (plan v3 G3): every config gets 1 warm-up launch (discarded), then PROBE_REPEATS launches; configs are
interleaved (the repeat loop is outer). Positions are reported as logical, translated and physical NoC0 (plan v3 G1).

Groups:
  clock      wall-clock rate: one core spins 2^30 ticks; host wall time of the program
  endpoints  dram_bank_to_noc_xy for both NoCs (plan v3 H3)
  debug_rect 2 paced readers in a row (H2 validation); debug_l: 2 paced readers on a diagonal (idle cores in the
             multicast rectangle)
  gate2      gate v2: (a) fat-K 32 cores and (b) thin-K 64 cores, generator on and off; (c) one core, bank 0, 16 KB
             bursts, depth 3
  e1v3       one core alone at each of the 110 positions: fat-K strip of bank 0..7, thin-K strips of banks {0-3}, {4-7}
  e2v3       one core (PROBE_E2_POS, logical "x,y") alone: access {strided 2 KB, contiguous 2 KB, burst 16 KB / N_b}
             x depth {1, 2, 4} x banks N_b {1, 2, 4, 8}; e2v3_dup: (strided, D 1, N_b 1) only (second process)
Records go to PROBE_OUT/<group>.json (one entry per config and repeat).
"""

import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.matmul_tuning import derive_decode_1d_config

KERNEL = "research_codes/mm_profiling/kernels/dram_read_probe.cpp"
SENDER = "research_codes/mm_profiling/kernels/dram_read_in0_sender.cpp"
NARGS, NARGS_SENDER = 24, 16
TILE_BYTES = 2048
MAX_BLOCKS = 64
REC_HDR = 8
REC_BYTES = 4 * (REC_HDR + 4 * MAX_BLOCKS)
SYNC_BYTES = 512
IN0_BLOCK = 16384  # in0_block_w 8 x one 2 KB tile row
SCRATCH_BYTES = 256 * 1024
CLOCK_MHZ = 1350.0  # AICLK (the clock group measured 1,344.9 MHz including host launch overhead)
REPEATS = int(os.getenv("PROBE_REPEATS", "3"))
GROUP = os.getenv("PROBE_GROUP", "gate2")
OUT = Path(os.getenv("PROBE_OUT", "generated/dram_read_probe"))


def phys(xy):
    """logical -> physical NoC0 (plan v3 G1): x 0-5 -> 1-6, 6-10 -> 11-15; y + 2."""
    x, y = xy
    return (x + 1 if x < 6 else x + 5, y + 2)


def crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def strip_args(Nt, c0, W, k0, n_blocks, iw):
    """Mode 0: a 1D-multicast matmul core's in1 strip (columns c0..c0+W-1, iw K rows per block)."""
    return dict(mode=0, n_blocks=n_blocks, reqs=iw * W, req_bytes=TILE_BYTES, p=(Nt, c0, W, k0))


def bank_args(nbanks, bank0, stride, offset, n_blocks, reqs, req_bytes):
    """Mode 1: request g -> bank (bank0 + g % nbanks) % 8, address offset + (g / nbanks) * stride."""
    return dict(mode=1, n_blocks=n_blocks, reqs=reqs, req_bytes=req_bytes, p=(nbanks, bank0, stride, offset))


def issued_requests(c):
    """Requests the kernel issues for one core's args (mode 0 / 1), skipped bank excluded (plan v3 changelog 51:
    counted on the host, not in the issue loop)."""
    n = c["n_blocks"] * c["reqs"]
    skip = c.get("skip", 0xFF)
    if skip >= 8:
        return n
    Nt, c0, W, k0 = c["p"]
    banks = []
    for g in range(n):
        if c["mode"] == 0:
            banks.append(((k0 + g // W) * Nt + c0 + g % W) % 8)
        else:
            banks.append((c0 + g % Nt) % 8)
    return sum(b != skip for b in banks)


def matmul_strips(K, N, pc=None):
    """{logical core: strip args} for the tuner's 1D-multicast config of a K x N weight (row-major core order)."""
    pc = pc or derive_decode_1d_config(32, K, N)
    gx, gy = pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y
    Kt, Nt, pcn, iw = K // 32, N // 32, pc.per_core_N, pc.in0_block_w
    out = {}
    for i in range(gx * gy):
        c0 = i * pcn
        if c0 >= Nt:
            break
        out[(i % gx, i // gx)] = strip_args(Nt, c0, min(pcn, Nt - c0), 0, Kt // iw, iw)
    return out, dict(grid=(gx, gy), iw=iw, pcN=pcn, Kt=Kt, Nt=Nt)


class Probe:
    def __init__(self, dev):
        self.dev = dev
        g = dev.compute_with_storage_grid_size()
        self.all = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.tr = {xy: self._virt(xy) for xy in self.all}
        self.workers = set(self.tr.values())
        self.l1_mc = lambda row_bytes: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs(self.all), [1, row_bytes // 4], ttnn.ShardOrientation.ROW_MAJOR),
        )

        def l1(row_bytes):
            shape = ttnn.Shape([1, 1, len(self.all), row_bytes // 4])
            return ttnn.allocate_tensor_on_device(shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, self.l1_mc(row_bytes))

        self.rec = l1(REC_BYTES)
        self.scratch = l1(SCRATCH_BYTES)
        self.in0buf = l1(3 * IN0_BLOCK)  # [0, 16 KB) sender source, then the readers' 2 slots
        # 64 MB bf16 tile tensor, DRAM interleaved: 32,768 pages of 2 KB, 4,096 per bank (8 MB per bank)
        self.dram = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 8192, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        self.base = self.dram.buffer_address()
        logger.info(f"[probe] grid {g.x}x{g.y}; DRAM base {self.base:#x}; rec {self.rec.buffer_address():#x}")

    def _virt(self, xy):
        v = self.dev.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        return v.x, v.y

    def rects(self, readers):
        """Worker-only translated rectangles covering the readers (plan v3 H2): one per run of consecutive translated
        x among the readers, spanning the readers' y range in that run; every coordinate inside must be a worker."""
        tv = sorted({self.tr[xy] for xy in readers})
        xs = sorted({x for x, _ in tv})
        runs, cur = [], [xs[0]]
        for x in xs[1:]:
            if x == cur[-1] + 1:
                cur.append(x)
            else:
                runs.append(cur)
                cur = [x]
        runs.append(cur)
        out = []
        for run in runs:
            ys = [y for x, y in tv if x in run]
            r = (run[0], min(ys), run[-1], max(ys))
            cells = [(x, y) for x in range(r[0], r[2] + 1) for y in range(r[1], r[3] + 1)]
            bad = [c for c in cells if c not in self.workers]
            assert not bad, f"multicast rectangle {r} contains non-workers {bad[:4]}"
            out.append((r, len(cells)))
        assert len(out) <= 4, f"{len(out)} rectangles"
        return out

    def run(self, cfgs, depth=1, vc=1, noc=0, paced=False):
        """cfgs: {logical core: args}; returns ({core: record dict}, sender info or None)."""
        sync = ttnn.from_torch(
            torch.zeros(1, 1, len(self.all), SYNC_BYTES // 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.dev,
            memory_config=self.l1_mc(SYNC_BYTES),
        )
        sync_addr = sync.buffer_address()
        sender = min(cfgs, key=lambda xy: (xy[1], xy[0])) if paced else None
        n_blocks = max(c["n_blocks"] for c in cfgs.values())
        # compile-time: remap (bit 0) for skip, and for redirect / bank0_xy outside the table path (bit 1)
        table = any(c.get("table") for c in cfgs.values())
        remap = any("skip" in c or (not table and ("redirect" in c or "bank0_xy" in c)) for c in cfgs.values())
        rt = ttnn.RuntimeArgs()
        for x, y in self.all:
            c = cfgs.get((x, y))
            if c is None:
                rt[x][y] = [0] * NARGS
                continue
            assert c["n_blocks"] <= MAX_BLOCKS and c["req_bytes"] <= 16384
            assert not paced or c.get("depth", depth) == 1
            sx, sy = self.tr[sender] if paced else (0, 0)
            a = [1, c["mode"], self.base, c["n_blocks"], c["reqs"], c["req_bytes"], c.get("depth", depth)]
            a += [c.get("vc", vc), *c["p"], self.scratch.buffer_address(), self.rec.buffer_address()]
            a += [int(paced), sx << 16 | sy, sync_addr, c.get("skip", 0xFF), c.get("redirect", 0xFFFF)]
            a += [c.get("bank0_xy", 0)]
            rt[x][y] = a + [0] * (NARGS - len(a))
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs(self.all),
                compile_time_args=[int(remap) | int(table) << 1],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1
                ),
            )
        ]
        rect_info = None
        if paced:
            rect_info = self.rects(cfgs)
            st = self.tr[sender]
            packed, nd = [], []
            for (x0, y0, x1, y1), n in rect_info:
                # sender on NoC1: start / end swapped, as the matmul factory does for an in0 NoC of NOC_1
                packed.append(x1 << 24 | y1 << 16 | x0 << 8 | y0)
                inside = x0 <= st[0] <= x1 and y0 <= st[1] <= y1
                nd.append(n | (1 << 31) if inside else n)
            buf = self.in0buf.buffer_address()
            sa = [1, n_blocks, len(cfgs), sync_addr, buf, buf + IN0_BLOCK, IN0_BLOCK, len(packed)]
            sa += packed + [0] * (4 - len(packed)) + nd + [0] * (4 - len(nd))
            srt = ttnn.RuntimeArgs()
            srt[sender[0]][sender[1]] = sa
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=SENDER,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=crs([sender]),
                    compile_time_args=[],
                    runtime_args=srt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                    ),
                )
            )
        ttnn.generic_op([self.rec, self.rec], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[]))
        ttnn.synchronize_device(self.dev)
        sw = ttnn.to_torch(sync).view(torch.int32).reshape(len(self.all), -1).numpy().astype("uint32")
        sync.deallocate(True)
        words = ttnn.to_torch(self.rec).view(torch.int32).reshape(len(self.all), -1).numpy().astype("uint32")
        info = None
        if paced:
            s = sw[self.all.index(sender)]
            info = dict(
                sender=sender,
                consumed=int(s[0]),
                timeout=int(s[16]),
                rects=rect_info,
                in0_valid={f"{x},{y}": int(sw[self.all.index((x, y))][4]) for x, y in cfgs},
            )
            assert not s[16], f"in0 sender wait timed out: consumed {s[0]} of {n_blocks * len(cfgs)}"
            assert s[0] == n_blocks * len(cfgs), f"consumed {s[0]} != {n_blocks * len(cfgs)}"
        out = {}
        for i, xy in enumerate(self.all):
            if xy not in cfgs:
                continue
            w = words[i]
            n = int(w[0])
            assert not w[3], f"pacing wait timed out on {xy}"
            c = cfgs[xy]
            out[xy] = dict(
                t_start=int(w[1]),
                t_end=int(w[2]),
                t_first=int(w[4]),
                issued=issued_requests(c),
                req_bytes=c["req_bytes"],
                blocks=w[REC_HDR : REC_HDR + 4 * n].reshape(n, 4).tolist(),
                virt=self.tr[xy],
                phys=phys(xy),
            )
        return out, info


def _ticks(a, b):
    return ((b - a) % (1 << 32)) / CLOCK_MHZ  # us


def summarize(res):
    """Plan v3 G2: per-core rate = bytes / (first issue -> last word); bulk = bytes delivered between the last core's
    first word and the first core's last word / that window; op-time proxy = first issue of any core -> last word of
    the last core; tail = first core's last word -> last core's last word; rho(physical y, last word)."""
    t0 = min(r["t_start"] for r in res.values())
    rates, ends, firsts = {}, {}, {}
    for xy, r in res.items():
        b = r["issued"] * r["req_bytes"]
        rates[xy] = b / _ticks(r["t_start"], r["t_end"]) / 1e3
        ends[xy] = _ticks(t0, r["t_end"])
        firsts[xy] = _ticks(t0, r["t_first"]) if r["t_first"] else _ticks(t0, r["t_start"])
    lo, hi = max(firsts.values()), min(ends.values())
    bulk = None
    if hi > lo:
        tot = 0.0
        for xy, r in res.items():
            per_block = r["issued"] * r["req_bytes"] / len(r["blocks"])
            ts = [firsts[xy]] + [_ticks(t0, b[1]) for b in r["blocks"]]
            cum = [0.0] + [(i + 1) * per_block for i in range(len(r["blocks"]))]
            tot += _interp(hi, ts, cum) - _interp(lo, ts, cum)
        bulk = tot / (hi - lo) / 1e3
    total = sum(r["issued"] * r["req_bytes"] for r in res.values())
    op = max(ends.values())
    v = list(rates.values())
    mean = sum(v) / len(v)
    cv = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / mean if len(v) > 1 else 0.0
    ys = [phys(xy)[1] for xy in res]
    rho = _spearman(ys, [ends[xy] for xy in res]) if len(set(ys)) > 1 else None
    return dict(
        rates=rates,
        ends=ends,
        firsts=firsts,
        bulk=bulk,
        op_us=op,
        tail_us=op - min(ends.values()),
        eff=total / op / 1e3,
        maxmin=max(v) / min(v),
        cv=cv,
        rho_y_end=rho,
    )


def _interp(x, ts, ys):
    if x <= ts[0]:
        return ys[0]
    for i in range(1, len(ts)):
        if x <= ts[i]:
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9)
    return ys[-1]


def _spearman(a, b):
    ra = torch.tensor(a, dtype=torch.float64).argsort().argsort().double()
    rb = torch.tensor(b, dtype=torch.float64).argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def group_configs(p):
    """[(name, cfgs, depth, vc, noc, paced)] for GROUP."""
    if GROUP == "debug_rect":
        cf = {(0, 0): strip_args(32, 0, 1, 0, 8, 8), (1, 0): strip_args(32, 1, 1, 0, 8, 8)}
        return [("dbg_rect", cf, 1, 1, 0, True)]
    if GROUP == "debug_l":
        cf = {(0, 0): strip_args(32, 0, 1, 0, 8, 8), (1, 1): strip_args(32, 1, 1, 0, 8, 8)}
        return [("dbg_l", cf, 1, 1, 0, True)]
    if GROUP == "gate2":
        fat, fat_pc = matmul_strips(8192, 1024)
        thin, thin_pc = matmul_strips(1024, 8192)
        logger.info(f"[probe] fat-K strips {fat_pc}; thin-K strips {thin_pc}")
        burst = {(0, 0): bank_args(1, 0, 16384, 0, 32, 1, 16384)}
        return [
            ("a_on", fat, 1, 1, 0, True),
            ("b_on", thin, 1, 1, 0, True),
            ("c_burst", burst, 3, 1, 0, False),
            ("a_off", fat, 1, 1, 0, False),
            ("b_off", thin, 1, 1, 0, False),
        ]
    if GROUP == "gate2_ep0":
        # gate v2 (a) / (b) configs with bank 0 read through endpoint (0, 0) = translated (17, 12) (record §9.2e)
        fat, _ = matmul_strips(8192, 1024)
        thin, _ = matmul_strips(1024, 8192)
        # both arms compile the same remap path (per-request work costs rate, changelog 51); only the endpoint differs
        ep = lambda cf, xy: {k: dict(v, bank0_xy=xy) for k, v in cf.items()}  # noqa: E731
        E0 = 17 | 12 << 6
        return [
            ("a_on", ep(fat, 0), 1, 1, 0, True),
            ("a_on_ep0", ep(fat, E0), 1, 1, 0, True),
            ("a_off", ep(fat, 0), 1, 1, 0, False),
            ("a_off_ep0", ep(fat, E0), 1, 1, 0, False),
            ("b_on", ep(thin, 0), 1, 1, 0, True),
            ("b_on_ep0", ep(thin, E0), 1, 1, 0, True),
        ]
    if GROUP == "p1_ep":
        # P1 next limit (record §9.2e): bank 0 through each of its endpoints (0, 11) default, (0, 0), (0, 1), with the
        # full fat-K load and with its 4 readers alone; bank 1's readers alone as the one-bank reference. Generator
        # off; fat-K 8 x 4, so core (x, y) reads bank x. Every arm compiles the same remap path.
        fat, _ = matmul_strips(8192, 1024)
        ep = lambda cf, xy: {k: dict(v, bank0_xy=xy) for k, v in cf.items()}  # noqa: E731
        only = lambda b: {k: v for k, v in fat.items() if k[0] == b}  # noqa: E731
        out = []
        for tag, xy in (("e11", 0), ("e00", 17 | 12 << 6), ("e01", 17 | 13 << 6)):
            out += [(f"full_{tag}", ep(fat, xy), 1, 1, 0, False), (f"b0only_{tag}", ep(only(0), xy), 1, 1, 0, False)]
        return out + [("b1only", ep(only(1), 0), 1, 1, 0, False)]
    if GROUP == "p1_ref":
        # E0 (ii) (one bank's 4 fat-K readers alone) with and without the remap path compiled in, banks 0 / 1 / 4:
        # does the p1_ep bank-0 deficit reproduce without remap (record §9.2e)?
        fat, _ = matmul_strips(8192, 1024)
        only = lambda b: {k: v for k, v in fat.items() if k[0] == b}  # noqa: E731
        out = []
        for b in (0, 1, 4):
            out += [(f"b{b}only", only(b), 1, 1, 0, False)]
            out += [(f"b{b}only_remap", {k: dict(v, bank0_xy=0) for k, v in only(b).items()}, 1, 1, 0, False)]
        return out
    if GROUP == "p1_tab":
        # P1 next limit on the table path (record §9.2e): the old path (no remap) as reference, then the table path
        # through (0, 11) and (0, 0), bank 0's / bank 1's readers alone and the full fat-K load. Generator off.
        fat, _ = matmul_strips(8192, 1024)
        only = lambda b: {k: v for k, v in fat.items() if k[0] == b}  # noqa: E731
        tab = lambda cf, xy: {k: dict(v, table=True, bank0_xy=xy) for k, v in cf.items()}  # noqa: E731
        E0 = 17 | 12 << 6
        return [
            ("full_old", fat, 1, 1, 0, False),
            ("b0only_old", only(0), 1, 1, 0, False),
            ("b1only_old", only(1), 1, 1, 0, False),
            ("full_tab_e11", tab(fat, 0), 1, 1, 0, False),
            ("b0only_tab_e11", tab(only(0), 0), 1, 1, 0, False),
            ("b1only_tab", tab(only(1), 0), 1, 1, 0, False),
            ("full_tab_e00", tab(fat, E0), 1, 1, 0, False),
            ("b0only_tab_e00", tab(only(0), E0), 1, 1, 0, False),
        ]
    if GROUP == "e1v3":
        out = []
        for xy in p.all:
            for b in range(8):
                out.append((f"fat_{xy[0]}_{xy[1]}_b{b}", {xy: strip_args(32, b, 1, 0, 32, 8)}, 1, 1, 0, False))
            for s, c0 in (("lo", 0), ("hi", 4)):
                out.append((f"thin_{xy[0]}_{xy[1]}_{s}", {xy: strip_args(256, c0, 4, 0, 4, 8)}, 1, 1, 0, False))
        return out
    if GROUP in ("e2v3", "e2v3_dup"):
        xy = tuple(int(v) for v in os.environ["PROBE_E2_POS"].split(","))
        out = []
        for a in ("strided2k", "contig2k", "burst"):
            for depth in (1, 2, 4):
                for nb in (1, 2, 4, 8):
                    if GROUP == "e2v3_dup" and (a, depth, nb) != ("strided2k", 1, 1):
                        continue
                    if a == "strided2k":
                        cfg = bank_args(nb, 0, 8192, 0, 32, 8, 2048)
                    elif a == "contig2k":
                        cfg = bank_args(nb, 0, 2048, 0, 32, 8, 2048)
                    else:  # one request per (block, bank): 16 KB / N_b
                        rb = 16384 // nb
                        cfg = bank_args(nb, 0, rb, 0, 32, nb, rb)
                    out.append((f"{a}_d{depth}_b{nb}", {xy: cfg}, depth, 1, 0, False))
        return out
    raise ValueError(f"unknown PROBE_GROUP {GROUP}")


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_dram_read_probe(mesh_device):
    p = Probe(mesh_device)
    OUT.mkdir(parents=True, exist_ok=True)
    if GROUP == "clock":
        ticks = 1 << 30
        cfg = {(0, 0): dict(mode=2, n_blocks=0, reqs=0, req_bytes=0, p=(ticks, 0, 0, 0))}
        p.run(cfg)  # compile
        t = time.perf_counter()
        r = p.run(cfg)[0][(0, 0)]
        wall = time.perf_counter() - t
        dev_ticks = (r["t_end"] - r["t_start"]) % (1 << 32)
        logger.info(f"[probe] clock: {dev_ticks} ticks in {wall * 1e3:.1f} ms host -> {dev_ticks / wall / 1e6:.1f} MHz")
        (OUT / "clock.json").write_text(json.dumps(dict(ticks=dev_ticks, wall_s=wall)))
        return
    if GROUP == "endpoints":
        p.run({(0, 0): dict(mode=4, n_blocks=0, reqs=0, req_bytes=0, p=(0, 0, 0, 0))})
        w = ttnn.to_torch(p.rec).view(torch.int32).reshape(len(p.all), -1).numpy().astype("uint32")[0]
        dec = lambda v: (int(v) & 0x3F, (int(v) >> 6) & 0x3F)  # noqa: E731  NOC_XY_ENCODING: x | y << 6
        ep = {b: dict(noc0=dec(w[REC_HDR + 2 * b]), noc1=dec(w[REC_HDR + 2 * b + 1])) for b in range(8)}
        logger.info(f"[probe] DRAM bank endpoints (translated x, y): {ep}")
        (OUT / "endpoints.json").write_text(json.dumps(ep))
        return
    groups = group_configs(p)
    for name, cfgs, depth, vc, noc, paced in groups:  # warm-up, discarded (plan v3 G3)
        p.run(cfgs, depth, vc, noc, paced)
    rows = []
    for rep in range(REPEATS):  # repeat loop outer, configs inner (plan v3 G4)
        for name, cfgs, depth, vc, noc, paced in groups:
            res, info = p.run(cfgs, depth, vc, noc, paced)
            s = summarize(res)
            rows.append(
                dict(
                    name=name,
                    rep=rep,
                    depth=depth,
                    vc=vc,
                    noc=noc,
                    paced=paced,
                    sender=info,
                    cores={f"{x},{y}": dict(res[(x, y)], rate=s["rates"][(x, y)]) for x, y in res},
                    **{k: s[k] for k in ("bulk", "op_us", "tail_us", "eff", "maxmin", "cv", "rho_y_end")},
                )
            )
            if len(groups) <= 64:
                r = list(s["rates"].values())
                bulk = f"{s['bulk']:.1f}" if s["bulk"] is not None else "-"
                rho = f"{s['rho_y_end']:+.2f}" if s["rho_y_end"] is not None else "-"
                logger.info(
                    f"[probe] {name} rep {rep}: {len(r)} cores, per-core median {_median(r):.2f} GB/s"
                    f" (min {min(r):.2f}, max {max(r):.2f}); bulk {bulk}; op {s['op_us']:.1f} us; tail"
                    f" {s['tail_us']:.1f} us; rho(phys y, end) {rho}" + (f"; sender {info['consumed']}" if info else "")
                )
        logger.info(f"[probe] repeat {rep} done ({len(groups)} configs)")
    (OUT / f"{GROUP}.json").write_text(json.dumps(rows))
