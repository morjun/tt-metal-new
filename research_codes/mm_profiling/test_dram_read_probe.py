# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM read probe harness (gemma4-12b record §9.2a, plan v2 T4): kernels/dram_read_probe.cpp on chosen cores.

Each probe core issues a DRAM read stream (a matmul's in1 strip of an interleaved bf16 tile tensor, or a direct
per-bank stream) and logs per block its issue time, completion time and NIU master counters. One config group per
process (PROBE_GROUP), run under `timeout`:

    TT_METAL_RUNTIME_ROOT=$PWD TT_METAL_HOME=$PWD PYTHONPATH=/tmp/lmhead_ttnn_overlay_e2e:$PWD TT_VISIBLE_DEVICES=0 \\
        PROBE_GROUP=gate PROBE_OUT=generated/dram_read_probe timeout 600 \\
        pytest -q -s research_codes/mm_profiling/test_dram_read_probe.py

Groups:
  clock   wall-clock rate: one core spins 2^30 ticks; host wall time of the program
  gate    harness gate (a) fat-K down_proj strip, 32 cores (tuner 8 x 4, iw8, pcN 1); (b) thin-K gate_proj strip,
          64 cores (tuner 8 x 8); (c) one core, contiguous 16 KB bursts from one bank, 3 blocks in flight
  e1      one core alone at each of the 110 positions, fat-K and thin-K strip patterns
  e2      core (0, 0) alone: access {strided 2 KB, contiguous 2 KB, 16 KB burst} x depth {1, 2, 4} x banks {1, 2, 4, 8}
Every config runs REPEATS times; records go to PROBE_OUT/<group>.json (one entry per config and repeat).
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
NARGS = 21
COORD = (10, 9)  # lockstep coordinator: outside every reader set used here
TILE_BYTES = 2048
MAX_BLOCKS = 64
REC_BYTES = 4 * (4 + 4 * MAX_BLOCKS)
SCRATCH_BYTES = 256 * 1024
CLOCK_MHZ = 1350.0  # AICLK; the clock group checks it
REPEATS = int(os.getenv("PROBE_REPEATS", "3"))
GROUP = os.getenv("PROBE_GROUP", "gate")
OUT = Path(os.getenv("PROBE_OUT", "generated/dram_read_probe"))


def crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def strip_args(Nt, c0, W, k0, n_blocks, iw):
    """Mode 0: a 1D-multicast matmul core's in1 strip (columns c0..c0+W-1, iw K rows per block)."""
    return dict(mode=0, n_blocks=n_blocks, reqs=iw * W, req_bytes=TILE_BYTES, p=(Nt, c0, W, k0))


def bank_args(nbanks, bank0, stride, offset, n_blocks, reqs, req_bytes):
    """Mode 1: request g -> bank (bank0 + g % nbanks) % 8, address offset + (g / nbanks) * stride."""
    return dict(mode=1, n_blocks=n_blocks, reqs=reqs, req_bytes=req_bytes, p=(nbanks, bank0, stride, offset))


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

        def l1(row_bytes):
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs(self.all), [1, row_bytes // 4], ttnn.ShardOrientation.ROW_MAJOR),
            )
            return ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, len(self.all), row_bytes // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, mc
            )

        self.l1_mc = lambda row_bytes: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs(self.all), [1, row_bytes // 4], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.rec = l1(REC_BYTES)
        self.scratch = l1(SCRATCH_BYTES)
        # 64 MB bf16 tile tensor, DRAM interleaved: 32,768 pages of 2 KB, 4,096 per bank (8 MB per bank)
        self.dram = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 8192, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        self.base = self.dram.buffer_address()
        self.pages = 8192 * 4096 // 1024
        logger.info(f"[probe] grid {g.x}x{g.y}; DRAM base {self.base:#x}; rec {self.rec.buffer_address():#x}")

    def run(self, cfgs, depth=1, vc=1, noc=0, lockstep=False, runahead=2):
        """cfgs: {logical core: args}; returns {core: record dict}. lockstep: readers coupled through a coordinator
        at COORD (block b starts once every reader completed block b - runahead); depth 1, NoC0 only."""
        sync = ttnn.from_torch(
            torch.zeros(1, 1, len(self.all), 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.dev,
            memory_config=self.l1_mc(64),
        )
        ls = [0] * 7
        if lockstep:
            assert depth == 1 and noc == 0 and COORD not in cfgs
            v = [self._virt(xy) for xy in cfgs]
            x0, y0, x1, y1 = min(a for a, _ in v), min(b for _, b in v), max(a for a, _ in v), max(b for _, b in v)
            cx, cy = self._virt(COORD)
            ls = [1, cx << 16 | cy, sync.buffer_address(), runahead, len(cfgs), x0 << 24 | y0 << 16 | x1 << 8 | y1]
            ls.append(len(cfgs))
        rt = ttnn.RuntimeArgs()
        n_blocks = max(c["n_blocks"] for c in cfgs.values())
        for x, y in self.all:
            c = cfgs.get((x, y))
            if lockstep and (x, y) == COORD:
                a = [1, 3, 0, n_blocks, 0, 0, 1, 1, 0, 0, 0, 0, 0, self.rec.buffer_address()]
                rt[x][y] = a + ls
                continue
            if c is None:
                rt[x][y] = [0] * NARGS
                continue
            assert c["n_blocks"] <= MAX_BLOCKS and c["req_bytes"] <= 16384
            a = [1, c["mode"], self.base, c["n_blocks"], c["reqs"], c["req_bytes"], c.get("depth", depth)]
            a += [c.get("vc", vc), *c["p"], self.scratch.buffer_address(), self.rec.buffer_address()]
            rt[x][y] = a + ls
        k = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs(self.all),
            compile_time_args=[],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1
            ),
        )
        ttnn.generic_op([self.rec, self.rec], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))
        ttnn.synchronize_device(self.dev)
        sw = ttnn.to_torch(sync).view(torch.int32).reshape(len(self.all), -1).numpy().astype("uint32")
        sync.deallocate(True)
        words = ttnn.to_torch(self.rec).view(torch.int32).reshape(len(self.all), -1).numpy().astype("uint32")
        if lockstep and words[self.all.index(COORD)][3]:
            logger.info(
                f"[probe] lockstep sync words: coordinator {sw[self.all.index(COORD)][[0, 4, 8]].tolist()}, readers "
                f"{[(xy, sw[self.all.index(xy)][[0, 4, 8]].tolist(), int(words[self.all.index(xy)][3])) for xy in list(cfgs)[:4]]}"
                f"; args {ls}"
            )
        if lockstep:
            assert not words[self.all.index(COORD)][3], "lockstep coordinator wait timed out"
        out = {}
        for i, xy in enumerate(self.all):
            if xy not in cfgs:
                continue
            w = words[i]
            n = int(w[0])
            blocks = w[4 : 4 + 4 * n].reshape(n, 4)
            c = cfgs[xy]
            assert not w[3], f"lockstep wait timed out on {xy}"
            out[xy] = dict(
                t_start=int(w[1]),
                t_end=int(w[2]),
                blocks=blocks.tolist(),
                bytes_per_block=c["reqs"] * c["req_bytes"],
                virt=tuple(self._virt(xy)),
            )
        return out

    def _virt(self, xy):
        v = self.dev.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        return v.x, v.y


def summarize(res):
    """Per-core rate (GB/s, start -> end), aggregate, bulk (all cores reading) rate, Spearman(row, t_end)."""
    t0 = min(r["t_start"] for r in res.values())
    rates, ends = {}, {}
    for xy, r in res.items():
        b = len(r["blocks"]) * r["bytes_per_block"]
        dt = ((r["t_end"] - r["t_start"]) % (1 << 32)) / CLOCK_MHZ  # us
        rates[xy] = b / dt / 1e3
        ends[xy] = ((r["t_end"] - t0) % (1 << 32)) / CLOCK_MHZ
    starts = [((r["t_start"] - t0) % (1 << 32)) / CLOCK_MHZ for r in res.values()]
    lo, hi = max(starts), min(ends.values())
    bulk = None
    if hi > lo:
        # bytes each core completed inside [lo, hi], from its block completion times (linear between them)
        tot = 0.0
        for r in res.values():
            ts = [((r["t_start"] - t0) % (1 << 32)) / CLOCK_MHZ] + [
                ((b[1] - t0) % (1 << 32)) / CLOCK_MHZ for b in r["blocks"]
            ]
            cum = [i * r["bytes_per_block"] for i in range(len(ts))]
            tot += _interp([hi], ts, cum)[0] - _interp([lo], ts, cum)[0]
        bulk = tot / (hi - lo) / 1e3
    agg = sum(len(r["blocks"]) * r["bytes_per_block"] for r in res.values()) / max(ends.values()) / 1e3
    rho = _spearman([xy[1] for xy in res], [ends[xy] for xy in res]) if len({xy[1] for xy in res}) > 1 else None
    return dict(rates=rates, ends=ends, bulk=bulk, agg=agg, rho_row_end=rho, window=(lo, hi))


def _interp(xs, ts, ys):
    out = []
    for x in xs:
        if x <= ts[0]:
            out.append(ys[0])
            continue
        for i in range(1, len(ts)):
            if x <= ts[i]:
                out.append(ys[i - 1] + (ys[i] - ys[i - 1]) * (x - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9))
                break
        else:
            out.append(ys[-1])
    return out


def _spearman(a, b):
    ra = torch.tensor(a, dtype=torch.float64).argsort().argsort().double()
    rb = torch.tensor(b, dtype=torch.float64).argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def group_configs(p):
    """[(name, cfgs, depth, vc, noc)] for GROUP."""
    if GROUP == "gate":
        fat, fat_pc = matmul_strips(8192, 1024)
        thin, thin_pc = matmul_strips(1024, 8192)
        logger.info(f"[probe] fat-K strips {fat_pc}; thin-K strips {thin_pc}")
        # (c): bank 0, contiguous 16 KB bursts, 3 blocks in flight (the DRAM-sharded reader's pipelining)
        burst = {(0, 0): bank_args(1, 0, 16384, 0, 32, 1, 16384)}
        return [
            ("a_fatK_down_32c", fat, 1, 1, 0),
            ("b_thinK_gate_64c", thin, 1, 1, 0),
            ("c_burst_1bank_d3", burst, 3, 1, 0),
        ]
    if GROUP == "e1":
        # one core alone at every position: fat-K strip (down_proj column 0: bank 0, 8 KB stride, 32 blocks of 8 tiles)
        # and thin-K strip (gate_proj core 0: columns 0-3 = banks 0-3, 8 x 4 tiles per block, 16 blocks: K extended
        # past gate's 32 tiles to 128 for a steadier rate)
        out = []
        for xy in p.all:
            out.append((f"fat_{xy[0]}_{xy[1]}", {xy: strip_args(32, 0, 1, 0, 32, 8)}, 1, 1, 0))
            out.append((f"thin_{xy[0]}_{xy[1]}", {xy: strip_args(256, 0, 4, 0, 16, 8)}, 1, 1, 0))
        return out
    if GROUP == "ls_debug":
        return [("dbg2", {(0, 0): strip_args(32, 0, 1, 0, 4, 8), (1, 0): strip_args(32, 1, 1, 0, 4, 8)}, 1, 1, 0, True)]
    if GROUP == "gate_ls":
        # gate (a') / (b'): the gate's (a) and (b) configs with the lockstep coupling, run-ahead 2
        fat, _ = matmul_strips(8192, 1024)
        thin, _ = matmul_strips(1024, 8192)
        return [("a_fatK_down_32c_ls", fat, 1, 1, 0, True), ("b_thinK_gate_64c_ls", thin, 1, 1, 0, True)]
    if GROUP == "banks":
        # E0 per-bank service: down_proj strips of bank b (columns b + 8 y), 32 blocks of 8 tiles, depth 1;
        # (i) core (0, 0) alone; (ii) the gate's 4 cores (b, 0..3); (iii) cores (0, 0..3); (iv) (0, 0) bursts, depth 3
        out = []
        for b in range(8):
            out.append((f"i_b{b}", {(0, 0): strip_args(32, b, 1, 0, 32, 8)}, 1, 1, 0))
            out.append((f"ii_b{b}", {(b, y): strip_args(32, b + 8 * y, 1, 0, 32, 8) for y in range(4)}, 1, 1, 0))
            out.append((f"iii_b{b}", {(0, y): strip_args(32, b + 8 * y, 1, 0, 32, 8) for y in range(4)}, 1, 1, 0))
            out.append((f"iv_b{b}", {(0, 0): bank_args(1, b, 16384, 0, 32, 1, 16384)}, 3, 1, 0))
        return out
    if GROUP == "e2":
        # core (0, 0) alone; 16 KB blocks (the fat-K block) of 8 x 2 KB or 1 x 16 KB; 32 blocks; banks 0..n-1
        access = {"strided2k": (8192, 8, 2048), "contig2k": (2048, 8, 2048), "burst16k": (16384, 1, 16384)}
        out = []
        for a, (stride, reqs, rb) in access.items():
            for depth in (1, 2, 4):
                for nb in (1, 2, 4, 8):
                    out.append(
                        (f"{a}_d{depth}_b{nb}", {(0, 0): bank_args(nb, 0, stride, 0, 32, reqs, rb)}, depth, 1, 0)
                    )
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
        r = p.run(cfg)[(0, 0)]
        wall = time.perf_counter() - t
        dev_ticks = (r["t_end"] - r["t_start"]) % (1 << 32)
        logger.info(f"[probe] clock: {dev_ticks} ticks in {wall * 1e3:.1f} ms host -> {dev_ticks / wall / 1e6:.1f} MHz")
        (OUT / "clock.json").write_text(json.dumps(dict(ticks=dev_ticks, wall_s=wall)))
        return
    rows = []
    for name, cfgs, depth, vc, noc, *ls in group_configs(p):
        lockstep = bool(ls and ls[0])
        p.run(cfgs, depth, vc, noc, lockstep)  # compile + warm
        for rep in range(REPEATS):
            res = p.run(cfgs, depth, vc, noc, lockstep)
            s = summarize(res)
            rows.append(
                dict(
                    name=name,
                    rep=rep,
                    depth=depth,
                    vc=vc,
                    noc=noc,
                    lockstep=lockstep,
                    cores={f"{x},{y}": dict(res[(x, y)], rate=s["rates"][(x, y)]) for x, y in res},
                    bulk=s["bulk"],
                    agg=s["agg"],
                    rho_row_end=s["rho_row_end"],
                )
            )
            r = list(s["rates"].values())
            bulk = f"{s['bulk']:.1f}" if s["bulk"] is not None else "-"
            rho = f"{s['rho_row_end']:+.2f}" if s["rho_row_end"] is not None else "-"
            logger.info(
                f"[probe] {name} rep {rep}: {len(r)} cores, per-core rate median {_median(r):.2f} GB/s"
                f" (min {min(r):.2f}, max {max(r):.2f}); aggregate {s['agg']:.1f} GB/s; bulk {bulk} GB/s;"
                f" span {max(s['ends'].values()):.1f} us; rho(row, end) {rho}"
            )
    (OUT / f"{GROUP}.json").write_text(json.dumps(rows))
