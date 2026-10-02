# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated A/B: does `out_block_w` (output passes per core) change lm_head op time or numerics?

MEASUREMENT_RECORD.md §5.5.9: the 38,720-column drafter head pin lowers ttnn's auto-config
L1 budget, so the drafter DRAM tail runs out_block_w=32 (2 passes) and the TARGET lm_head
runs 25 (3 passes) instead of 64 / 75 (1 pass). This times each shape in the same process
under every blocking that fits, and compares every arm's output bit-for-bit to the
automatic config without a pin.

Arms (all bf16, HiFi2, packer_l1_acc, no fp32 dest acc — what the auto path resolves to):
  auto      : program_config=None (ttnn picks; with a pin present it reads the budget)
  obw=<n>   : explicit 1D mcast, 11x10, in0_block_w=2, per_core_N as auto, out_block_w=n

The pin is a stand-in L1 WIDTH_SHARDED tensor of the head slice's exact size (720,896 B/bank
on 110 cores). With nothing else in L1 it lands at 851,968, the address in the capture.

Run (one process, device must be idle):
  pytest research_codes/mm_profiling/test_lmhead_out_block_w.py -s
Env: OBW_ROUNDS (counterbalanced rounds, default 3), OBW_INNER / OBW_REPLAYS (trace sizing).
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

GRID = ttnn.CoreCoord(11, 10)
PIN_COLS = 38_720  # [1024, 352] bf16 shard per core = 720,896 B/bank on 110 cores
EXPECTED_PIN_ADDR = 851_968
ROUNDS = int(os.getenv("OBW_ROUNDS", "3"))
INNER = int(os.getenv("OBW_INNER", "20"))
REPLAYS = int(os.getenv("OBW_REPLAYS", "20"))


def _ckc(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _subblock_w(out_block_w):
    """Largest divisor of out_block_w <= 8: what ttnn's subblock helper picks at h=1, no fp32 dest."""
    return max(d for d in range(1, 9) if out_block_w % d == 0)


def _pc(per_core_n, out_block_w):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=_subblock_w(out_block_w),
        out_block_h=1,
        out_block_w=out_block_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _timed(device, fn):
    """Per-call us: `INNER` calls in one trace, replayed `REPLAYS` times (same method as
    test_assistant_standalone_l1._op_timed)."""
    outs = [fn() for _ in range(INNER)]
    ttnn.synchronize_device(device)
    for o in outs:
        o.deallocate(True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [fn() for _ in range(INNER)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    for _ in range(3):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / REPLAYS / INNER * 1e6
    ttnn.release_trace(device, tid)
    for o in outs:
        o.deallocate(True)
    return us


def _pin(device):
    spec = ttnn.create_sharded_memory_config(
        shape=(1024, PIN_COLS // 110),
        core_grid=ttnn.CoreGrid(y=10, x=11),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # Contents are irrelevant: the stand-in only has to occupy the head's address range.
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1024, PIN_COLS]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, spec
    )
    addr = t.buffer_address()
    assert addr == EXPECTED_PIN_ADDR, f"stand-in pin at {addr}, capture had {EXPECTED_PIN_ADDR}"
    return t


def _run_shape(device, name, k, n, rows, arms_unpinned, arms_pinned):
    """arms_*: list of out_block_w ints or 'auto'. Returns {(pinned, arm): [us...]}, outputs."""
    # div_up(div_up(N, cores), 32), as get_mcast_1d_config (matmul_program_config.cpp:380)
    per_core_n = -(-(-(-n // 110)) // 32)
    x = ttnn.rand([1, 1, rows, k], device=device, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    w = ttnn.rand([1, 1, k, n], device=device, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
    ckc = _ckc(device)

    def call(arm):
        if arm == "auto":
            return lambda: ttnn.linear(x, w, compute_kernel_config=ckc)
        return lambda: ttnn.linear(x, w, program_config=_pc(per_core_n, arm), compute_kernel_config=ckc)

    def label(pinned, arm):
        return f'{"pin" if pinned else "nopin"}/{arm if arm == "auto" else f"obw={arm}"}'

    outputs, times, cb_sizes = {}, {}, {}
    for pinned, arms in ((False, arms_unpinned), (True, arms_pinned)):
        if not arms:
            continue
        pin = _pin(device) if pinned else None
        for arm in arms:
            o = call(arm)()  # compile / warm outside the graph capture
            o.deallocate(True)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            o = call(arm)()
            graph = ttnn.graph.end_graph_capture()
            cbs = [
                (int(nd["params"]["address"]), int(nd["params"]["size"]))
                for nd in graph
                if nd.get("node_type") == "circular_buffer_allocate" and nd["params"].get("globally_allocated") == "0"
            ]
            cb_sizes[label(pinned, arm)] = cbs
            outputs[label(pinned, arm)] = ttnn.to_torch(o).float()
            o.deallocate(True)
        # Counterbalanced: forward order, then reversed, ROUNDS times.
        for r in range(ROUNDS):
            order = arms if r % 2 == 0 else list(reversed(arms))
            for arm in order:
                times.setdefault(label(pinned, arm), []).append(_timed(device, call(arm)))
        if pin is not None:
            pin.deallocate(True)
    ref = outputs[label(False, "auto")]
    logger.info(f"[obw] {name}: K={k} N={n} rows={rows} per_core_N={per_core_n}")
    for key, us in times.items():
        out = outputs[key]
        same = torch.equal(out, ref)
        nd = int((out != ref).sum())
        logger.info(
            f"[obw] {name:12s} {key:14s} median {statistics.median(us):9.2f} us  runs={[round(u, 2) for u in us]}"
            f"  bit-identical-to-nopin/auto={same} (diff elems {nd})  argmax={int(out.reshape(-1, out.shape[-1])[0].argmax())}"
            f"  CBs(addr,size)={cb_sizes[key]}"
        )
    x.deallocate(True)
    w.deallocate(True)
    return times, outputs


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_lmhead_out_block_w(mesh_device, reset_seeds):
    # Drafter DRAM tail: K=1024, N=223,424 (262,144 - 38,720), one activation row.
    tail_t, tail_o = _run_shape(
        mesh_device,
        "drafter_tail",
        1024,
        223_424,
        1,
        arms_unpinned=["auto", 64],
        arms_pinned=["auto", 32, 64],
    )
    # Target head: K=3840, N=262,144, 4 verify rows. 75 cannot fit under the pin (would end 887,680).
    tgt_t, tgt_o = _run_shape(
        mesh_device,
        "target_head",
        3840,
        262_144,
        4,
        arms_unpinned=["auto", 75, 25],
        arms_pinned=["auto", 25],
    )
    for name, t in (("drafter_tail", tail_t), ("target_head", tgt_t)):
        med = {k: statistics.median(v) for k, v in t.items()}
        logger.info(f"[obw] SUMMARY {name}: " + ", ".join(f"{k}={v:.2f}" for k, v in med.items()))


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_lmhead_out_block_w_sweep(mesh_device, reset_seeds):
    """Every divisor of per_core_N as out_block_w, explicit configs only (no pin needed:
    an explicit config does not read the L1 budget, and the A/B above shows the pin
    itself changes nothing at a fixed config)."""
    _run_shape(
        mesh_device, "drafter_tail", 1024, 223_424, 1, arms_unpinned=["auto", 64, 32, 16, 8, 4, 2, 1], arms_pinned=[]
    )
    _run_shape(
        mesh_device, "target_head", 3840, 262_144, 4, arms_unpinned=["auto", 75, 25, 15, 5, 3, 1], arms_pinned=[]
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_drafter_full_head_out_block_w_sweep(mesh_device, reset_seeds):
    """The unpinned baseline the pin was compared against: the drafter's full head,
    K=1024, N=262,144, which the auto path runs 75-wide (1 pass)."""
    _run_shape(mesh_device, "drafter_full", 1024, 262_144, 1, arms_unpinned=["auto", 75, 25, 15, 5, 3], arms_pinned=[])


def _arms(env, default):
    """Arm list from a comma-separated env var (e.g. "auto,1"), else the default."""
    raw = os.getenv(env)
    return [a if a == "auto" else int(a) for a in raw.split(",")] if raw else default


def _timed_multi(device, fn):
    """_timed for an fn returning a tuple of outputs (the pinned head is two ops)."""

    def run():
        return fn()

    outs = [run() for _ in range(INNER)]
    ttnn.synchronize_device(device)
    for group in outs:
        for o in group:
            o.deallocate(True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [run() for _ in range(INNER)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    for _ in range(3):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / REPLAYS / INNER * 1e6
    ttnn.release_trace(device, tid)
    for group in outs:
        for o in group:
            o.deallocate(True)
    return us


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_drafter_whole_head_unpinned_vs_pinned(mesh_device, reset_seeds):
    """Whole drafter head (all 262,144 columns), unpinned vs pinned, at every out_block_w.

    unpinned : one matmul, K=1024 x N=262,144 from DRAM; out_block_w over the divisors of 75.
    pinned   : the 38,720-column slice from L1 (its own explicit 11-wide config, as the model's
               `lm_head_l1_pc`) + the 223,424-column tail from DRAM, timed TOGETHER in one trace;
               out_block_w (of the tail) over the divisors of 64.
    Phases: unpinned, pinned, unpinned again (drift check). Each arm 3 counterbalanced runs.
    Writes generated/lmhead_l1_profile/out_block_w/whole_head.json.
    """
    import json
    from pathlib import Path

    K, N, COLS = 1024, 262_144, PIN_COLS
    ckc = _ckc(mesh_device)
    x = ttnn.rand([1, 1, 1, K], device=mesh_device, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    w = ttnn.rand([1, 1, K, N], device=mesh_device, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
    full_n = -(-(-(-N // 110)) // 32)  # 75
    tail_n = -(-(-(-(N - COLS) // 110)) // 32)  # 64

    def full(arm):
        if arm == "auto":
            return lambda: ttnn.linear(x, w, compute_kernel_config=ckc)
        return lambda: ttnn.linear(x, w, program_config=_pc(full_n, arm), compute_kernel_config=ckc)

    unpinned_arms = _arms("OBW_WHOLE_UNPINNED", ["auto", 75, 25, 15, 5, 3])
    ref = ttnn.to_torch(full("auto")()).float().reshape(-1)[:N]

    def run_unpinned(tag):
        res, exact = {}, {}
        for arm in unpinned_arms:
            out = ttnn.to_torch(full(arm)()).float().reshape(-1)[:N]
            exact[str(arm)] = bool(torch.equal(out, ref))
        for r in range(ROUNDS):
            for arm in unpinned_arms if r % 2 == 0 else list(reversed(unpinned_arms)):
                res.setdefault(str(arm), []).append(_timed(mesh_device, full(arm)))
        for arm in unpinned_arms:
            logger.info(
                f"[whole] unpinned {tag} {str(arm):>4s} median {statistics.median(res[str(arm)]):9.2f} "
                f"runs={[round(v, 2) for v in res[str(arm)]]} bit-identical={exact[str(arm)]}"
            )
        return res, exact

    unpinned_a, exact_a = run_unpinned("A")

    # Pinned: split the same weight. Slice to L1 width-sharded (created in DRAM, then moved: one L1 copy).
    head_dram = ttnn.slice(w, [0, 0, 0, 0], [1, 1, K, COLS])
    tail = ttnn.slice(w, [0, 0, 0, COLS], [1, 1, K, N])
    spec = ttnn.create_sharded_memory_config(
        shape=(K, COLS // 110),
        core_grid=ttnn.CoreGrid(y=10, x=11),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    pinned_slice = ttnn.to_memory_config(head_dram, spec)
    head_dram.deallocate(True)
    assert pinned_slice.buffer_address() == EXPECTED_PIN_ADDR, pinned_slice.buffer_address()
    slice_pc = _pc(COLS // 32 // 110, COLS // 32 // 110)  # per_core_N = out_block_w = 11

    def pinned(arm):
        def tail_call():
            if arm == "auto":
                return ttnn.linear(x, tail, compute_kernel_config=ckc)
            return ttnn.linear(x, tail, program_config=_pc(tail_n, arm), compute_kernel_config=ckc)

        return lambda: (ttnn.linear(x, pinned_slice, program_config=slice_pc, compute_kernel_config=ckc), tail_call())

    pinned_arms = _arms("OBW_WHOLE_PINNED", ["auto", 64, 32, 16, 8, 4, 2])
    res_p, exact_p = {}, {}
    for arm in pinned_arms:
        s_out, t_out = pinned(arm)()
        joined = torch.cat(
            [ttnn.to_torch(s_out).float().reshape(-1)[:COLS], ttnn.to_torch(t_out).float().reshape(-1)[: N - COLS]]
        )
        exact_p[str(arm)] = bool(torch.equal(joined, ref))
    for r in range(ROUNDS):
        for arm in pinned_arms if r % 2 == 0 else list(reversed(pinned_arms)):
            res_p.setdefault(str(arm), []).append(_timed_multi(mesh_device, pinned(arm)))
    for arm in pinned_arms:
        logger.info(
            f"[whole] pinned     {str(arm):>4s} median {statistics.median(res_p[str(arm)]):9.2f} "
            f"runs={[round(v, 2) for v in res_p[str(arm)]]} slice+tail bit-identical to unpinned={exact_p[str(arm)]}"
        )
    pinned_slice.deallocate(True)
    tail.deallocate(True)

    unpinned_b, exact_b = run_unpinned("B")
    out = (
        Path(__file__).resolve().parents[2]
        / "generated/lmhead_l1_profile/out_block_w"
        / os.getenv("OBW_WHOLE_OUT", "whole_head.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "K": K,
                "N": N,
                "pinned_cols": COLS,
                "per_core_N": {"unpinned": full_n, "tail": tail_n, "slice": 11},
                "auto_resolves_to": {"unpinned": 75, "pinned_tail": 32},
                "unpinned_A": unpinned_a,
                "pinned": res_p,
                "unpinned_B": unpinned_b,
                "bit_identical": {"unpinned_A": exact_a, "pinned": exact_p, "unpinned_B": exact_b},
                "rounds": ROUNDS,
                "inner": INNER,
                "replays": REPLAYS,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"[whole] wrote {out}")
    assert all(exact_a.values()) and all(exact_p.values()) and all(exact_b.values())


def _banks_per_step(per_core_n, n_tiles, width, rows=2, cores=110, banks=8):
    """DRAM banks touched by ALL cores in one K-step, if page p lives in bank p % 8.

    Core c reads K-rows r in [0, rows) (in0_block_w = 2) and tile columns
    c * per_core_N + j, j in [0, width), of a row-major interleaved [Kt, Nt] weight.
    """
    return len(
        {(r * n_tiles + c * per_core_n + j) % banks for r in range(rows) for c in range(cores) for j in range(width)}
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_bank_hotspot(mesh_device, reset_seeds):
    """Does narrow-width matmul speed follow per_core_N mod 8 (DRAM bank spread), not size?

    Part A: width 1 at per_core_N = 60..72 and 75, N = per_core_N * 32 * 110 (every core full).
    Part B: every legal width at per_core_N 63 (odd) vs 64 (multiple of 8).
    All weights stay resident; arms run in counterbalanced order (forward, reverse, forward).
    Writes generated/lmhead_l1_profile/out_block_w/bank_hotspot.json.
    """
    import json
    from pathlib import Path

    K = 1024
    ckc = _ckc(mesh_device)
    x = ttnn.rand([1, 1, 1, K], device=mesh_device, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    pcns = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 75]
    weights = {
        p: ttnn.rand([1, 1, K, p * 32 * 110], device=mesh_device, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
        for p in pcns
    }
    curves = {63: [1, 3, 7, 9, 21, 63], 64: [1, 2, 4, 8, 16, 32, 64]}
    arms = [(p, 1) for p in pcns] + [(p, w) for p, ws in curves.items() for w in ws if w != 1]

    def call(p, w):
        return lambda: ttnn.linear(x, weights[p], program_config=_pc(p, w), compute_kernel_config=ckc)

    exact = {}
    for p in pcns:  # width 1 vs 1 pass: same numbers?
        a = ttnn.to_torch(call(p, 1)()).float()
        b = ttnn.to_torch(call(p, p)()).float()
        exact[p] = bool(torch.equal(a, b))
    times = {}
    for r in range(ROUNDS):
        for p, w in arms if r % 2 == 0 else list(reversed(arms)):
            times.setdefault(f"{p}/{w}", []).append(_timed(mesh_device, call(p, w)))
    rows = []
    for p, w in arms:
        us = statistics.median(times[f"{p}/{w}"])
        mb = K * p * 32 * 110 * 2 / 1e6
        banks = _banks_per_step(p, p * 110, w)
        rows.append(
            {
                "per_core_N": p,
                "width": w,
                "passes": p // w,
                "banks_per_step": banks,
                "us": us,
                "runs": times[f"{p}/{w}"],
                "MB": mb,
                "GBps": mb / us * 1e3,
                "us_per_pass_tile": us / p,
            }
        )
        logger.info(
            f"[bank] per_core_N={p:3d} mod8={p % 8} width={w:2d} passes={p // w:3d} banks/step={banks} "
            f"{us:9.2f} us  {mb / us * 1e3:6.1f} GB/s  runs={[round(v, 2) for v in times[f'{p}/{w}']]}"
        )
    logger.info(f"[bank] width-1 vs 1-pass bit-identical: {exact}")
    out = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/out_block_w/bank_hotspot.json"
    out.write_text(
        json.dumps(
            {"rows": rows, "bit_identical_w1_vs_1pass": exact, "rounds": ROUNDS, "inner": INNER, "replays": REPLAYS},
            indent=2,
        )
        + "\n"
    )
    for t in weights.values():
        t.deallocate(True)
    assert all(exact.values())
