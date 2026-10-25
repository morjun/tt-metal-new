# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM-sharded weights for the 12B drafter's fat-K matmuls, isolated (gemma4-12b record §9.2a, gap #1 of §9.2d).

Fat-K weights (pre_projection 7680 x 1024, o_proj 4096 / 8192 x 1024, down_proj 8192 x 1024) read at ~10.3 GB/s per
core on their 32-core 1D-multicast grid. A DRAM WIDTH_SHARDED weight read by
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig puts one contiguous N/8 slice in each bank. Per shape:

  b   the drafter's config: in0 and output DRAM interleaved, weight DRAM interleaved, the tuner's 1D-multicast
      config (derive_decode_1d_config: 32 cores, in0_block_w 8, per_core_N 1), HiFi2 as compute_config_for
  f   DRAM-sharded weight (8 banks), in0 L1 WIDTH_SHARDED on <cores> cores, output L1 WIDTH_SHARDED on the same
      cores; swept over cores in {8, 16, 32} (dividing Kt and Nt) and every legal in0_block_w <= 8
  f+  f with the reshards a model pays inside the timed region: in0 DRAM interleaved -> L1 width-sharded, output
      -> DRAM interleaved

Timing: INNER ops per trace, REPLAYS replays, host wall clock per op (as test_matmul_weight_placement.py). Output:
bit-identity of f / f+ against b (same bf16 bits?) and PCC against an fp32 reference. Class B (a different kernel;
the K accumulation order may differ) until the identity check says otherwise.

    TT_METAL_RUNTIME_ROOT=$PWD TT_METAL_HOME=$PWD PYTHONPATH=/tmp/lmhead_ttnn_overlay_e2e:$PWD TT_VISIBLE_DEVICES=0 \\
        pytest -q -s research_codes/mm_profiling/test_dram_sharded_fatk.py
"""

import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.matmul_tuning import derive_decode_1d_config

INNER, REPLAYS = 20, 30
T = 32
SHAPES = [("o_proj (sliding)", 4096, 1024), ("pre_projection", 7680, 1024), ("down_proj / o_proj (global)", 8192, 1024)]
# thin-K (DSH_SET=thin): K 1024; "bW" adds W's wide-grid 1D-multicast config (GEMMA4_MM_GRID_SHAPES of +gridwide)
THIN = [
    ("qkv / gate / up (sliding)", 1024, 8192, 3),
    ("qkv_proj (global)", 1024, 9216, 3),
    ("post_projection", 1024, 3840, 2),
]
if __import__("os").getenv("DSH_SET") == "thin":
    SHAPES = THIN
ONLY = __import__("os").getenv("DSH_ONLY")  # e.g. "pre_projection": run one shape


def _time(md, fn):
    outs = [fn() for _ in range(INNER)]  # compile + warm
    ttnn.synchronize_device(md)
    for o in outs:
        o.deallocate(True)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    outs = [fn() for _ in range(INNER)]
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    us = (time.perf_counter() - t0) / REPLAYS / INNER * 1e6
    ttnn.release_trace(md, tid)
    ref = ttnn.to_torch(outs[-1]).float()
    for o in outs:
        o.deallocate(True)
    return us, ref


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _grid(cores):
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(cores, 8) - 1, cores // 8 - 1 if cores >= 8 else 0))}
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_dram_sharded_fatk(mesh_device, reset_seeds):
    md = mesh_device
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    banks = md.dram_grid_size().x
    dgrid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    for name, K, N, *wide in SHAPES:
        if ONLY and ONLY not in name:
            continue
        Kt, Nt = K // T, N // T
        x_t = torch.randn(1, 1, T, K, dtype=torch.bfloat16)
        w_t = torch.randn(1, 1, K, N, dtype=torch.bfloat16) * K**-0.5
        gold = (x_t.float() @ w_t.float()).squeeze()
        x_dram = ttnn.from_torch(
            x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w_il = ttnn.from_torch(
            w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w_ds = ttnn.from_torch(
            w_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dgrid, [K, N // banks], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

        pc_b = derive_decode_1d_config(1, K, N)
        us_b, out_b = _time(
            md,
            lambda: ttnn.linear(
                x_dram, w_il, program_config=pc_b, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        logger.info(
            f"[dsh] {name} K={K} N={N} b (1D mcast, {pc_b.compute_with_storage_grid_size.x}x{pc_b.compute_with_storage_grid_size.y},"
            f" iw{pc_b.in0_block_w}): {us_b:8.2f} us  pcc={_pcc(out_b, gold):.6f}"
        )
        if wide:
            # W's DRAM-fed config: the tuner's in0_block_w on the row-major prefix of ceil(Nt / pcN) cores of the full
            # grid (matmul_tuning.py config_for, GEMMA4_MM_GRID_SHAPES branch)
            full = md.compute_with_storage_grid_size()
            pcn = wide[0]
            cores_w = -(-Nt // pcn)
            pc_w = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(min(cores_w, full.x), -(-cores_w // full.x)),
                in0_block_w=pc_b.in0_block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=pcn,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            us_bw, out_bw = _time(
                md,
                lambda: ttnn.linear(
                    x_dram, w_il, program_config=pc_w, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                ),
            )
            logger.info(
                f"[dsh] {name} bW (1D mcast, {cores_w} cores, pcN {pcn}, iw{pc_b.in0_block_w}): {us_bw:8.2f} us"
                f" ({us_bw / us_b - 1:+.1%} vs b)  bit-identical to b: {torch.equal(out_bw.to(torch.bfloat16), out_b.to(torch.bfloat16))}"
            )
            us_b, out_b = us_bw, out_bw  # f / f+ below are compared with W's config

        for cores in [int(c) for c in __import__("os").getenv("DSH_CORES", "1,2,4,8,16,32").split(",")]:
            if Kt % cores or Nt % cores:
                continue
            g = _grid(cores)
            in0_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(g, [T, K // cores], ttnn.ShardOrientation.ROW_MAJOR),
            )
            out_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(g, [T, N // cores], ttnn.ShardOrientation.ROW_MAJOR),
            )
            x_sh = ttnn.to_memory_config(x_dram, in0_mc)
            shard_w = Kt // cores
            for iw in [d for d in range(1, 9) if shard_w % d == 0 and (d == 8 or cores >= 8)]:
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=iw, per_core_M=1, per_core_N=Nt // cores, fused_activation=None
                )

                def f():
                    return ttnn.linear(x_sh, w_ds, program_config=pc, compute_kernel_config=ckc, memory_config=out_mc)

                def fplus():
                    xs = ttnn.to_memory_config(x_dram, in0_mc)
                    o = ttnn.linear(xs, w_ds, program_config=pc, compute_kernel_config=ckc, memory_config=out_mc)
                    xs.deallocate(True)
                    r = ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG)
                    o.deallocate(True)
                    return r

                try:
                    us_f, out_f = _time(md, f)
                    us_fp, out_fp = _time(md, fplus)
                except Exception as e:  # noqa: BLE001 - report the failing config and go on
                    logger.info(f"[dsh] {name} f cores={cores} iw{iw}: FAILED {str(e).splitlines()[0][:200]}")
                    continue
                same = torch.equal(out_f.to(torch.bfloat16), out_b.to(torch.bfloat16))
                diff = int((out_f.to(torch.bfloat16) != out_b.to(torch.bfloat16)).sum())
                logger.info(
                    f"[dsh] {name} f cores={cores} iw{iw}: {us_f:8.2f} us ({us_f / us_b - 1:+.1%} vs {'bW' if wide else 'b'})"
                    f"  f+ {us_fp:8.2f} us ({us_fp / us_b - 1:+.1%} vs {'bW' if wide else 'b'})  bit-identical to b: {same} ({diff} of {out_b.numel()} differ)"
                    f"  pcc={_pcc(out_f, gold):.6f}"
                )
            x_sh.deallocate(True)
        for t in (x_dram, w_il, w_ds):
            t.deallocate(True)
