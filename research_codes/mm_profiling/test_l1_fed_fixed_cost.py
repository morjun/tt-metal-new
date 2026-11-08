# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated 1D-multicast matmul sweep (gemma4-12b record §9.2b, "What an L1-fed matmul's time is made of").

M = 32 (one tile row), K = 1024, bf16 weight, activation [32, 1024] bf16 in DRAM (as the drafter's head input),
HiFi2, packer_l1_acc on, fp32 dest off, out_block_w = per_core_N, out_subblock_w 1 (slice A's settings). Each cell
is captured as its own trace of REPS identical calls, replayed once and flushed, so the device profile holds REPS
back-to-back instances per cell. Cell list: CELLS below; the order is written to $L1FC_OUT/cells.json (trace i =
cell i). Run under tracy via run_l1_fed_fixed_cost.py.
"""

import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.weight_placement import prefix_core_range_set

REPS = int(os.getenv("L1FC_REPS", "30"))
K = 1024
T = 32


def cells():
    out = []
    # W: width at iw 2 / 8, L1-fed, 110 cores, output L1 interleaved; D: the same widths DRAM-fed
    for feed in ("l1", "dram"):
        for iw in (2, 8):
            for pcn in (1, 2, 3, 4, 6, 8, 11, 14, 17):
                out.append(dict(group="W", feed=feed, cores=110, pcn=pcn, iw=iw, out="l1"))
    # K: K-steps at the narrowest and the widest width, L1-fed
    for pcn in (1, 17):
        for iw in (1, 4, 16, 32):
            out.append(dict(group="K", feed="l1", cores=110, pcn=pcn, iw=iw, out="l1"))
    # O: output destination, L1-fed
    for pcn in (1, 17):
        for iw in (2, 32):
            for dest in ("dram", "sharded"):
                out.append(dict(group="O", feed="l1", cores=110, pcn=pcn, iw=iw, out=dest))
    # U: up_proj L0's configuration (86 cores of 11 x 8, per_core_N 3, iw8, N = 8192)
    for feed in ("l1", "dram"):
        out.append(dict(group="U", feed=feed, cores=86, pcn=3, iw=8, out="l1"))
    return out


def grid_of(cores):
    return (11, 10) if cores == 110 else (11, 8)


def core_set(cores):
    return prefix_core_range_set(cores, 11)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_l1_fed_fixed_cost(mesh_device, reset_seeds):
    out_dir = Path(os.environ["L1FC_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = cells()
    (out_dir / "cells.json").write_text(json.dumps(todo, indent=1) + "\n")
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(1, 1, T, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    ttnn.ReadDeviceProfiler(mesh_device)
    for i, c in enumerate(todo):
        cores, pcn = c["cores"], c["pcn"]
        n = 8192 if c["group"] == "U" else cores * pcn * T
        crs = core_set(cores)
        if c["feed"] == "l1":
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [K, pcn * T], ttnn.ShardOrientation.ROW_MAJOR),
            )
        else:
            mc = ttnn.DRAM_MEMORY_CONFIG
        w = ttnn.from_torch(
            torch.randn(1, 1, K, n) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=mc,
        )
        if c["out"] == "l1":
            omc = ttnn.L1_MEMORY_CONFIG
        elif c["out"] == "dram":
            omc = ttnn.DRAM_MEMORY_CONFIG
        else:
            omc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [T, pcn * T], ttnn.ShardOrientation.ROW_MAJOR),
            )
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_of(cores),
            in0_block_w=c["iw"],
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=pcn,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        def call():
            return ttnn.linear(x, w, program_config=pc, memory_config=omc, compute_kernel_config=ckc)

        y = call()  # compile
        ttnn.synchronize_device(mesh_device)
        y.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(REPS):
            y = call()
            y.deallocate(True)  # trace buffers: the next call reuses the address
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(mesh_device)
        ttnn.release_trace(mesh_device, tid)
        w.deallocate(True)
        logger.info(f"[l1fc] cell {i}: {c} N={n}")
