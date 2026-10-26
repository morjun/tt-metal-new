# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2b (gemma4-12b record §9.2e): does a fat-K core read faster when its column rotates over all 8 banks along K?

down_proj's shape (K 8192) on the production 1D-multicast matmul, per_core_N 1, in0_block_w 8, on an 11-wide prefix:
  E2B_NT=32  N 1,024 on the first 32 cores of the 11 x 3 grid: column c lies in bank c mod 8 for every K row (the
             drafter's pattern, 8 KB stride within one bank)
  E2B_NT=33  N 1,056 (32 real + 1 padding column) on all 33 cores: column c of K row r lies in bank (33 r + c) mod 8,
             so each core's reads rotate over the 8 banks
Eager, REPEATS calls after a warm-up, with the in1 reader's NIU sampler (TT_MM1D_IN1_NIU_SAMPLE=1, per block) and the
device profiler on; run by e2b_nt33.sh under tracy, which keeps the sampler rows (niu_samples.csv) per arm.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

K = 8192
NT = int(os.getenv("E2B_NT", "32"))
REPEATS = int(os.getenv("E2B_REPEATS", "5"))


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_fatk_nt33(mesh_device):
    md = mesh_device
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, K, NT * 32, dtype=torch.bfloat16) * K**-0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 3),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    for i in range(1 + REPEATS):
        o = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(md)
        o.deallocate(True)
    ttnn.ReadDeviceProfiler(md)
    logger.info(f"[e2b] Nt {NT}: {1 + REPEATS} calls (first = warm-up) on the 11 x 3 prefix")
