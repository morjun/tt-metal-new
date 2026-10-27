# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E6 on the production kernel (gemma4-12b record §9.2e, plan v3 E6 / changelog 53), NIU-sampled.

gate_proj's thin-K shape on the production 1D-multicast matmul, one arm per process (E6_ARM, the names of
test_endpoint_swap.py's EP_E6 set): gate_proj_8x8 (A64, tuner 8 x 8, pcN 4), gate_proj_86 (A86, 11-wide prefix, pcN 3),
gate_w2_8x8_pcn3 (W2: A64's cores at pcN 3, N 6,144). Eager, E6_REPEATS calls after a warm-up, with the in1 reader's
NIU sampler (TT_MM1D_IN1_NIU_SAMPLE=1, per block) and the device profiler on; run by run_e6_grid.py under tracy.
TT_MM1D_IN1_BANK0_XY (ep0) is passed through by the runner.
"""

import os
import sys
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

ARM = os.environ.get("E6_ARM", "gate_proj_8x8")
REPEATS = int(os.getenv("E6_REPEATS", "5"))


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_e6_grid(mesh_device):
    md = mesh_device
    os.environ["EP_E6"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_endpoint_swap import _configs

    name, K, N, pc = next(c for c in _configs() if c[0] == ARM)
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
        torch.randn(1, 1, K, N, dtype=torch.bfloat16) * K**-0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _ in range(1 + REPEATS):
        o = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(md)
        o.deallocate(True)
    ttnn.ReadDeviceProfiler(md)
    logger.info(f"[e6] {name}: {1 + REPEATS} calls (first = warm-up), ep0 {os.getenv('TT_MM1D_IN1_BANK0_XY')}")
