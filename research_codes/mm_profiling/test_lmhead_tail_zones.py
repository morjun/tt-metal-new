# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-zone profile of the 12B drafter's DRAM tail matmul at one out_block_w.

Needs the kernel zones in lmhead_tail_zones.patch applied (matmul in0 sender / receiver,
in1 reader-writer, compute) and the device profiler on. One width per process:

  git apply research_codes/mm_profiling/lmhead_tail_zones.patch
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SUM=1 TT_METAL_CACHE=<fresh dir> \
  TT_METAL_PROFILER_DIR=<out dir> ZONES_OBW=8 \
  pytest research_codes/mm_profiling/test_lmhead_tail_zones.py -s

Per call: one warm-up call (JIT + first-touch), its profile read and discarded by tag, then
ZONES_CALLS profiled calls, each followed by ReadDeviceProfiler so the per-core marker buffer
(250 markers / RISC) never fills. Same shape, config and compute config as
test_lmhead_out_block_w.py.
"""

import os

import ttnn
from loguru import logger
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

from research_codes.mm_profiling.test_lmhead_out_block_w import _ckc, _pc

K, N = 1024, 223_424  # drafter tail: 262,144 - 38,720 pinned columns
PER_CORE_N = 64


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_lmhead_tail_zones(mesh_device, reset_seeds):
    width = int(os.environ["ZONES_OBW"])
    calls = int(os.getenv("ZONES_CALLS", "3"))
    assert PER_CORE_N % width == 0
    ckc = _ckc(mesh_device)
    x = ttnn.rand([1, 1, 1, K], device=mesh_device, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    w = ttnn.rand([1, 1, K, N], device=mesh_device, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
    pc = _pc(PER_CORE_N, width)

    def call():
        out = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc)
        ttnn.synchronize_device(mesh_device)
        out.deallocate(True)

    call()  # warm-up: JIT compile and first touch
    ttnn.ReadDeviceProfiler(mesh_device)
    for i in range(calls):
        call()
        ttnn.ReadDeviceProfiler(mesh_device)
    logger.info(f"[zones] out_block_w={width}: 1 warm-up + {calls} profiled calls")
