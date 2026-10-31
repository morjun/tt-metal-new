# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NoC trace of the drafter's lm_head DRAM tail at one width (gemma4-12b record §9.4, the 99th core).

The tail matmul as in test_tail_width_sweep.py (66 / ow6, in0_block_w 2, logits in L1; vc1 + ep0 from the env), eager:
one warm-up call, then NT_CALLS calls, each followed by ReadDeviceProfiler so the NoC-trace markers of one call never
share the profiler buffer with another. Run under tracy with --collect-noc-traces (run_1031_tail_noc.sh).

    NT_TILES=6467 python -m tracy -r -p -v --collect-noc-traces -m pytest -s -q research_codes/mm_profiling/test_tail_noc_trace.py
"""

import os

from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from research_codes.mm_profiling.test_lmhead_out_block_w import _ckc, _pc

K = 1024


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_tail_noc_trace(mesh_device, reset_seeds):
    tiles = int(os.environ["NT_TILES"])
    calls = int(os.getenv("NT_CALLS", "3"))
    ckc = _ckc(mesh_device)
    pc = _pc(66, 6)
    x = ttnn.rand([1, 1, 1, K], device=mesh_device, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    w = ttnn.rand([1, 1, K, tiles * 32], device=mesh_device, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)

    def call():
        out = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh_device)
        out.deallocate(True)

    call()  # warm-up: JIT compile and first touch
    ttnn.ReadDeviceProfiler(mesh_device)
    for _ in range(calls):
        call()
        ttnn.ReadDeviceProfiler(mesh_device)
    logger.info(f"[tail-noc] {tiles} tiles ({-(-tiles // 66)} cores): 1 warm-up + {calls} traced calls")
