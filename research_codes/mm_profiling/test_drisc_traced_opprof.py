# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device profile of the TRACED fused K=3 drafter iteration (gemma4-12b §5.5.14, reload hiding).

The eager iteration of test_lm_head_l1_address_capture has milliseconds of host time between ops, so a
DRISC push started by the trigger finishes before any window matmul begins and cannot slow it. Here the
standalone rig's fused K=3 body is captured and replayed OPPROF_REPLAYS times, flushing the device
profiler after each replay, so the profile has the ops back to back as in the timed runs.

    python -m tracy -r -p -v -m pytest -s -q research_codes/mm_profiling/test_drisc_traced_opprof.py
"""

import hashlib
import os

from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tests.unit.test_assistant_standalone_l1 import _build_standalone, _make_fused_k_body

REPLAYS = int(os.getenv("OPPROF_REPLAYS", "2"))
# OPPROF_FLUSH=0 reads the device profiler only at device close, as the 2026-08-07 traced harness did
# (50+ replays, no flush): reproduces the "Op N not present in cpp_device_perf_report.csv" merge failure.
FLUSH = os.getenv("OPPROF_FLUSH", "1") == "1"
TRACES = int(os.getenv("OPPROF_TRACES", "1"))  # 2 = two traces back to back, as that harness timed per arm


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_traced_iteration_opprof(mesh_device, reset_seeds):
    mode = os.getenv("GEMMA4_STANDALONE_PLACEMENT", "dram")  # l1_sharded: GEMMA4_L1_PIN weights (record §9.2)
    rig = _build_standalone(mesh_device, mode, tune_matmuls=os.getenv("GEMMA4_TUNE_MATMULS", "1") == "1")
    if rig["placement"].enabled:
        logger.info("\n" + rig["placement"].report())
    body = _make_fused_k_body(rig, 3)
    idx, hidden = body()  # compile before capture
    ttnn.synchronize_device(mesh_device)
    idx.deallocate(True)
    hidden.deallocate(True)
    if FLUSH:
        ttnn.ReadDeviceProfiler(mesh_device)
    for _ in range(TRACES):
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        idx, hidden = body()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            if FLUSH:
                ttnn.ReadDeviceProfiler(mesh_device)
        # output fingerprint after the replays, as the standalone runner logs it, so profiled cells are checked too
        h_bytes = ttnn.to_torch(ttnn.get_device_tensors(hidden)[0]).float().numpy().tobytes()
        tok = ttnn.to_torch(ttnn.get_device_tensors(idx)[0]).flatten()[:1].tolist()
        logger.info(f"[fused-k] K=3 output: token={tok} hidden_sha256={hashlib.sha256(h_bytes).hexdigest()[:16]}")
        ttnn.release_trace(mesh_device, tid)
        idx.deallocate(True)
        hidden.deallocate(True)
