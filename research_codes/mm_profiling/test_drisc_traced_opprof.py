# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device profile of the TRACED fused K=3 drafter iteration (gemma4-12b §5.5.14, reload hiding).

The eager iteration of test_lm_head_l1_address_capture has milliseconds of host time between ops, so a
DRISC push started by the trigger finishes before any window matmul begins and cannot slow it. Here the
standalone rig's fused K=3 body is captured and replayed OPPROF_REPLAYS times, flushing the device
profiler after each replay, so the profile has the ops back to back as in the timed runs.

    python -m tracy -r -p -v -m pytest -s -q research_codes/mm_profiling/test_drisc_traced_opprof.py
"""

import os

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tests.unit.test_assistant_standalone_l1 import _build_standalone, _make_fused_k_body

REPLAYS = int(os.getenv("OPPROF_REPLAYS", "2"))


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_traced_iteration_opprof(mesh_device, reset_seeds):
    rig = _build_standalone(mesh_device, "dram", tune_matmuls=os.getenv("GEMMA4_TUNE_MATMULS", "1") == "1")
    body = _make_fused_k_body(rig, 3)
    idx, hidden = body()  # compile before capture
    ttnn.synchronize_device(mesh_device)
    idx.deallocate(True)
    hidden.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    idx, hidden = body()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    idx.deallocate(True)
    hidden.deallocate(True)
