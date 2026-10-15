#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-level check of GEMMA4_LMHEAD_LOGITS_L1 on the split drafter head (gemma4-12b record §8, logits in L1).

The 59,840-col L1 slice (width-sharded on 110 cores, the model's config) and the DRAM tail (per_core_N 64 /
out_block_w 8), random bf16 weights: logits written to DRAM vs L1, then the model's chain
(spec_decode._untilize_logits on both parts -> concat -> argmax). Prints whether L1 and DRAM logits are
bit-identical and the device vs host argmax. The slice weights are freed before the concat, as the reload
bracket does in the model.

    PYTHONPATH=<overlay>:$PWD TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD python check_logits_l1_split.py
"""

import os

import torch

import ttnn
from models.demos.gemma4.tt.spec_decode import _untilize_logits
from models.demos.gemma4.tt.matmul_tuning import lm_head_out_block_w_config

dev = ttnn.open_device(device_id=0, l1_small_size=0)
torch.manual_seed(1)
S, N, K, T = 59840, 262144, 1024, 32
g = dev.compute_with_storage_grid_size()
cores = g.x * g.y
x = torch.randn(1, 1, 1, K).bfloat16()
w = (torch.randn(K, N) * 0.03).bfloat16()
xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
shard = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))}),
    [K, S // cores],
    ttnn.ShardOrientation.ROW_MAJOR,
)
wh = ttnn.from_torch(
    w[:, :S],
    layout=ttnn.TILE_LAYOUT,
    device=dev,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard),
)
wt = ttnn.from_torch(w[:, S:], layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
pc_h = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=g,
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=1,
    per_core_M=1,
    per_core_N=S // cores // T,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
)
os.environ["GEMMA4_LMHEAD_DRAFT_OBW"], os.environ["GEMMA4_LMHEAD_DRAFT_PCN"] = "8", "64"
pc_t, ckc_t = lm_head_out_block_w_config("GEMMA4_LMHEAD_DRAFT_OBW", xt, wt)
res, rms = {}, {}
for name, mc in (("dram", None), ("l1", ttnn.L1_MEMORY_CONFIG)):
    h = ttnn.linear(xt, wh, program_config=pc_h, compute_kernel_config=ckc, memory_config=mc)
    t = ttnn.linear(xt, wt, program_config=pc_t, compute_kernel_config=ckc_t, memory_config=mc)
    print(name, "logits in", h.memory_config().buffer_type, t.memory_config().buffer_type)
    res[name] = torch.cat([ttnn.to_torch(h).float().reshape(-1), ttnn.to_torch(t).float().reshape(-1)])
    rms[name] = [_untilize_logits(p) for p in (h, t)]
    print(
        name,
        "untilized",
        [list(r.padded_shape) for r in rms[name]],
        [str(r.memory_config().buffer_type) for r in rms[name]],
    )
    h.deallocate(True)
    t.deallocate(True)
wh.deallocate(True)  # the model frees the head around the concat (reload bracket)
for name in ("dram", "l1"):
    j = ttnn.concat(rms[name], dim=-1)
    a = ttnn.to_torch(ttnn.argmax(j, dim=-1, keepdim=False)).reshape(-1)[0].item()
    print(name, "joined", list(j.padded_shape), torch.equal(ttnn.to_torch(j).float().reshape(-1), res[name]))
    print(f"{name}: device argmax {a}, host argmax {int(res[name].argmax())}")
print("L1 logits == DRAM logits:", torch.equal(res["l1"], res["dram"]))
ttnn.close_device(dev)
