#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Why GEMMA4_LMHEAD_LOGITS_L1 is refused on the whole (unsplit) drafter head (gemma4-12b record §8).

One 1024 x N matmul, random bf16 weights, logits written to DRAM vs L1, for the tail's explicit config
(N 202,304, per_core_N 64 / w8), the whole head at w5, and the whole head on ttnn's automatic config.
Then ttnn.untilize (to DRAM) and ttnn.argmax directly on it, as the dense path of argmax_last does.

    PYTHONPATH=<overlay>:$PWD TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD python check_logits_l1_whole.py
"""

import os

import torch

import ttnn
from models.demos.gemma4.tt.matmul_tuning import lm_head_out_block_w_config

dev = ttnn.open_device(device_id=0, l1_small_size=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 1, 1024).bfloat16()
xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
for N, pcn, obw in [(202304, "64", "8"), (262144, "", "5"), (262144, "", "")]:
    os.environ["GEMMA4_LMHEAD_DRAFT_OBW"] = obw
    os.environ["GEMMA4_LMHEAD_DRAFT_PCN"] = pcn
    w = (torch.randn(1024, N) * 0.03).bfloat16()
    wt = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    pc, ckc = lm_head_out_block_w_config("GEMMA4_LMHEAD_DRAFT_OBW", xt, wt)
    yd = ttnn.to_torch(ttnn.linear(xt, wt, program_config=pc, compute_kernel_config=ckc)).float().reshape(-1)
    y = ttnn.linear(xt, wt, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
    yl = ttnn.to_torch(y).float().reshape(-1)
    u = ttnn.untilize(y, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ut = ttnn.to_torch(u).float().reshape(-1)
    a = ttnn.to_torch(ttnn.argmax(u, dim=-1, keepdim=False)).reshape(-1)[0].item()
    ud = ttnn.untilize(
        ttnn.from_torch(yd.reshape(1, 1, 1, -1).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev), use_multicore=True
    )
    ad = ttnn.to_torch(ttnn.argmax(ud, dim=-1, keepdim=False)).reshape(-1)[0].item()
    print(
        f"N={N} pcn={pcn or 'min'} obw={obw or 'auto'}: L1==DRAM logits {torch.equal(yl, yd)} "
        f"(diff cols {(yl != yd).sum().item()}); untilize==L1 logits {torch.equal(ut, yl)}; "
        f"argmax(L1 path) {a}, argmax(DRAM logits) {ad}, host {int(yd.argmax())}, u shape {list(u.shape)} {u.memory_config().buffer_type}"
    )
    for t in (y, u, ud, wt):
        t.deallocate(True)
ttnn.close_device(dev)
