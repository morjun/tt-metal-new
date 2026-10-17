#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Why GEMMA4_LMHEAD_LOGITS_L1 is refused on the whole (unsplit) drafter head (gemma4-12b record §8).

One 1024 x N matmul, random bf16 weights, logits written to DRAM vs L1, for the tail's explicit config
(N 202,304, per_core_N 64 / w8), the whole head at w5, and the whole head on ttnn's automatic config.
Then, with the logits in L1, either plain ttnn.untilize (to DRAM; keeps the 32-row pad for an L1 input) or
spec_decode._untilize_logits (untilize_with_unpadding to DRAM), and ttnn.argmax directly on the row, as the
dense path of argmax_last does. Also whether each explicit config's DRAM logits equal the automatic config's.

    PYTHONPATH=<overlay>:$PWD TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD python check_logits_l1_whole.py
"""

import os

import torch

import ttnn
from models.demos.gemma4.tt.matmul_tuning import lm_head_out_block_w_config

from models.demos.gemma4.tt.spec_decode import _untilize_logits

dev = ttnn.open_device(device_id=0, l1_small_size=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 1, 1024).bfloat16()
xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def argmax_of(u):
    return ttnn.to_torch(ttnn.argmax(u, dim=-1, keepdim=False)).reshape(-1)[0].item()


for N, pcn, obw in [(202304, "64", "8"), (262144, "", "5"), (262144, "", "75"), (262144, "", "")]:
    os.environ["GEMMA4_LMHEAD_DRAFT_OBW"] = obw
    os.environ["GEMMA4_LMHEAD_DRAFT_PCN"] = pcn
    w = (torch.randn(1024, N) * 0.03).bfloat16()
    wt = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    pc, ckc = lm_head_out_block_w_config("GEMMA4_LMHEAD_DRAFT_OBW", xt, wt)
    ya = ttnn.to_torch(ttnn.linear(xt, wt)).float().reshape(-1)  # automatic config, DRAM: arm A's logits
    yd = ttnn.to_torch(ttnn.linear(xt, wt, program_config=pc, compute_kernel_config=ckc)).float().reshape(-1)
    host = int(yd.argmax())
    out = [
        f"N={N} pcn={pcn or 'min'} obw={obw or 'auto'}: DRAM logits == auto-config DRAM logits {torch.equal(yd, ya)}"
    ]
    for name, untilize in (
        ("plain untilize", lambda y: ttnn.untilize(y, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)),
        ("untilize_with_unpadding", _untilize_logits),
    ):
        y = ttnn.linear(xt, wt, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
        yl = ttnn.to_torch(y).float().reshape(-1)
        u = untilize(y)
        ok_vals = torch.equal(ttnn.to_torch(u).float().reshape(-1), yl)
        y.deallocate(True)
        a = argmax_of(u)
        out.append(
            f"  L1 logits + {name}: L1==DRAM logits {torch.equal(yl, yd)} (diff cols {(yl != yd).sum().item()}); "
            f"row == logits {ok_vals}; row padded {list(u.padded_shape)} {u.memory_config().buffer_type}; "
            f"argmax {a} vs host {host} -> {'OK' if a == host else 'WRONG'}"
        )
        u.deallocate(True)
    print("\n".join(out), flush=True)
    wt.deallocate(True)
ttnn.close_device(dev)
