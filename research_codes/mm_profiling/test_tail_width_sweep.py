# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated width sweep of the drafter's lm_head DRAM tail (gemma4-12b record §9.4, "why the tail's tiles get slower").

The tail matmul alone, as the drafter runs it: in0 [1, 1, 1, 1024] bf16 DRAM-interleaved, weight [1, 1, 1024, N] bf16
DRAM-interleaved, logits L1-interleaved, 1D multicast on the 11 x 10 grid, in0_block_w 2, HiFi2, packer_l1_acc, no fp32
dest. Production knobs come from the environment (the factory reads them when it builds a program):
TT_MM1D_IN1_VC_MODE=1 (vc1), TT_MM1D_IN1_BANK0_XY=785 (ep0).

Configs (tiles = N / 32; cores = ceil(tiles / per_core_N)):
  - the drafter's tail widths: 6,322 (F0) and 6,652 (b) at 66 / ow6; 6,580 (H1), 6,652, 7,354, 8,192 at 75 / ow5
  - per_core_N 66, ow6, full cores 80 ... 110 (tiles = 66 x cores): the time as cores are added at fixed work per core
  - per_core_N 75, ow5, full cores 80 ... 109
Timing: INNER calls in one trace, REPLAYS replays, host wall clock per call; ROUNDS rounds, config order reversed in
odd rounds. Writes TW_OUT/<arm>.json.

    TT_MM1D_IN1_VC_MODE=1 TT_MM1D_IN1_BANK0_XY=785 TW_ARM=prod TW_OUT=generated/lmhead_l1_profile/tail_width_sweep \
      pytest -q -s research_codes/mm_profiling/test_tail_width_sweep.py
"""

import json
import os
import statistics
from pathlib import Path

from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from research_codes.mm_profiling.test_lmhead_out_block_w import GRID, _ckc, _pc, _timed

ROUNDS = int(os.getenv("TW_ROUNDS", "3"))
OUT = Path(os.getenv("TW_OUT", "generated/lmhead_l1_profile/tail_width_sweep"))
ARM = os.getenv("TW_ARM", "prod")
K = 1024


def configs():
    out = [(6322, 66, 6), (6652, 66, 6), (6580, 75, 5), (6652, 75, 5), (7354, 75, 5), (8192, 75, 5)]
    out += [(66 * c, 66, 6) for c in range(80, 111, 2)]
    out += [(75 * c, 75, 5) for c in range(80, 110, 3)]
    return list(dict.fromkeys(out))


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_tail_width_sweep(mesh_device, reset_seeds):
    ckc = _ckc(mesh_device)
    x = ttnn.rand([1, 1, 1, K], device=mesh_device, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    cfgs = configs()
    times = {c: [] for c in cfgs}
    for r in range(ROUNDS):
        for tiles, pcn, obw in cfgs if r % 2 == 0 else cfgs[::-1]:
            w = ttnn.rand([1, 1, K, tiles * 32], device=mesh_device, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
            pc = _pc(pcn, obw)

            def fn():
                return ttnn.linear(
                    x, w, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG
                )

            us = _timed(mesh_device, fn)
            w.deallocate(True)
            times[(tiles, pcn, obw)].append(us)
            logger.info(f"[tail-width] round {r} tiles {tiles} pcN {pcn} ow{obw}: {us:.2f} us/call")
    rows = [
        {
            "tiles": t,
            "per_core_N": p,
            "out_block_w": o,
            "cores": -(-t // p),
            "us": v,
            "median_us": statistics.median(v),
            "ns_per_tile": statistics.median(v) / t * 1e3,
        }
        for (t, p, o), v in times.items()
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    env = {k: os.getenv(k) for k in ("TT_MM1D_IN1_VC_MODE", "TT_MM1D_IN1_BANK0_XY")}
    (OUT / f"{ARM}.json").write_text(
        json.dumps({"arm": ARM, "env": env, "grid": [GRID.x, GRID.y], "rows": rows}, indent=1)
    )
    for row in sorted(rows, key=lambda d: (d["per_core_N"], d["tiles"])):
        logger.info(
            f"[tail-width] {row['tiles']:5d} tiles pcN {row['per_core_N']} cores {row['cores']:3d}: "
            f"{row['median_us']:8.2f} us  {row['ns_per_tile']:.2f} ns/tile"
        )
