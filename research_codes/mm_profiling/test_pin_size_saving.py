# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measured op-level saving of a larger drafter lm_head pin (gemma4-12b MEASUREMENT_RECORD §5.5.14).

The "moving pinned weights" study modelled the saving of pinning more columns: 3 heads x (extra DRAM
bytes / 396 GB/s - extra L1-slice time). This measures it instead.

The whole drafter head (K=1024, N=262,144, bf16) is split as the model splits it, at
38,720 / 42,240 / 45,760 / 49,280 / 52,800 / 56,320 pinned columns:
- the L1 slice: width-sharded on 11x10, with the exact config of `_split_lm_head_to_l1`
  (tt/assistant/model.py): in0_block_w 2, subblock 1, per_core_N = cols / 3,520, fuse_batch, HiFi2,
  packer L1 acc;
- the DRAM tail: the config of `matmul_tuning.lm_head_out_block_w_config` (GEMMA4_LMHEAD_DRAFT_OBW),
  at every legal out_block_w.

Slice + tail are timed together in one trace (as test_lmhead_out_block_w's whole-head test).

There is no SDPA here, so there is no 801,024 CB ceiling: this is the lm_head alone. A tail width
whose CBs clash with the slice (host-side CB check) is recorded as "no fit"; the demo would hit the
same wall. Width 1 is included only where per_core_N is prime (otherwise only 1 pass).

Order: sizes forward / reverse / forward, every width per size, ROUNDS rounds. Every arm's
slice+tail output must equal the unsplit auto head bit-for-bit.
Writes generated/lmhead_l1_profile/pin_size_saving/saving.json.

    pytest research_codes/mm_profiling/test_pin_size_saving.py -s
"""

import json
import os
import statistics
import sys
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

sys.path.insert(0, str(Path(__file__).parent))
from test_lmhead_out_block_w import _ckc, _pc, _timed_multi  # noqa: E402

K, N, CORES, T = 1024, 262_144, 110, 32
SIZES = [int(c) for c in os.getenv("SAVING_SIZES", "38720,42240,45760,49280,52800,56320").split(",")]
ROUNDS = int(os.getenv("SAVING_ROUNDS", "3"))
BASE_COLS, BASE_WIDTH = 38_720, 8  # arm D of §5.5.13
MODEL_GBPS, SLICE_US_PER_COL = 396e9, 42.6 / 38_720  # the Gate 1 model


def tail_pcn(cols):
    return -(-(-(-(N - cols) // CORES)) // T)


def widths(pcn):
    divs = [d for d in range(1, pcn + 1) if pcn % d == 0]
    return divs if len(divs) == 2 else [d for d in divs if d > 1]  # prime: 1 and itself


def slice_pc(cols):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=cols // T // CORES,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_pin_size_saving(mesh_device, reset_seeds):
    dev = mesh_device
    ckc = _ckc(dev)
    slice_ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    x = ttnn.rand([1, 1, 1, K], device=dev, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    w = ttnn.rand([1, 1, K, N], device=dev, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
    ref = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc)).float().reshape(-1)[:N]

    def build(cols):
        head = ttnn.slice(w, [0, 0, 0, 0], [1, 1, K, cols])
        tail = ttnn.slice(w, [0, 0, 0, cols], [1, 1, K, N])
        shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))}),
            [K, cols // CORES],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        pinned = ttnn.to_memory_config(
            head, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)
        )
        head.deallocate(True)
        return pinned, tail

    def pair(pinned, tail, cols, width):
        spc, tpc = slice_pc(cols), _pc(tail_pcn(cols), width)
        return lambda: (
            ttnn.linear(x, pinned, program_config=spc, compute_kernel_config=slice_ckc),
            ttnn.linear(x, tail, program_config=tpc, compute_kernel_config=ckc),
        )

    times, exact, nofit, addr = {}, {}, {}, {}
    for r in range(ROUNDS):
        for cols in SIZES if r % 2 == 0 else list(reversed(SIZES)):
            pinned, tail = build(cols)
            addr[cols] = pinned.buffer_address()
            ws = widths(tail_pcn(cols))
            for width in ws if r % 2 == 0 else list(reversed(ws)):
                key = f"{cols}/{width}"
                if key in nofit:
                    continue
                fn = pair(pinned, tail, cols, width)
                if key not in exact:
                    try:
                        s_out, t_out = fn()
                    except RuntimeError as e:  # host-side CB clash with the slice
                        nofit[key] = str(e).splitlines()[0][:300]
                        logger.info(f"[saving] {key}: does not fit: {nofit[key]}")
                        continue
                    joined = torch.cat(
                        [
                            ttnn.to_torch(s_out).float().reshape(-1)[:cols],
                            ttnn.to_torch(t_out).float().reshape(-1)[: N - cols],
                        ]
                    )
                    exact[key] = bool(torch.equal(joined, ref))
                    s_out.deallocate(True)
                    t_out.deallocate(True)
                times.setdefault(key, []).append(_timed_multi(dev, fn))
            pinned.deallocate(True)
            tail.deallocate(True)

    base = statistics.median(times[f"{BASE_COLS}/{BASE_WIDTH}"])
    rows = []
    for cols in SIZES:
        extra = cols - BASE_COLS
        model_saved = extra * K * 2 / MODEL_GBPS * 1e6 - extra * SLICE_US_PER_COL
        for width in widths(tail_pcn(cols)):
            key = f"{cols}/{width}"
            row = {"pin_cols": cols, "slice_addr": addr[cols], "tail_per_core_N": tail_pcn(cols), "width": width}
            if key in nofit:
                row["no_fit"] = nofit[key]
            else:
                med = statistics.median(times[key])
                row.update(
                    {
                        "us": med,
                        "runs": times[key],
                        "bit_exact": exact[key],
                        "saved_vs_base_us": base - med,
                        "model_saved_us": model_saved,
                    }
                )
            rows.append(row)
            logger.info(
                f"[saving] {cols:>6} cols slice@{addr[cols]:>8} tail per_core_N={tail_pcn(cols)} width={width:>2}: "
                + (
                    f"no fit"
                    if key in nofit
                    else f"{row['us']:8.2f} us  saved vs {BASE_COLS}/w{BASE_WIDTH} {row['saved_vs_base_us']:+7.2f} us "
                    f"(model {model_saved:+6.2f})  exact={row['bit_exact']}  runs={[round(v, 2) for v in times[key]]}"
                )
            )
    out = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/pin_size_saving/saving.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"base": f"{BASE_COLS}/{BASE_WIDTH}", "base_us": base, "rows": rows, "rounds": ROUNDS}, indent=1)
        + "\n"
    )
    logger.info(f"[saving] wrote {out}")
    x.deallocate(True)
    w.deallocate(True)
    assert all(exact.values())
