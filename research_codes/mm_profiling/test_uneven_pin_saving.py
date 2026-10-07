# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone timing of an uneven drafter lm_head pin and of over-specified tail per_core_N
(gemma4-12b MEASUREMENT_RECORD §5.5.14, schedule C step 1).

Arms, all on the whole drafter head (K=1024, N=262,144, bf16, random weights), with every part
timed together in one trace and the joined output checked bit-exact against the unsplit head:

- base: 38,720 uniform (11 tiles x 110 cores) + tail at per_core_N 64, width 8 (§5.5.13 arm D).
- over-specified tail per_core_N ("pcn"): the tail at a per_core_N larger than ceil(Nt / 110), so
  fewer than 110 cores carry it and a composite width becomes legal. This targets the prime
  per_core_N of 49,280 (61) and 56,320 (59), and tests per_core_N 64 at every pin size.
- uneven: 38,720 on 110 cores (11 tiles) + a second L1 slice of k tiles/core on 46 cores + tail.
  k = 6 is schedule C's static depth (46 x 17-tile cores): 47,552 cols; k = 5: 46,080.
  PROXY PLACEMENT: the stock 1D matmul runs on a row-major prefix of a (0,0)-anchored rectangle
  (matmul_multicore_reuse_mcast_1d_program_factory.cpp:5213-5240), so the second slice sits on
  the first 46 cores (rows 0-3 + (0,4), (1,4)), not on the 46 cores outside the 8x8 SDPA block.
  Same core count and tiles per core; different placement. The real layout needs a custom
  matmul or sub-devices.

There is no SDPA here, so no 801,024 ceiling: the lm_head alone, as test_pin_size_saving.py. A
part whose CBs clash with the L1 slices is recorded as "no fit".

    pytest research_codes/mm_profiling/test_uneven_pin_saving.py -s
Writes generated/lmhead_l1_profile/pin_size_saving/uneven.json.
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

K, N, T = 1024, 262_144, 32
ROUNDS = int(os.getenv("UNEVEN_ROUNDS", "3"))
FULL = [(x, y) for y in range(10) for x in range(11)]


def arm(name, segments, tail_pcn, widths):
    """segments: [(col_start, col_end, n_cores)], each an L1 slice on the first n_cores (row-major)."""
    return {"name": name, "segments": segments, "tail_pcn": tail_pcn, "widths": widths}


def pcn_min(cols):
    return -(-(-(-(N - cols) // 110)) // T)


ARMS = [arm("base_38720", [(0, 38_720, 110)], 64, [8])]
for cols in (42_240, 45_760, 49_280, 52_800, 56_320):
    ARMS.append(arm(f"u{cols}_pcn64", [(0, cols, 110)], 64, [4, 8, 16]))
ARMS += [
    arm("u49280_pcn62", [(0, 49_280, 110)], 62, [2, 31]),
    arm("u56320_pcn60", [(0, 56_320, 110)], 60, [2, 4, 6, 10, 12]),
    arm("u56320_pcn62", [(0, 56_320, 110)], 62, [2, 31]),
    arm("uneven_k6_47552", [(0, 38_720, 110), (38_720, 47_552, 46)], 62, [2, 31]),
    arm("uneven_k6_47552_pcn64", [(0, 38_720, 110), (38_720, 47_552, 46)], 64, [4, 8, 16]),
    arm("uneven_k5_46080", [(0, 38_720, 110), (38_720, 46_080, 46)], 62, [2, 31]),
    arm("uneven_k5_46080_pcn64", [(0, 38_720, 110), (38_720, 46_080, 46)], 64, [4, 8, 16]),
]


def crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def seg_pc(per_core_n):
    # The model's slice config (_split_lm_head_to_l1): in0_block_w 2, subblock 1, fuse_batch.
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_uneven_pin_saving(mesh_device, reset_seeds):
    dev = mesh_device
    ckc = _ckc(dev)
    seg_ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    x = ttnn.rand([1, 1, 1, K], device=dev, dtype=ttnn.bfloat16, low=-1.0, high=1.0, seed=1)
    w = ttnn.rand([1, 1, K, N], device=dev, dtype=ttnn.bfloat16, low=-0.05, high=0.05, seed=2)
    ref = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc)).float().reshape(-1)[:N]

    def build(a):
        segs = []
        for c0, c1, n in a["segments"]:
            per_core_cols = (c1 - c0) // n
            head = ttnn.slice(w, [0, 0, 0, c0], [1, 1, K, c1])
            spec = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs(FULL[:n]), [K, per_core_cols], ttnn.ShardOrientation.ROW_MAJOR),
            )
            segs.append((ttnn.to_memory_config(head, spec), per_core_cols // T))
            head.deallocate(True)
        last = a["segments"][-1][1]
        tail = ttnn.slice(w, [0, 0, 0, last], [1, 1, K, N])
        return segs, tail, last

    def call(segs, tail, a, width):
        tpc = _pc(a["tail_pcn"], width)
        return lambda: tuple(
            [ttnn.linear(x, s, program_config=seg_pc(p), compute_kernel_config=seg_ckc) for s, p in segs]
            + [ttnn.linear(x, tail, program_config=tpc, compute_kernel_config=ckc)]
        )

    times, exact, nofit, info = {}, {}, {}, {}
    for r in range(ROUNDS):
        for a in ARMS if r % 2 == 0 else list(reversed(ARMS)):
            try:
                segs, tail, last = build(a)
            except RuntimeError as e:
                for wd in a["widths"]:
                    nofit[f"{a['name']}/{wd}"] = "build: " + str(e).splitlines()[0][:300]
                continue
            info[a["name"]] = {
                "slice_addrs": [s.buffer_address() for s, _ in segs],
                "pinned_cols": last,
                "tail_cols": N - last,
                "tail_pcn": a["tail_pcn"],
                "tail_cores": -(-((N - last) // T) // a["tail_pcn"]),
                "tail_pcn_min": pcn_min(last),
            }
            for wd in a["widths"] if r % 2 == 0 else list(reversed(a["widths"])):
                key = f"{a['name']}/{wd}"
                if key in nofit:
                    continue
                fn = call(segs, tail, a, wd)
                if key not in exact:
                    try:
                        outs = fn()
                    except RuntimeError as e:
                        nofit[key] = str(e).splitlines()[0][:300]
                        logger.info(f"[uneven] {key}: does not fit / rejected: {nofit[key]}")
                        continue
                    sizes = [c1 - c0 for c0, c1, _ in a["segments"]] + [N - last]
                    joined = torch.cat([ttnn.to_torch(o).float().reshape(-1)[:n] for o, n in zip(outs, sizes)])
                    exact[key] = bool(torch.equal(joined, ref))
                    for o in outs:
                        o.deallocate(True)
                times.setdefault(key, []).append(_timed_multi(dev, fn))
            for s, _ in segs:
                s.deallocate(True)
            tail.deallocate(True)

    base = statistics.median(times["base_38720/8"])
    rows = []
    for a in ARMS:
        for wd in a["widths"]:
            key = f"{a['name']}/{wd}"
            row = {"arm": a["name"], "width": wd, **info.get(a["name"], {})}
            if key in nofit:
                row["no_fit"] = nofit[key]
                logger.info(f"[uneven] {key:28s} no fit / rejected")
            else:
                med = statistics.median(times[key])
                row.update({"us": med, "runs": times[key], "bit_exact": exact[key], "saved_vs_base_us": base - med})
                logger.info(
                    f"[uneven] {key:28s} pinned={row['pinned_cols']:>6} tail_pcn={row['tail_pcn']} "
                    f"(min {row['tail_pcn_min']}, {row['tail_cores']} cores) {med:8.2f} us  "
                    f"saved {base - med:+7.2f} us  exact={exact[key]}  runs={[round(v, 2) for v in times[key]]}  "
                    f"slices@{row['slice_addrs']}"
                )
            rows.append(row)
    out = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/pin_size_saving/uneven.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base_us": base, "rows": rows, "rounds": ROUNDS}, indent=1) + "\n")
    logger.info(f"[uneven] wrote {out}")
    x.deallocate(True)
    w.deallocate(True)
    assert all(exact.values())
