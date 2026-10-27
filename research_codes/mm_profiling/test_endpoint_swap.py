# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E6b arm (iv), isolated (gemma4-12b record §9.2e, plan v3 changelog 52): DRAM bank 0 read through another endpoint.

The production 1D-multicast in1 reader reads bank 0's tiles through the NoC0 endpoint given by TT_MM1D_IN1_BANK0_XY
(translated x | y << 6; 785 = (17, 12) = physical (0, 0)) instead of bank 0's default (0, 11). One process per arm
(the factory reads the env var when it builds a program): run once without and once with the env var, then compare.

Per shape (bf16, HiFi2, the drafter's DRAM-interleaved in0 / weight / output):
  down_proj  8192 x 1024, tuner 8 x 4 (32 cores, iw8, pcN 1)       (P1)
  gate_proj  1024 x 8192, tuner 8 x 8 (64 cores, iw8, pcN 4)       (P2)
  gate_proj  1024 x 8192, W's 86 cores (pcN 3, 11-wide prefix)     (P2)
Timing: INNER ops per trace, REPLAYS replays, host wall clock per op (as test_dram_sharded_fatk.py). Inputs are seeded;
outputs are saved to EP_OUT/<arm>.pt for the bit-identity comparison (compare_endpoint_swap in this file's __main__).

    EP_ARM=base EP_OUT=generated/dram_read_probe/ep0 pytest -q -s research_codes/mm_profiling/test_endpoint_swap.py
    TT_MM1D_IN1_BANK0_XY=785 EP_ARM=ep0 EP_OUT=... pytest ...
    python research_codes/mm_profiling/test_endpoint_swap.py generated/dram_read_probe/ep0
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

INNER, REPLAYS = 20, 30
ARM = os.getenv("EP_ARM", "base")
OUT = Path(os.getenv("EP_OUT", "generated/dram_read_probe/ep0"))


def _configs():
    from models.demos.gemma4.tt.matmul_tuning import derive_decode_1d_config

    def wide(K, N, pcn):
        Nt = N // 32
        cores = -(-Nt // pcn)
        base = derive_decode_1d_config(32, K, N)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(min(cores, 11), -(-cores // 11)),
            in0_block_w=base.in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    out = [
        ("down_proj_8x4", 8192, 1024, derive_decode_1d_config(32, 8192, 1024)),
        ("gate_proj_8x8", 1024, 8192, derive_decode_1d_config(32, 1024, 8192)),
        ("gate_proj_86", 1024, 8192, wide(1024, 8192, 3)),
    ]
    if os.getenv("EP_TAIL"):
        # the 59,840-column pin's DRAM tail (1024 x 202,304) on its best config, 66 / ow6 (lm_head_out_block_w_config):
        # 96 cores, the last with 52 valid columns of 66 (it also reads tiles past N)
        out = [("lm_tail_66_6", 1024, 202304, "tail")]
    return out


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_endpoint_swap(mesh_device):
    OUT.mkdir(parents=True, exist_ok=True)
    md = mesh_device
    if True:
        ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        outs, times = {}, {}
        for name, K, N, pc in _configs():
            g = torch.Generator().manual_seed(1234 + K + N)
            x_t = torch.randn(1, 1, 32, K, generator=g).to(torch.bfloat16)
            w_t = (torch.randn(1, 1, K, N, generator=g) * K**-0.5).to(torch.bfloat16)
            x = ttnn.from_torch(
                x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            w = ttnn.from_torch(
                w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            ck = ckc
            if isinstance(pc, str):
                from models.demos.gemma4.tt.matmul_tuning import lm_head_out_block_w_config

                os.environ["EP_TAIL_OBW"], os.environ["EP_TAIL_PCN"] = "6", "66"
                pc, ck = lm_head_out_block_w_config("EP_TAIL_OBW", x, w)

            def f():
                return ttnn.linear(
                    x, w, program_config=pc, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

            warm = [f() for _ in range(INNER)]
            ttnn.synchronize_device(md)
            outs[name] = ttnn.to_torch(warm[-1]).to(torch.bfloat16).clone()
            for o in warm:
                o.deallocate(True)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            traced = [f() for _ in range(INNER)]
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            per_round = []
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(REPLAYS):
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                per_round.append((time.perf_counter() - t0) / REPLAYS / INNER * 1e6)
            ttnn.release_trace(md, tid)
            for o in traced:
                o.deallocate(True)
            x.deallocate(True)
            w.deallocate(True)
            times[name] = per_round
            print(f"[ep] {ARM} {name}: us per op {[round(t, 2) for t in per_round]}", flush=True)
        torch.save(outs, OUT / f"{ARM}.pt")
        (OUT / f"{ARM}.json").write_text(json.dumps(dict(arm=ARM, env=os.getenv("TT_MM1D_IN1_BANK0_XY"), times=times)))


def compare(out, arm="ep0"):
    """Bit-identity of every output against the first base process; op time per arm pooled over its processes
    (5 rounds each), with the pooled SD of a round and the between-process difference within each arm."""
    import statistics

    arms = {"base": ["base", "base_b"], arm: [arm, f"{arm}_b"]}
    outs = {a: torch.load(out / f"{a}.pt") for v in arms.values() for a in v if (out / f"{a}.pt").exists()}
    tim = {a: json.loads((out / f"{a}.json").read_text())["times"] for a in outs}
    ref = outs["base"]
    for name in ref:
        same = {a: torch.equal(ref[name], o[name]) for a, o in outs.items()}
        x = {k: [t for a in v if a in tim for t in tim[a][name]] for k, v in arms.items()}
        groups = [tim[a][name] for a in tim]
        sd = (sum((len(g) - 1) * statistics.variance(g) for g in groups) / sum(len(g) - 1 for g in groups)) ** 0.5
        mb, me = statistics.mean(x["base"]), statistics.mean(x[arm])
        within = {k: [round(statistics.mean(tim[a][name]), 2) for a in v if a in tim] for k, v in arms.items()}
        print(
            f"{name}: bit-identical to base {same}; base {mb:.2f} us {within['base']}, {arm} {me:.2f} us {within[arm]};"
            f" delta {me - mb:+.2f} us ({(me - mb) / mb * 100:+.1f}%, {(me - mb) / sd:+.1f} pooled SD of a round, SD {sd:.3f})"
        )


if __name__ == "__main__":
    compare(Path(sys.argv[1]), *sys.argv[2:3])
