#!/usr/bin/env python3
"""S3 isolated correctness under HYBRID (gemma4-12b record §9.4; plan v5). Run with TT_METAL_ALLOCATOR_MODE_HYBRID=1.

Every check compares an L1-pinned matmul with the same matmul on a DRAM-interleaved copy of the weight, through the
drafter's own code paths (weight_placement.shard_l1_width, DecodeMatmulTuner, assistant.model.build_head_slice_b), and
requires the outputs to be bit-identical (torch.equal):
  T1  thin-K weight [1024, 8192] per-core on the 86-core prefix (per_core_N 3), tuner config;
  T2  the same weight per-core on the offset rectangle from row 4 (64 cores, per_core_N 4; allowed_worker_cores);
  T3  lm_head-style slice B at 12 and 3 tiles/core on the complement of the 86-core prefix (2 + 22 cores), iw2 /
      HiFi2 / packer L1 acc, against the same columns on an origin grid.
Buffers of one check are freed before the next, so every per-core buffer is the first per-core allocation on its cores.
Prints one PASS / FAIL line per comparison and exits non-zero on any FAIL.
"""

import sys

import torch

import ttnn
from models.demos.gemma4.tt.assistant.model import build_head_slice_b
from models.demos.gemma4.tt.matmul_tuning import DecodeMatmulTuner
from models.demos.gemma4.tt.weight_placement import l1_core_addresses, shard_l1_width

K = 1024


def main():
    torch.manual_seed(0)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    fails = 0
    try:
        grid = dev.compute_with_storage_grid_size()
        tuner = DecodeMatmulTuner(dev, enabled=True, label="s3")
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, K),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def dram(t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def check(tag, a, b):
            nonlocal fails
            ok = torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} {tag}", flush=True)

        w = dram(torch.randn(1, 1, K, 8192) * 0.02)
        ckc = tuner.compute_config_for(x, w)
        tuned_ref = tuner.linear(x, w)  # the drafter's ordinary DRAM-weight call (tuner grid)

        for tag, pcn, row in (("T1 per-core 86-core prefix", 3, 0), ("T2 per-core offset rows 4-9", 4, 4)):
            wp = shard_l1_width(w, dev, per_core_n=pcn, per_core=True, origin_row=row)
            addrs = sorted(set(l1_core_addresses(wp).values()))
            pc = tuner.config_for(x, wp)
            print(
                f"{tag}: shard grid {wp.memory_config().shard_spec.grid}, addresses {addrs}, "
                f"grid {pc.compute_with_storage_grid_size}, allowed {getattr(pc, 'allowed_worker_cores', '?')}",
                flush=True,
            )
            out_l1 = tuner.linear(x, wp)
            out_ref = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc)
            check(f"{tag}: L1 vs DRAM weight, same config", out_l1, out_ref)
            check(f"{tag}: L1 vs the tuner's DRAM call", out_l1, tuned_ref)
            for t in (wp, out_l1, out_ref):
                t.deallocate(True)

        hckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        for tiles in (12, 3):
            n = tiles * 32 * (grid.x * grid.y - 86)
            head = dram(torch.randn(1, 1, K, n) * 0.02)
            slices, end = build_head_slice_b(head, 0, tiles, 86, grid, 2)
            parts = [ttnn.linear(x, wb, program_config=pcb, compute_kernel_config=hckc) for wb, pcb in slices]
            got = torch.cat([ttnn.to_torch(p) for p in parts], dim=-1)
            cores = -(-(n // 32) // tiles)
            ref_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, -(-cores // grid.x)),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=tiles,
                per_core_M=1,
                per_core_N=tiles,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            ref = ttnn.to_torch(ttnn.linear(x, head, program_config=ref_pc, compute_kernel_config=hckc))
            ok = end == n and torch.equal(got, ref)
            fails += not ok
            print(
                f"{'PASS' if ok else 'FAIL'} T3 slice B {tiles} tiles/core on {[str(w.memory_config().shard_spec.grid) for w, _ in slices]}: "
                f"vs the same columns on an origin grid ({cores} cores)",
                flush=True,
            )
            for wb, _ in slices:
                wb.deallocate(True)
            for p in parts:
                p.deallocate(True)
            head.deallocate(True)
    finally:
        ttnn.close_mesh_device(dev)
    print(f"S3 {'PASS' if fails == 0 else f'FAIL ({fails})'}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
