#!/usr/bin/env python3
"""S1 isolated correctness of the spread thin-K pin (gemma4-12b record §9.5, stage 1). Run on build_codex_current
(the run_lmhead_e2e overlay) with TT_MM1D_IN1_SPREAD=1.

A spread pin (weight_placement.shard_l1_width(..., spread=True)) width-shards a thin-K weight at per_core_N 3 on all
110 cores: the first Nt - 2 * 110 cores hold 3 real tiles, the others 2 real tiles + 1 zero pad tile. The 1D-mcast
factory (TT_MM1D_IN1_SPREAD) runs it on all 110 cores and its writer drops the pad tiles. For N = 8192 (36 x 3 +
74 x 2) and N = 9216 (68 x 3 + 42 x 2) this compares, bitwise (torch.equal):
  (a) the spread pin vs the same 11 x 10 / per_core_N 3 config with the DRAM weight (only placement differs);
  (b) the spread pin vs the existing prefix pin at per_core_N 3 (86 / 96 cores);
  (c) the spread pin vs the tuner's ordinary DRAM call (8 x 8 grid).
Without the factory mode the op would read the spread bytes as a prefix layout and fail all three, so a PASS also
shows that the knob engaged. Prints one PASS / FAIL line per comparison and exits non-zero on any FAIL.
"""

import sys

import torch

import ttnn
from models.demos.gemma4.tt.matmul_tuning import DecodeMatmulTuner
from models.demos.gemma4.tt.weight_placement import l1_core_addresses, shard_l1_width

K = 1024


def main():
    torch.manual_seed(0)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    fails = 0
    try:
        tuner = DecodeMatmulTuner(dev, enabled=True, label="spread")
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, K),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def check(tag, a, b):
            nonlocal fails
            ok = torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} {tag}", flush=True)

        for n in (8192, 9216):
            w = ttnn.from_torch(
                torch.randn(1, 1, K, n) * 0.02,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ckc = tuner.compute_config_for(x, w)
            tuned_ref = tuner.linear(x, w)
            prefix = shard_l1_width(w, dev, per_core_n=3)
            out_prefix = tuner.linear(x, prefix)
            prefix_cores = prefix.memory_config().shard_spec.grid.num_cores()
            prefix.deallocate(True)

            ws = shard_l1_width(w, dev, per_core_n=3, spread=True)
            pc = tuner.config_for(x, ws)
            addrs = sorted(set(l1_core_addresses(ws).values()))
            print(
                f"N={n}: spread shard grid {ws.memory_config().shard_spec.grid.num_cores()} cores at {addrs}, "
                f"config grid {pc.compute_with_storage_grid_size} per_core_N {pc.per_core_N} "
                f"in0_block_w {pc.in0_block_w}",
                flush=True,
            )
            out_spread = tuner.linear(x, ws)
            out_ref = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc)
            check(f"N={n} (a) spread vs the same 11x10 config, DRAM weight", out_spread, out_ref)
            check(f"N={n} (b) spread vs the {prefix_cores}-core prefix pin", out_spread, out_prefix)
            check(f"N={n} (c) spread vs the tuner's DRAM call", out_spread, tuned_ref)
            for t in (ws, w, out_spread, out_ref, out_prefix, tuned_ref):
                t.deallocate(True)
    finally:
        ttnn.close_mesh_device(dev)
    print(f"S1 {'PASS' if fails == 0 else f'FAIL ({fails})'}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
