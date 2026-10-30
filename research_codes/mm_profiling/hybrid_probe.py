#!/usr/bin/env python3
"""Is the device open in the HYBRID allocator mode? (gemma4-12b record §9.4, S1 engagement check.)

tt-metal logs no allocator mode, so this allocates two single-core per-core L1 tensors on cores (0, 0) and (1, 0).
Per-core allocation is accepted only in HYBRID mode (allocator.cpp:144); in LOCKSTEP it throws. Prints
"HYBRID: per-core addresses ..." or "LOCKSTEP: ..." and exits 0 either way; run it with and without
TT_METAL_ALLOCATOR_MODE_HYBRID=1 through the same environment path as the drafter cells.
"""

import torch

import ttnn


def per_core_tensor(device, core, nbytes):
    spec = ttnn.ShardSpec(ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [1, nbytes], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    mc.experimental_set_per_core_allocation(True)
    data = torch.zeros(1, nbytes, dtype=torch.uint8)
    return ttnn.from_torch(data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)


def main():
    device = ttnn.open_device(device_id=0)
    try:
        cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
        try:
            ts = [per_core_tensor(device, c, 1024) for c in cores]
        except RuntimeError as e:
            print(f"LOCKSTEP: per-core allocation refused ({str(e).splitlines()[0][:160]})", flush=True)
            return
        addrs = [t.experimental_per_core_buffer_address(c) for t, c in zip(ts, cores)]
        print(f"HYBRID: per-core addresses {addrs} on (0, 0), (1, 0)", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
