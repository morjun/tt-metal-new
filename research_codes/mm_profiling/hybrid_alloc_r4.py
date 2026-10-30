#!/usr/bin/env python3
"""Allocation-only check of layout R4 under HYBRID (gemma4-12b record §9.4 S2; plan v5).

R4 puts 4 thin-K pins (pcN 4, 262,144 B/core) per core on the rectangle x 0-10 x y 4-9 (the first 64 of its 66
cores, row-major), a per-core head slice B of 16 tiles on y 0-3 (44 cores), and a lockstep head slice A of 1 tile on
all 110 cores, allocated in that order. The drafter cannot consume pins on an offset rectangle until S3, so this
allocates zero-filled buffers of the same shapes on a bare device and prints every buffer's per-core addresses.
Run with TT_METAL_ALLOCATOR_MODE_HYBRID=1. It does not include the drafter's other L1 allocations.
"""

import ttnn

K = 1024
T = 32


def shard_mc(cores, width, per_core):
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores]),
            [K, width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    if per_core:
        mc.experimental_set_per_core_allocation(True)
    return mc


def addresses(t, cores):
    if t.is_per_core_allocated():
        return {c: t.experimental_per_core_buffer_address(ttnn.CoreCoord(*c)) for c in cores}
    return {c: t.buffer_address() for c in cores}


def main():
    device = ttnn.open_device(device_id=0)
    try:
        grid = device.compute_with_storage_grid_size()
        rect = [(x, y) for y in range(4, grid.y) for x in range(grid.x)][:64]
        top = [(x, y) for y in range(0, 4) for x in range(grid.x)]
        allc = [(x, y) for y in range(grid.y) for x in range(grid.x)]
        keep = []
        plan = [(f"pin {i}", rect, 4 * T, True) for i in range(4)]
        plan += [("slice B (16 tiles)", top, 16 * T, True), ("slice A (1 tile)", allc, 1 * T, False)]
        for name, cores, width, per_core in plan:
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, K, width * len(cores)]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                device,
                shard_mc(cores, width, per_core),
            )
            keep.append(t)
            a = addresses(t, cores)
            print(
                f"R4 {name}: {'per-core' if per_core else 'lockstep'} on {len(cores)} cores, "
                f"addresses {sorted(set(a.values()))}, {K * width * 2} B/core",
                flush=True,
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
