// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Copy a byte range of one row-major DRAM row into another row at an offset (tt/assistant/model.py,
// _join_split_logits: the two parts of the split lm_head's logits into one vocab row).
//
// ttnn.concat stages a whole output row (524 KiB for the vocab) in L1 on one core, which reaches into the
// pinned lm_head slice there. This kernel moves the row through a `chunk`-byte scratch CB instead, so its
// L1 footprint is `chunk` bytes per core, and each core handles its own byte range.
//
// Both tensors are [1, 1, 1, N] ROW_MAJOR in DRAM: one page (the whole row), in one bank.
// Compile-time args: chunk bytes, src row bytes, dst row bytes, then the src and dst TensorAccessorArgs.
// Runtime args: src buffer address, dst buffer address, src byte offset, dst byte offset, bytes.
// CB 0: `chunk` bytes of scratch.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t chunk = get_compile_time_arg_val(0);
    constexpr uint32_t src_row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t dst_row_bytes = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t src_off = get_arg_val<uint32_t>(2);
    const uint32_t dst_off = get_arg_val<uint32_t>(3);
    const uint32_t bytes = get_arg_val<uint32_t>(4);
    const auto src = TensorAccessor(src_args, src_addr, src_row_bytes);
    const auto dst = TensorAccessor(dst_args, dst_addr, dst_row_bytes);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t done = 0; done < bytes; done += chunk) {
        const uint32_t n = bytes - done < chunk ? bytes - done : chunk;
        noc_async_read(src.get_noc_addr(0, src_off + done), scratch, n);
        noc_async_read_barrier();
        noc_async_write(scratch, dst.get_noc_addr(0, dst_off + done), n);
        noc_async_write_barrier();
    }
}
