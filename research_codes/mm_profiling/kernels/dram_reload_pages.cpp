// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reload a contiguous run of pages from a DRAM interleaved tensor into this core's L1 (research:
// reloading the overwritten part of a pinned lm_head after a global SDPA, gemma4-12b
// MEASUREMENT_RECORD §5.5.14).
//
// Compile-time args: page_bytes, then the source's TensorAccessorArgs.
// Runtime args: src_addr, start_page, num_pages, dst_l1_addr.
// All reads are issued back to back on this kernel's NoC, then one barrier.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t dst_addr = get_arg_val<uint32_t>(3);
    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    for (uint32_t i = 0; i < num_pages; ++i) {
        noc_async_read(src.get_noc_addr(start_page + i), dst_addr + i * page_bytes, page_bytes);
    }
    noc_async_read_barrier();
}
