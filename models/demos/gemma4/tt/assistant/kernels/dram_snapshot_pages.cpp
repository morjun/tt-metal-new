// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Copy a run of this core's L1 into consecutive pages of a DRAM interleaved tensor (the DRAM copy
// that dram_reload_pages.cpp reads back; gemma4-12b MEASUREMENT_RECORD §5.5.14 Gate 4).
// Compile-time args: page_bytes, then the destination's TensorAccessorArgs.
// Runtime args: dst_addr, start_page, num_pages, src_l1_addr.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t src_addr = get_arg_val<uint32_t>(3);
    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    for (uint32_t i = 0; i < num_pages; ++i) {
        noc_async_write(src_addr + i * page_bytes, dst.get_noc_addr(start_page + i), page_bytes);
    }
    noc_async_write_barrier();
}
