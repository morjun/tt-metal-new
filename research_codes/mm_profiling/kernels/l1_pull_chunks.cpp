// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Pull byte ranges from other cores' L1 into this core's L1 (research: moving pinned lm_head
// weight shards, gemma4-12b MEASUREMENT_RECORD §5.5.14 Gate 2).
//
// Compile-time arg 0: max pieces (runtime-arg capacity). Runtime args: n, then n x (src_noc_x,
// src_noc_y, src_l1_addr, dst_l1_addr, size_bytes), zero-padded to the capacity.
// All reads are issued back to back on this kernel's NoC, then one barrier.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t max_pieces = get_compile_time_arg_val(0);  // runtime-arg capacity of this program
    uint32_t idx = 0;
    uint32_t n = get_arg_val<uint32_t>(idx++);
    if (n > max_pieces) {
        n = 0;  // never walk past the runtime args (a stale or truncated arg list would read garbage)
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t x = get_arg_val<uint32_t>(idx++);
        const uint32_t y = get_arg_val<uint32_t>(idx++);
        const uint32_t src = get_arg_val<uint32_t>(idx++);
        const uint32_t dst = get_arg_val<uint32_t>(idx++);
        const uint32_t size = get_arg_val<uint32_t>(idx++);
        noc_async_read(get_noc_addr(x, y, src), dst, size);
    }
    noc_async_read_barrier();
}
