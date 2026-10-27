// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// in0 multicast generator for the DRAM read probe (gemma4-12b record §9.2e, plan v3 H2). Runs on RISCV_1 (NoC1) of the
// first reader core, as the 1D-multicast matmul's in0 sender does on DRAM-interleaved in0. Per block b it waits until
// every reader has consumed block b - 2 (in0 CB of 2 blocks), multicasts one in0 block (block_bytes) from L1 into the
// readers' in0 slot b % 2, then multicasts in0-valid = b + 1. Readers consume and count (dram_read_probe.cpp, paced).
//
// Runtime args (fixed width NARGS_SENDER):
//   0 enabled   1 n_blocks   2 n_readers (consumed-counter target per block)   3 sync_l1   4 in0_src_l1
//   5 in0_dst_l1 (2 slots of block_bytes)   6 block_bytes   7 n_rects (1..4)
//   8..11  rectangle i: start_x << 24 | start_y << 16 | end_x << 8 | end_y, translated coordinates, already in this
//          NoC's start / end order (the host swaps them for NoC1, as the matmul factory does)
//   12..15 rectangle i: number of destinations | loopback << 31 (loopback when the sender core is inside it; the
//          count then includes the sender)
// sync words: [0] consumed counter (incremented by the readers), [4] in0-valid (destination on the readers),
//   [8] this kernel's multicast source; sender record at sync word 16: [16] a wait timed out, [17] t_start,
//   [18] t_end, [20 + b] time block b's in0-valid was sent.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

namespace {
constexpr uint32_t SPIN_LIMIT_TICKS = 135000000;  // 100 ms
}

void kernel_main() {
    const uint32_t enabled = get_arg_val<uint32_t>(0);
    const uint32_t n_blocks = get_arg_val<uint32_t>(1);
    const uint32_t n_readers = get_arg_val<uint32_t>(2);
    const uint32_t sync_addr = get_arg_val<uint32_t>(3);
    const uint32_t src = get_arg_val<uint32_t>(4);
    const uint32_t dst = get_arg_val<uint32_t>(5);
    const uint32_t block_bytes = get_arg_val<uint32_t>(6);
    const uint32_t n_rects = get_arg_val<uint32_t>(7);
    if (!enabled) {
        return;
    }
    volatile tt_l1_ptr uint32_t* sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_addr);
    sync[16] = 0;
    sync[17] = get_timestamp_32b();
    auto wait_consumed = [&](uint32_t v) {
        const uint32_t t0 = get_timestamp_32b();
        while (sync[0] < v) {
            invalidate_l1_cache();
            if (get_timestamp_32b() - t0 > SPIN_LIMIT_TICKS) {
                sync[16] = 1;
                return;
            }
        }
    };
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (b >= 2) {
            wait_consumed((b - 1) * n_readers);
        }
        const uint32_t slot = dst + (b % 2) * block_bytes;
        for (uint32_t i = 0; i < n_rects; ++i) {
            const uint32_t r = get_arg_val<uint32_t>(8 + i);
            const uint32_t nd = get_arg_val<uint32_t>(12 + i);
            const uint64_t a =
                get_noc_multicast_addr(r >> 24, (r >> 16) & 0xff, (r >> 8) & 0xff, r & 0xff, slot, noc_index);
            if (nd >> 31) {
                noc_async_write_multicast_loopback_src(src, a, block_bytes, nd & 0x7fffffff, false, noc_index);
            } else {
                noc_async_write_multicast(src, a, block_bytes, nd, false, noc_index);
            }
        }
        sync[8] = b + 1;
        for (uint32_t i = 0; i < n_rects; ++i) {
            const uint32_t r = get_arg_val<uint32_t>(8 + i);
            const uint32_t nd = get_arg_val<uint32_t>(12 + i);
            const uint64_t a =
                get_noc_multicast_addr(r >> 24, (r >> 16) & 0xff, (r >> 8) & 0xff, r & 0xff, sync_addr + 16, noc_index);
            if (nd >> 31) {
                noc_semaphore_set_multicast_loopback_src(sync_addr + 32, a, nd & 0x7fffffff, false, noc_index);
            } else {
                noc_semaphore_set_multicast(sync_addr + 32, a, nd, false, noc_index);
            }
        }
        noc_async_write_barrier(noc_index);  // sync[8] is rewritten for the next block
        sync[20 + b] = get_timestamp_32b();
    }
    wait_consumed(n_blocks * n_readers);
    sync[18] = get_timestamp_32b();
}
