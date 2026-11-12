// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tensix side of the resident DRISC reload (drisc_reload_resident.cpp, gemma4-12b MEASUREMENT_RECORD
// §5.5.14): spin until every DRISC has finished the command the trigger sent. Each poll, over NOC0
// (the DRISCs' NIU0 is in stream mode, so reads land in their L1 and registers at local addresses):
//   1. read every DRISC's go counter (one barrier); any nonzero = a command not taken yet, poll again;
//   2. then read every DRISC's (taken, done) counters (one barrier); any taken != done = still pushing.
// A DRISC raises `taken` before it consumes the go count, so go == 0 seen in step 1 means step 2
// sees the current command. Nothing is reset, so the trace replays this kernel unchanged.
// (One DRISC at a time, two round trips each, cost 11.95 us per wait when all were idle.)
//
// Gives up after timeout_iters polls in total instead of hanging the device; the reload is then
// incomplete, which the bit-exact checks downstream catch.
//
// Compile-time args: num_driscs, go_stream, sync_addr (DRISC L1), timeout_iters.
// A DRISC with bit 31 of its word set writes on NoC1 (its NIU1 is in stream mode) and is polled over NOC1:
// its NIU0 is in NOC2AXI mode, where a read of a local address returns GDDR, not its L1.
//
// Runtime args: num_driscs words (noc1 << 31 | noc_x << 16 | noc_y). CB 0: 32 * num_driscs B of scratch.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "noc_overlay_parameters.h"

void kernel_main() {
    constexpr uint32_t num_driscs = get_compile_time_arg_val(0);
    constexpr uint32_t go_stream = get_compile_time_arg_val(1);
    constexpr uint32_t sync_addr = get_compile_time_arg_val(2);
    constexpr uint32_t timeout_iters = get_compile_time_arg_val(3);
    constexpr uint32_t go_reg = STREAM_REG_ADDR(go_stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    const uint32_t go_buf = get_write_ptr(0);            // 16 B per DRISC
    const uint32_t sync_buf = go_buf + 16 * num_driscs;  // 16 B per DRISC
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_buf);
    volatile tt_l1_ptr uint32_t* sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_buf);
    noc_local_state_init(1 - noc_index);  // the kernel's own NoC is initialised by the firmware; the other is not
    for (uint32_t iters = 0; iters < timeout_iters; ++iters) {
        for (uint32_t d = 0; d < num_driscs; ++d) {
            const uint32_t xy = get_arg_val<uint32_t>(d);
            const uint8_t noc = xy >> 31 ? 1 : 0;
            noc_async_read(
                get_noc_addr((xy >> 16) & 0x7FFF, xy & 0xFFFF, go_reg, noc), go_buf + 16 * d, sizeof(uint32_t), noc);
        }
        noc_async_read_barrier(0);
        noc_async_read_barrier(1);
        bool taken = true;
        for (uint32_t d = 0; d < num_driscs; ++d) {
            taken = taken && (go[4 * d] & ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1)) == 0;
        }
        if (!taken) {
            continue;
        }
        for (uint32_t d = 0; d < num_driscs; ++d) {
            const uint32_t xy = get_arg_val<uint32_t>(d);
            const uint8_t noc = xy >> 31 ? 1 : 0;
            noc_async_read(
                get_noc_addr((xy >> 16) & 0x7FFF, xy & 0xFFFF, sync_addr, noc),
                sync_buf + 16 * d,
                2 * sizeof(uint32_t),
                noc);
        }
        noc_async_read_barrier(0);
        noc_async_read_barrier(1);
        bool done = true;
        for (uint32_t d = 0; d < num_driscs; ++d) {
            done = done && sync[4 * d] == sync[4 * d + 1];
        }
        if (done) {
            break;
        }
    }
}
