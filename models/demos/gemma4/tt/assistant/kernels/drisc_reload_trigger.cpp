// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tensix side of the resident DRISC reload (drisc_reload_resident.cpp, gemma4-12b MEASUREMENT_RECORD
// §5.5.14): raise every DRISC's go counter by `cmd` (1 = reload the small core set, 2 = all cores,
// 3 = quit) and return. Runs on one core, on NOC0: measured (test_drisc_signal.py), one inline register
// write over NOC1 adds 3 to the DRISC's counter, over NOC0 (NIU in stream mode) exactly 1.
//
// A DRISC whose kernel writes on NoC1 (bit 31 of its word set; its NIU1 is the one in stream mode) is
// reached over NOC1 instead: its NIU0 stays in NOC2AXI mode, where a write is not delivered to its registers.
//
// Compile-time args: cmd, go_stream, num_driscs. Runtime args: num_driscs words
// (noc1 << 31 | noc_x << 16 | noc_y).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "noc_overlay_parameters.h"

void kernel_main() {
    constexpr uint32_t cmd = get_compile_time_arg_val(0);
    constexpr uint32_t go_stream = get_compile_time_arg_val(1);
    constexpr uint32_t num_driscs = get_compile_time_arg_val(2);
    constexpr uint32_t reg = STREAM_REG_ADDR(go_stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    noc_local_state_init(1 - noc_index);  // the kernel's own NoC is initialised by the firmware; the other is not
    for (uint32_t d = 0; d < num_driscs; ++d) {
        const uint32_t xy = get_arg_val<uint32_t>(d);
        const uint8_t noc = xy >> 31 ? 1 : 0;
        noc_inline_dw_write<InlineWriteDst::REG>(
            get_noc_addr((xy >> 16) & 0x7FFF, xy & 0xFFFF, reg, noc), cmd << REMOTE_DEST_BUF_WORDS_FREE_INC, 0xF, noc);
    }
    noc_async_write_barrier(0);
    noc_async_write_barrier(1);
}
