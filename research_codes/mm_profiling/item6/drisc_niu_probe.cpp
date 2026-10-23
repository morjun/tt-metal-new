// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// gemma4-12b record §9.2, item 6b, DRAM side: resident DRISC probe of this DRAM endpoint's NIU slave counters, the
// DRAM-side half of a direct per-bank read-bandwidth measurement (test_drisc_niu_probe.py).
//
// Scope: it runs on the 16 DRISC senders Metal exposes, two of each bank's three DRAM endpoints: the NOC1 endpoint and
// the one that serves neither NoC. The NOC0 endpoint, which serves the 1D-mcast in1 reader, is owned by syseng
// firmware and runs no DRISC firmware (tt_metal/llrt/metal_soc_descriptor.hpp:47-53), so it is never sampled here and
// no conclusion about NOC0 reads follows from this probe. Record §9.2 uses no result of it.
//
// Launched once (ttnn experimental launch_resident_drisc_kernel), outside the command queue. Waits on stream
// GO_STREAM's counter, which drisc_reload_trigger.cpp raises (an inline write over NOC0 adds the command value):
//   1 = start sampling, >= 3 = quit. While sampling, every `interval` wall-clock ticks it appends one record to
//   its L1 buffer until the buffer is full or a quit arrives.
// L1 layout at runtime arg 0 (buf): buf[0] = records written, buf[1] = state (0 waiting, 1 sampling, 2 done),
//   records from buf[4], 6 words each:
//   t (wall clock, 32 b), NOC0 RD_DATA_WORD_SENT, NOC1 RD_DATA_WORD_SENT, NOC0 RD_REQ_RECEIVED, NOC1 RD_REQ_RECEIVED, 0
// Compile-time args: go_stream, max_records, interval (ticks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/drisc_mode.h"
#include "noc_overlay_parameters.h"
#include "risc_common.h"

namespace {

constexpr uint32_t go_stream = get_compile_time_arg_val(0);
constexpr uint32_t max_records = get_compile_time_arg_val(1);
constexpr uint32_t interval = get_compile_time_arg_val(2);
constexpr uint32_t CMD_START = 1, CMD_QUIT = 3;

inline uint32_t stream_count(uint32_t stream) {
    return NOC_STREAM_READ_REG(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
           ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
}

// Consume and return the pending command count (0 = none).
inline uint32_t take_cmd() {
    const uint32_t cmd = stream_count(go_stream);
    if (cmd != 0) {
        NOC_STREAM_WRITE_REG(
            go_stream,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            (-cmd) << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }
    return cmd;
}

}  // namespace

void kernel_main() {
    volatile uint32_t* buf = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0));
    // Stream mode on NIU0, as drisc_reload_resident.cpp: without it the trigger's +1 can arrive as +3 (= quit).
    experimental::drisc_set_stream_mode();
    NOC_STREAM_WRITE_REG(go_stream, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);  // zero the go counter
    buf[0] = 0;
    buf[1] = 0;

    uint32_t cmd = 0;
    while ((cmd = take_cmd()) == 0) {
    }
    if (cmd >= CMD_QUIT) {
        buf[1] = 2;
        experimental::drisc_set_noc2axi_mode();
        return;
    }
    buf[1] = 1;
    uint32_t n = 0;
    bool quit = false;
    uint32_t next = get_timestamp_32b();
    while (n < max_records) {
        while (static_cast<int32_t>(get_timestamp_32b() - next) < 0) {
        }
        next += interval;
        volatile uint32_t* r = buf + 4 + 6 * n;
        r[0] = get_timestamp_32b();
        r[1] = NOC_STATUS_READ_REG(0, NIU_SLV_RD_DATA_WORD_SENT);
        r[2] = NOC_STATUS_READ_REG(1, NIU_SLV_RD_DATA_WORD_SENT);
        r[3] = NOC_STATUS_READ_REG(0, NIU_SLV_RD_REQ_RECEIVED);
        r[4] = NOC_STATUS_READ_REG(1, NIU_SLV_RD_REQ_RECEIVED);
        r[5] = 0;
        buf[0] = ++n;
        if (take_cmd() >= CMD_QUIT) {
            quit = true;
            break;
        }
    }
    // the buffer filled first: wait for the quit
    while (!quit && take_cmd() < CMD_QUIT) {
    }
    buf[1] = 2;
    experimental::drisc_set_noc2axi_mode();  // leave the firmware default
}
