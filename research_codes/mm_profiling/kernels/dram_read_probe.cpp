// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DRAM read probe (gemma4-12b record §9.2a, plan v2 T4): one core issues a configurable stream of DRAM reads and
// logs, per block, when it issued the block, when the block's data had all arrived, and its NIU master counters.
//
// Runtime args (fixed width NARGS, so generic_op's program cache never truncates them; see l1_mover.py):
//   0 enabled (0 = idle core)        1 mode: 0 interleaved pages, 1 direct bank stream, 2 spin (clock calibration)
//   2 base_addr (DRAM buffer)        3 n_blocks        4 reqs_per_block     5 req_bytes
//   6 depth: 1 = full read barrier per block; 2..4 = transaction IDs, `depth` blocks in flight
//   7 request VC (0..3): written into the read command buffer's NOC_CTRL, as MM_IN1_VC_MODE does, because in
//     DM_DEDICATED_NOC mode a read's VC is the sticky NOC_CTRL value, not noc_async_read's argument; restored to 1
//   mode 0 (a matmul's in1 strip): 8 Nt, 9 c0, 10 W, 11 k0 -> request g: page (k0 + g / W) * Nt + c0 + g % W,
//          bank page % 8, bank address base + (page / 8) * req_bytes  (the interleaved layout of a 2 KB-tile tensor)
//   mode 1: 8 nbanks, 9 bank0, 10 stride_bytes, 11 offset0 -> request g: bank (bank0 + g % nbanks) % 8,
//          bank address base + offset0 + (g / nbanks) * stride_bytes
//   mode 2: 8 spin ticks
//   12 dst_l1 (scratch, SCRATCH_BYTES; reads land at offsets modulo its size, so their data is not kept)   13 rec_l1
// Records (uint32): [0] n_blocks, [1] t_start, [2] t_end, then per block b at 4 + 4b:
//   t_issue (before the block's first read), t_done (after its barrier), NIU_MST_RD_REQ_SENT,
//   NIU_MST_RD_DATA_WORD_RECEIVED
// With depth > 1, t_done is when the block's transaction-ID barrier returned: an upper bound on its arrival.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

namespace {
constexpr uint32_t NUM_DRAM_BANKS_PROBE = 8;
constexpr uint32_t SCRATCH_BYTES = 256 * 1024;  // test_dram_read_probe.py SCRATCH_BYTES

inline void set_read_vc(uint32_t vc) {
    while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        read_cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc));
}

inline void wait_trid_slots(uint32_t trid) {
    while (noc_available_transactions(noc_index, trid) < ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2)) {
    }
}
}  // namespace

void kernel_main() {
    const uint32_t enabled = get_arg_val<uint32_t>(0);
    const uint32_t mode = get_arg_val<uint32_t>(1);
    const uint32_t base = get_arg_val<uint32_t>(2);
    const uint32_t n_blocks = get_arg_val<uint32_t>(3);
    const uint32_t reqs = get_arg_val<uint32_t>(4);
    const uint32_t req_bytes = get_arg_val<uint32_t>(5);
    const uint32_t depth = get_arg_val<uint32_t>(6);
    const uint32_t vc = get_arg_val<uint32_t>(7);
    const uint32_t p0 = get_arg_val<uint32_t>(8);
    const uint32_t p1 = get_arg_val<uint32_t>(9);
    const uint32_t p2 = get_arg_val<uint32_t>(10);
    const uint32_t p3 = get_arg_val<uint32_t>(11);
    const uint32_t dst = get_arg_val<uint32_t>(12);
    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(13));
    if (!enabled) {
        return;
    }
    rec[0] = n_blocks;
    rec[1] = get_timestamp_32b();
    if (mode == 2) {
        const uint32_t t0 = get_timestamp_32b();
        while (get_timestamp_32b() - t0 < p0) {
        }
        rec[2] = get_timestamp_32b();
        return;
    }
    const uint32_t block_bytes = reqs * req_bytes;
    set_read_vc(vc);

    auto issue_block = [&](uint32_t b) {
        const uint32_t slot = (depth > 1 ? (b % depth) : 0) * block_bytes;
        for (uint32_t i = 0; i < reqs; ++i) {
            const uint32_t g = b * reqs + i;
            uint32_t bank, addr;
            if (mode == 0) {
                const uint32_t page = (p3 + g / p2) * p0 + p1 + g % p2;
                bank = page % NUM_DRAM_BANKS_PROBE;
                addr = base + (page / NUM_DRAM_BANKS_PROBE) * req_bytes;
            } else {
                bank = (p1 + g % p0) % NUM_DRAM_BANKS_PROBE;
                addr = base + p3 + (g / p0) * p2;
            }
            noc_async_read(
                get_noc_addr_from_bank_id<true>(bank, addr, noc_index),
                dst + (slot + i * req_bytes) % SCRATCH_BYTES,
                req_bytes,
                noc_index);
        }
    };
    auto log_done = [&](uint32_t b) {
        volatile tt_l1_ptr uint32_t* r = rec + 4 + 4 * b;
        r[1] = get_timestamp_32b();
        r[2] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_REQ_SENT);
        r[3] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_DATA_WORD_RECEIVED);
    };

    if (depth <= 1) {
        for (uint32_t b = 0; b < n_blocks; ++b) {
            rec[4 + 4 * b] = get_timestamp_32b();
            issue_block(b);
            noc_async_read_barrier(noc_index);
            log_done(b);
        }
    } else {
        // transaction IDs 1..depth, one per block in flight (as the DRAM-sharded reader, which uses 1..3)
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t trid = b % depth + 1;
            if (b >= depth) {
                noc_async_read_barrier_with_trid(trid, noc_index);
                log_done(b - depth);
            }
            noc_async_read_set_trid(trid, noc_index);
            wait_trid_slots(trid);
            rec[4 + 4 * b] = get_timestamp_32b();
            issue_block(b);
        }
        const uint32_t first = n_blocks > depth ? n_blocks - depth : 0;
        for (uint32_t b = first; b < n_blocks; ++b) {
            noc_async_read_barrier_with_trid(b % depth + 1, noc_index);
            log_done(b);
        }
        noc_async_read_set_trid(0, noc_index);
    }
    rec[2] = get_timestamp_32b();
    set_read_vc(1);
}
