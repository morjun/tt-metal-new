// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DRAM read probe v2 (gemma4-12b record §9.2e, plan v3 harness H1-H3): one core issues a configurable stream of DRAM
// reads and logs, per block, when it issued the block, when its last word arrived and its NIU master counters, plus
// the core's first-word time. Optionally paced by an in0 multicast sender (dram_read_in0_sender.cpp, plan v3 H2).
//
// Runtime args (fixed width NARGS, so generic_op's program cache never truncates them; see l1_mover.py):
//   0 enabled (0 = idle core)
//   1 mode: 0 interleaved in1 strip, 1 direct bank stream, 2 spin (clock calibration), 4 endpoint dump
//   2 base_addr (DRAM buffer)   3 n_blocks   4 reqs_per_block   5 req_bytes
//   6 depth: 1 = full read barrier per block (production); 2..4 = transaction IDs, `depth` blocks outstanding
//   7 request VC (0..3): written into the read command buffer's NOC_CTRL, as MM_IN1_VC_MODE does, because in
//     DM_DEDICATED_NOC mode a read's VC is the sticky NOC_CTRL value, not noc_async_read's argument; restored to 1
//   mode 0 (production in1 reader order, reader_bmm_tile_layout_in1_sender_writer_padding.cpp:459-476: K rows
//          outer, N columns inner): 8 Nt, 9 c0, 10 W (per_core_N), 11 k0 -> request g: page (k0 + g / W) * Nt +
//          c0 + g % W, bank page % 8, bank address base + (page / 8) * req_bytes
//   mode 1: 8 nbanks, 9 bank0, 10 stride_bytes, 11 offset0 -> request g: bank (bank0 + g % nbanks) % 8,
//          bank address base + offset0 + (g / nbanks) * stride_bytes
//   mode 2: 8 spin ticks
//   12 dst_l1 (scratch, SCRATCH_BYTES; reads land at offsets modulo its size, so their data is not kept)
//   13 rec_l1
//   14 paced (0 / 1): in0 consumption pacing (H2). Block b is issued once this core has consumed block b - 2; it
//      consumes block b once its in1 block b arrived and in0-valid >= b + 1, then increments the sender's counter.
//      Depth 1 only.
//   15 sender translated x << 16 | y   16 sync_l1 (zeroed by the host before each run)
//   17 skip bank (0..7: requests to this bank are skipped; >= 8: none)
//   18 redirect: src bank << 8 | dst bank (requests to src go to dst at the same bank offset); 0xffff: none
//      17 / 18 act only with compile-time arg 0 bit 0 set: any per-request work in the issue loop slows a 2 KB
//      stream (17 ns per request measured, 20.2 -> 17.1 GB/s for one core; plan v3 changelog 51)
//   19 bank-0 endpoint override: translated x | y << 6 (0: the default endpoint); like the production knob
//      TT_MM1D_IN1_BANK0_XY (record §9.2e E6b arm iv). Acts only with compile-time arg 0 bit 0 (remap) or bit 1
//      (table)
//   Compile-time arg 0 bit 1 (table): every request's NoC address is a per-bank base from an 8-entry table built
//      once before the stream, with 18 / 19 applied in the table, so every request costs the same whatever the
//      endpoint. The remap path (bit 0) slows bank-0 requests by 10% even with no override (record §9.2e p1_ref)
//   20..23 reserved (0)
// sync words (16 B apart: a NoC write keeps the source's offset within 16 B): [0] consumed counter (on the sender),
//   [4] in0-valid (on each reader, set by the sender's multicast), [8] the sender's multicast source
// Records (uint32): [0] n_blocks, [1] t_start (before the first issue), [2] t_end (final barrier returned = last word
//   arrived), [3] a pacing wait timed out, [4] t_first_word (first sample with NIU_MST_RD_DATA_WORD_RECEIVED moved,
//   polled only in barrier waits and between blocks: exact to the polling interval at depth 1, an upper bound by
//   <= depth blocks' issue time at depth > 1), [5..7] 0 (requests issued are counted on the host), then per block b at
//   8 + 4b: t_issue, t_done (its barrier returned),
//   NIU_MST_RD_REQ_SENT, NIU_MST_RD_DATA_WORD_RECEIVED. Mode 4 writes dram_bank_to_noc_xy[noc][b] (x | y << 6) for
//   noc 0 / 1 at 8 + 2b / 9 + 2b, b < 8.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

namespace {
constexpr uint32_t NUM_DRAM_BANKS_PROBE = 8;
constexpr uint32_t SCRATCH_BYTES = 256 * 1024;    // test_dram_read_probe.py SCRATCH_BYTES
constexpr uint32_t SPIN_LIMIT_TICKS = 135000000;  // 100 ms: a pacing wait gives up, sets rec[3] and goes on
constexpr uint32_t REC_BLOCKS = 8;                // first per-block record

// true if *p reached v; false (and rec[3] = 1) after SPIN_LIMIT_TICKS
inline bool wait_at_least(volatile tt_l1_ptr uint32_t* p, uint32_t v, volatile tt_l1_ptr uint32_t* rec) {
    const uint32_t t0 = get_timestamp_32b();
    while (*p < v) {
        invalidate_l1_cache();
        if (get_timestamp_32b() - t0 > SPIN_LIMIT_TICKS) {
            rec[3] = 1;
            return false;
        }
    }
    return true;
}

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
    const uint32_t paced = get_arg_val<uint32_t>(14);
    const uint32_t sender_xy = get_arg_val<uint32_t>(15);
    const uint32_t sync_addr = get_arg_val<uint32_t>(16);
    const uint32_t skip_bank = get_arg_val<uint32_t>(17);
    const uint32_t redirect = get_arg_val<uint32_t>(18);
    const uint32_t bank0_xy = get_arg_val<uint32_t>(19);
    volatile tt_l1_ptr uint32_t* sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_addr);
    if (!enabled) {
        return;
    }
    rec[0] = n_blocks;
    rec[3] = 0;
    rec[4] = 0;
    rec[5] = 0;
    if (mode == 4) {
        for (uint32_t b = 0; b < NUM_DRAM_BANKS_PROBE; ++b) {
            rec[REC_BLOCKS + 2 * b] = dram_bank_to_noc_xy[0][b];
            rec[REC_BLOCKS + 2 * b + 1] = dram_bank_to_noc_xy[1][b];
        }
        return;
    }
    rec[1] = get_timestamp_32b();
    if (mode == 2) {
        const uint32_t t0 = get_timestamp_32b();
        while (get_timestamp_32b() - t0 < p0) {
        }
        rec[2] = get_timestamp_32b();
        return;
    }
    const uint32_t block_bytes = reqs * req_bytes;
    constexpr bool kRemap = get_compile_time_arg_val(0) & 1;
    constexpr bool kTable = get_compile_time_arg_val(0) & 2;
    const uint32_t redirect_src = redirect >> 8, redirect_dst = redirect & 0xff;
    uint64_t bank_base[NUM_DRAM_BANKS_PROBE];
    if constexpr (kTable) {
        for (uint32_t b = 0; b < NUM_DRAM_BANKS_PROBE; ++b) {
            const uint32_t s = b == redirect_src ? redirect_dst : b;
            const uint64_t xy = (s == 0 && bank0_xy) ? bank0_xy : dram_bank_to_noc_xy[noc_index][s];
            bank_base[b] = (xy << NOC_ADDR_COORD_SHIFT) | bank_to_dram_offset[s];
        }
    }
    set_read_vc(vc);
    const uint32_t w0 = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_DATA_WORD_RECEIVED);
    uint32_t t_first = 0;
    auto poll_first = [&]() {
        if (!t_first && NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_DATA_WORD_RECEIVED) != w0) {
            t_first = get_timestamp_32b() | 1;  // never 0
        }
    };

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
            uint64_t src;
            if constexpr (kTable) {
                src = bank_base[bank] + addr;
            } else {
                src = get_noc_addr_from_bank_id<true>(bank, addr, noc_index);
            }
            if constexpr (kRemap) {
                if (bank == skip_bank) {
                    continue;
                }
            }
            if constexpr (kRemap && !kTable) {
                if (bank == redirect_src) {
                    bank = redirect_dst;
                    src = get_noc_addr_from_bank_id<true>(bank, addr, noc_index);
                }
                if (bank == 0 && bank0_xy) {
                    src = NOC_XY_ADDR(bank0_xy & 0x3f, bank0_xy >> 6, addr + bank_to_dram_offset[0]);
                }
            }
            noc_async_read(src, dst + (slot + i * req_bytes) % SCRATCH_BYTES, req_bytes, noc_index);
        }
    };
    auto log_done = [&](uint32_t b) {
        volatile tt_l1_ptr uint32_t* r = rec + REC_BLOCKS + 4 * b;
        r[1] = get_timestamp_32b();
        r[2] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_REQ_SENT);
        r[3] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_DATA_WORD_RECEIVED);
    };

    if (depth <= 1) {
        const uint64_t consumed_sem = get_noc_addr(sender_xy >> 16, sender_xy & 0xffff, sync_addr, noc_index);
        uint32_t consumed = 0;
        auto consume_up_to = [&](uint32_t n) {  // consume blocks [consumed, n)
            while (consumed < n) {
                wait_at_least(sync + 4, consumed + 1, rec);
                ++consumed;
                noc_semaphore_inc(consumed_sem, 1, noc_index);
            }
        };
        for (uint32_t b = 0; b < n_blocks; ++b) {
            if (paced && b >= 2) {
                consume_up_to(b - 1);  // in1 CB of 2 blocks: block b needs block b - 2 consumed
            }
            rec[REC_BLOCKS + 4 * b] = get_timestamp_32b();
            issue_block(b);
            while (!ncrisc_noc_reads_flushed(noc_index)) {
                poll_first();
            }
            invalidate_l1_cache();
            log_done(b);
        }
        rec[2] = get_timestamp_32b();  // last word arrived; consumption below does not move it
        if (paced) {
            consume_up_to(n_blocks);
            noc_async_atomic_barrier(noc_index);
        }
    } else {
        // transaction IDs 1..depth, one per block in flight (as the DRAM-sharded reader, which uses 1..3)
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t trid = b % depth + 1;
            if (b >= depth) {
                while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
                    poll_first();
                }
                invalidate_l1_cache();
                log_done(b - depth);
            }
            noc_async_read_set_trid(trid, noc_index);
            wait_trid_slots(trid);
            rec[REC_BLOCKS + 4 * b] = get_timestamp_32b();
            issue_block(b);
            poll_first();
        }
        const uint32_t first = n_blocks > depth ? n_blocks - depth : 0;
        for (uint32_t b = first; b < n_blocks; ++b) {
            while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, b % depth + 1)) {
                poll_first();
            }
            invalidate_l1_cache();
            log_done(b);
        }
        noc_async_read_set_trid(0, noc_index);
        rec[2] = get_timestamp_32b();
    }
    rec[4] = t_first;
    set_read_vc(1);
}
