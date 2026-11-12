// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Resident DRISC kernel: the lm_head reload of lm_head_reload.py, run on the DRAM cores alongside the
// Tensix ops before the lm_head (gemma4-12b MEASUREMENT_RECORD §5.5.14, reload hiding).
//
// Launched once, outside the command queue (ttnn experimental launch_resident_drisc_kernel). Loops:
//   1. wait on stream GO_STREAM's counter, which a Tensix trigger kernel (drisc_reload_trigger.cpp)
//      raises by a command: 1 = reload the small core set, 2 = reload all cores, 3 = quit;
//   2. for every core of that set assigned to this DRISC: DMA this bank's pages of the core's
//      snapshot rows from GDDR into DRISC L1 and NoC-write each 2 KB page to [S + i * page) on the core;
//   3. after every write is acked, count the command done.
// Two counters in this DRISC's L1, right after the DMA buffers, carry the state: commands taken
// (raised before the go count is consumed) and commands done. drisc_reload_wait.cpp reads them and
// the go register over NOC0 and is done when go == 0 and taken == done on every DRISC. The state
// lives here, not on the Tensix waiter: Tensix firmware zeroes all 64 CB stream counters after every
// program (init_sync_registers), which would erase a done count sent there.
//
// The snapshot is lm_head_reload.py's DRAM interleaved tensor: page p = core_index * pages + i, page p
// in bank p % num_banks at bank address snap_base + (p / num_banks) * page. So one core's pages in this
// bank are contiguous in GDDR and land every num_banks pages in its L1.
//
// Compile-time args: page_bytes, chunk_pages (pages per DMA, two chunks double-buffered in DRISC L1),
//   go_stream, pace (wall-clock ticks per page; 0 = unpaced). Paced, the n-th page of a push is not
//   issued before start + n * pace, which caps this DRISC at page_bytes per pace ticks.
// sync[2] holds the wall-clock ticks the last push took (debugging).
// Runtime args: snap_base, num_banks, bank, pages, dst_l1 (S), drisc_l1_buf, n_small, n_all, noc,
//   then n_small (core_index, noc_x << 16 | noc_y) pairs, then n_all pairs. `noc` is the NoC this DRISC
//   writes on: 0 (the firmware keeps NIU 0 in stream mode on every Metal-owned DRISC).
// With DIAG_SIGNAL defined the kernel only records the go counts it reads (sync[2..9], no push) and quits on
// a count >= 3 (test_drisc_signal.py).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"
#include "noc_overlay_parameters.h"
#include "risc_common.h"

namespace {

constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
constexpr uint32_t chunk_pages = get_compile_time_arg_val(1);
constexpr uint32_t go_stream = get_compile_time_arg_val(2);
constexpr uint32_t pace = get_compile_time_arg_val(3);
constexpr uint8_t dma_stream = 0;
constexpr uint32_t CMD_SMALL = 1, CMD_ALL = 2, CMD_QUIT = 3;

inline uint32_t stream_count(uint32_t stream) {
    return NOC_STREAM_READ_REG(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
           ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
}

struct Job {
    uint64_t snap_base;
    uint32_t num_banks, bank, pages, dst_l1;
};

// One DMA chunk: `n` pages of core `ci` (index into its pair list) starting at its k-th page in this bank.
struct Chunk {
    uint32_t ci, k, n;
};

// Number of this bank's pages among core c's pages, and the first one's offset within the core.
inline void bank_run(const Job& j, uint32_t c, uint32_t& first, uint32_t& count) {
    const uint32_t base = c * j.pages;
    first = (j.bank + j.num_banks - base % j.num_banks) % j.num_banks;
    count = first < j.pages ? (j.pages - first + j.num_banks - 1) / j.num_banks : 0;
}

void reload(const Job& j, uint32_t n_cores, uint32_t arg0, uint32_t buf0, uint8_t noc_id) {
    Noc noc(noc_id);
    UnicastEndpoint src;
    UnicastEndpoint dst;
    const uint32_t bufs[2] = {buf0, buf0 + chunk_pages * page_bytes};
    // Flat chunk sequence over (core, chunk of its run), double-buffered: DMA chunk t+1 while chunk t is
    // written out. A buffer is refilled only after the writes out of it have left L1 (flushed).
    auto next = [&](Chunk c) -> Chunk {
        uint32_t first, count;
        bank_run(j, get_arg_val<uint32_t>(arg0 + 2 * c.ci), first, count);
        c.k += c.n;
        while (c.ci < n_cores) {
            bank_run(j, get_arg_val<uint32_t>(arg0 + 2 * c.ci), first, count);
            if (c.k < count) {
                c.n = count - c.k < chunk_pages ? count - c.k : chunk_pages;
                return c;
            }
            ++c.ci;
            c.k = 0;
        }
        c.n = 0;
        return c;
    };
    const uint32_t t0 = get_timestamp_32b();
    uint32_t issued = 0;  // pages issued so far in this push
    auto issue_dma = [&](const Chunk& c, uint32_t buf) {
        if constexpr (pace != 0) {
            while (static_cast<int32_t>(get_timestamp_32b() - (t0 + issued * pace)) < 0) {
            }
        }
        issued += c.n;
        uint32_t first, count;
        const uint32_t core = get_arg_val<uint32_t>(arg0 + 2 * c.ci);
        bank_run(j, core, first, count);
        const uint32_t p = core * j.pages + first + c.k * j.num_banks;  // global page index, in this bank
        experimental::dma_async_read(
            dma_stream, j.snap_base + (uint64_t)(p / j.num_banks) * page_bytes, buf, c.n * page_bytes);
    };
    Chunk cur = next({0, 0, 0});
    if (cur.n == 0) {
        return;
    }
    issue_dma(cur, bufs[0]);
    for (uint32_t t = 0; cur.n != 0; ++t) {
        const Chunk nxt = next(cur);
        if (nxt.n != 0) {
            noc.async_writes_flushed();  // writes out of bufs[(t + 1) % 2] (chunk t - 1) left L1
            issue_dma(nxt, bufs[(t + 1) % 2]);
        }
        experimental::dma_async_read_wait_n(dma_stream, nxt.n != 0 ? 1 : 0);  // chunk t landed
        uint32_t first, count;
        const uint32_t core = get_arg_val<uint32_t>(arg0 + 2 * cur.ci);
        const uint32_t xy = get_arg_val<uint32_t>(arg0 + 2 * cur.ci + 1);
        bank_run(j, core, first, count);
        for (uint32_t q = 0; q < cur.n; ++q) {
            const uint32_t i = first + (cur.k + q) * j.num_banks;  // page within the core
            noc.async_write(
                src,
                dst,
                page_bytes,
                {.addr = bufs[t % 2] + q * page_bytes},
                {.noc_x = xy >> 16, .noc_y = xy & 0xFFFF, .addr = j.dst_l1 + i * page_bytes});
        }
        cur = nxt;
    }
    noc.async_write_barrier();
}

}  // namespace

void kernel_main() {
    Job j;
    j.snap_base = get_arg_val<uint32_t>(0);
    j.num_banks = get_arg_val<uint32_t>(1);
    j.bank = get_arg_val<uint32_t>(2);
    j.pages = get_arg_val<uint32_t>(3);
    j.dst_l1 = get_arg_val<uint32_t>(4);
    const uint32_t buf0 = get_arg_val<uint32_t>(5);
    const uint32_t n_small = get_arg_val<uint32_t>(6);
    const uint32_t n_all = get_arg_val<uint32_t>(7);
    const uint8_t noc_id = get_arg_val<uint32_t>(8);
    // sync[0] = commands taken, sync[1] = commands done (read by drisc_reload_wait.cpp).
    volatile uint32_t* sync = reinterpret_cast<volatile uint32_t*>(buf0 + 2 * chunk_pages * page_bytes);

    // The DRISC firmware puts NIU 0 of every Metal-owned DRISC in stream mode at boot (experimental/drisc_mode.h),
    // which lets this kernel initiate NoC writes on NoC0; kernels never switch NIU modes.
    noc_local_state_init(noc_id);  // drisck.cc initialises only NOC_INDEX's counters
    NOC_STREAM_WRITE_REG(go_stream, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);  // zero the go counter
    for (uint32_t k = 0; k < 10; ++k) {
        sync[k] = 0;
    }

    while (true) {
        uint32_t cmd;
        while ((cmd = stream_count(go_stream)) == 0) {
        }
        sync[0] = sync[0] + 1;  // taken, before the go count drops to 0
        NOC_STREAM_WRITE_REG(
            go_stream,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            (-cmd) << REMOTE_DEST_BUF_WORDS_FREE_INC);
#ifdef DIAG_SIGNAL
        sync[2 + (sync[0] - 1) % 8] = cmd;
        sync[1] = sync[0];
        if (cmd >= CMD_QUIT) {
            break;
        }
        continue;
#endif
        if (cmd != CMD_SMALL && cmd != CMD_ALL) {
            break;  // CMD_QUIT (or a corrupt count): stop
        }
        const uint32_t t = get_timestamp_32b();
        if (cmd == CMD_SMALL) {
            reload(j, n_small, 9, buf0, noc_id);
        } else {
            reload(j, n_all, 9 + 2 * n_small, buf0, noc_id);
        }
        sync[2] = get_timestamp_32b() - t;
        sync[1] = sync[1] + 1;  // done: reload() returned after every write was acked
    }
}
