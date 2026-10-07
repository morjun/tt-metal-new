// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// SDPA stand-in for gemma4-12b MEASUREMENT_RECORD §5.5.14 Gate 4: fill this core's whole CB c_0
// (declared by the host to span the global SDPA's CB stack, up to 801,024) with a junk pattern,
// so everything a real SDPA could overwrite IS overwritten.
// Runtime args: n_bytes (the CB's size), pattern.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t n_bytes = get_arg_val<uint32_t>(0);
    const uint32_t pattern = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
    for (uint32_t i = 0; i < n_bytes / 4; ++i) {
        p[i] = pattern ^ i;
    }
}
