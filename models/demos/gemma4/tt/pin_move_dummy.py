# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Research knob: in-situ cost of moving pinned lm_head bytes around the global-layer SDPAs.

Off unless GEMMA4_PIN_MOVE_DUMMY_B is set (gemma4-12b MEASUREMENT_RECORD §5.5.14).

GEMMA4_PIN_MOVE_DUMMY_B=<bytes per SDPA core> runs, around every global-layer SDPA decode, the L1 -> L1
round trip that moving that many head bytes per SDPA core would need:
- out before the SDPA: the parking cores pull from the SDPA cores;
- back after it: the SDPA cores pull from the parking cores.

Core sets match the standalone test (research_codes/mm_profiling/test_pin_move_cost.py):
- batch 1 (drafter): 8x4 -> the other 78 cores;
- otherwise (target): 8x8 -> the other 46 cores. The target SDPA itself still runs on its own grid.

The data are junk in scratch tensors that are freed before the SDPA runs, so the SDPA's CBs never
see them. Only the moves' time is real.

GEMMA4_PIN_MOVE_DUMMY_RISCS=1|2 (default 2) picks one RISC/NoC or both.
"""

import os

from loguru import logger

_BYTES = int(os.environ.get("GEMMA4_PIN_MOVE_DUMMY_B", "0"))
_BOTH_RISCS = os.environ.get("GEMMA4_PIN_MOVE_DUMMY_RISCS", "2") == "2"


def enabled(head_dim):
    return _BYTES > 0 and head_dim >= 512


def _move(device, batch_size, direction):
    import ttnn
    from research_codes.mm_profiling.l1_mover import core_set, l1_sharded, round_trip_programs, share_bytes

    gx, gy = (8, 4) if batch_size == 1 else (8, 8)
    src_cores = core_set(lambda x, y: x < gx and y < gy)
    dst_cores = core_set(lambda x, y: not (x < gx and y < gy))
    src = l1_sharded(device, src_cores, _BYTES)
    park = l1_sharded(device, dst_cores, share_bytes(len(src_cores), _BYTES, len(dst_cores)))
    s_addr, p_addr = src.buffer_address(), park.buffer_address()
    out_prog, back_prog = round_trip_programs(device, src_cores, dst_cores, _BYTES, s_addr, p_addr, s_addr, _BOTH_RISCS)
    if direction == "out":
        ttnn.generic_op([src, park], out_prog)
    else:
        ttnn.generic_op([park, src], back_prog)
    src.deallocate(True)
    park.deallocate(True)
    logger.info(
        f"[pin-move-dummy] {direction} batch={batch_size} {gx}x{gy}->{len(dst_cores)} cores "
        f"{_BYTES} B/core riscs={2 if _BOTH_RISCS else 1} src@{s_addr} park@{p_addr}"
    )


def move_out(device, batch_size):
    _move(device, batch_size, "out")


def move_back(device, batch_size):
    _move(device, batch_size, "back")
