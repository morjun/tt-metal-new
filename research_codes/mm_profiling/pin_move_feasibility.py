#!/usr/bin/env python3
"""Can a larger lm_head pin survive by MOVING its overlapping part away from colliding CBs?

For every candidate pin size (uniform 3,520-column shard steps on 110 cores, i.e. 65,536 B/bank
each), find every op invocation of one iteration whose program-local CBs reach above the head's
start address S on some core, how many bytes would have to move off those cores (CB end - S per
colliding core), and how much free L1 the *other* cores have below S at that moment (parking room).
Then compare the bytes moved per iteration (out and back) with the DRAM bytes the extra pinned
columns save in the 3 drafter lm_head calls per iteration.

    python pin_move_feasibility.py CAPTURE_DIR [--target-sdpa-grid 8x8] [--json OUT]

`--target-sdpa-grid` models re-gridding the full-grid global-layer SDPA (the 801,024 stack on 110
cores) to an X x Y sub-grid with the same per-core CB stack (num_cores_per_head saturates at 16).
Gain model: op-level medians of gemma4-12b MEASUREMENT_RECORD §5.5.12 (DRAM tail at 396 GB/s,
L1 slice 42.6 us per 38,720 columns). Model only; no device access.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lm_head_l1_analyze import cores  # noqa: E402

TOP, BASE = 1_572_864, 111_488
SHARD_STEP_B, SHARD_STEP_COLS = 65_536, 3_520
GRID = [(x, y) for y in range(10) for x in range(11)]
BASELINE_COLS = 38_720
DRAM_GBPS = 396e9
SLICE_US_PER_COL = 42.6 / 38_720
DRAFTER_HEADS_PER_ITER = 3


def invocations(graph):
    """Outermost *DeviceOperation spans with their per-core program-local CB end."""
    stack, out, cur = [], [], None
    for node in graph:
        kind, params = node["node_type"], node.get("params", {})
        if kind == "function_start":
            name = params.get("name", "")
            stack.append(name)
            if cur is None and name.endswith("DeviceOperation"):
                cur = {"op": name.replace("DeviceOperation", ""), "cb": collections.defaultdict(int), "d": len(stack)}
        elif kind == "function_end":
            if stack:
                stack.pop()
            if cur is not None and len(stack) < cur["d"]:
                out.append(cur)
                cur = None
        elif kind == "circular_buffer_allocate" and cur is not None and params.get("globally_allocated") == "0":
            end = int(params["address"]) + int(params["size"])
            for c in cores(params.get("core_range_set", "")):
                cur["cb"][c] = max(cur["cb"][c], end)
    return out


def regrid_full_grid_sdpa(invs, gx, gy):
    """Model the full-grid global SDPA running on a gx x gy sub-grid with the same per-core stack."""
    ceiling = max((max(v["cb"].values()) for v in invs if v["op"] == "SdpaDecode" and v["cb"]), default=0)
    moved = 0
    for v in invs:
        if v["op"] == "SdpaDecode" and len(v["cb"]) == len(GRID) and max(v["cb"].values()) == ceiling:
            v["cb"] = {c: e for c, e in v["cb"].items() if c[0] < gx and c[1] < gy}
            moved += 1
    return moved


def evaluate(invs, steps):
    rows = []
    for k in steps:
        head_b = k * SHARD_STEP_B
        start = TOP - head_b
        cols = k * SHARD_STEP_COLS
        types = collections.Counter()
        moved = no_room = 0
        worst = None
        for v in invs:
            over = {c: e - start for c, e in v["cb"].items() if e > start}
            if not over:
                continue
            types[v["op"]] += 1
            need = sum(over.values())
            room = sum(start - max(BASE, v["cb"].get(c, BASE)) for c in GRID if c not in over)
            moved += need
            no_room += room < need
            if worst is None or need / max(room, 1) > worst["ratio"]:
                worst = {"op": v["op"], "need_B": need, "room_B": room, "ratio": need / max(room, 1)}
        extra = cols - BASELINE_COLS
        saved_b = DRAFTER_HEADS_PER_ITER * extra * 1024 * 2
        gain_us = DRAFTER_HEADS_PER_ITER * (extra * 1024 * 2 / DRAM_GBPS * 1e6 - extra * SLICE_US_PER_COL)
        rows.append(
            {
                "pin_cols": cols,
                "head_B_per_bank": head_b,
                "head_start": start,
                "colliding_ops_per_iter": sum(types.values()),
                "colliding_by_op": dict(types.most_common()),
                "ops_without_parking_room": no_room,
                "moved_B_per_iter_out_and_back": 2 * moved,
                "worst_op": worst,
                "dram_saved_B_per_iter": saved_b,
                "gain_if_moves_free_us": gain_us,
                "break_even_move_rate_GBps": (2 * moved / (gain_us * 1e-6) / 1e9) if gain_us > 0 and moved else None,
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--target-sdpa-grid", help="model the full-grid global SDPA on a sub-grid, e.g. 8x8")
    p.add_argument("--max-steps", type=int, default=21)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    invs = invocations(json.loads((a.capture_dir / "iteration.graph.json").read_text()))
    note = "as captured"
    if a.target_sdpa_grid:
        gx, gy = map(int, a.target_sdpa_grid.lower().split("x"))
        note = f"full-grid global SDPA re-gridded to {gx}x{gy} ({regrid_full_grid_sdpa(invs, gx, gy)} invocations)"
    rows = evaluate(invs, range(BASELINE_COLS // SHARD_STEP_COLS, a.max_steps + 1))
    print(f"{a.capture_dir.name}: {len(invs)} op invocations; {note}")
    print(
        f"{'pin cols':>8} | {'colliding':>9} {'no room':>7} | {'moved/iter':>11} | {'DRAM saved':>10} {'gain*':>7} | "
        f"{'break-even':>11} | colliding ops"
    )
    for r in rows:
        be = f"{r['break_even_move_rate_GBps']:,.0f} GB/s" if r["break_even_move_rate_GBps"] else "-"
        top = ", ".join(f"{k}:{v}" for k, v in list(r["colliding_by_op"].items())[:4])
        print(
            f"{r['pin_cols']:>8,} | {r['colliding_ops_per_iter']:>9} {r['ops_without_parking_room']:>7} | "
            f"{r['moved_B_per_iter_out_and_back'] / 1e6:>8.1f} MB | {r['dram_saved_B_per_iter'] / 1e6:>7.1f} MB "
            f"{r['gain_if_moves_free_us']:>5.0f}us | {be:>11} | {top}"
        )
    print("* gain if moves were free: 3 drafter heads x (extra DRAM bytes / 396 GB/s - extra L1 slice time)")
    if a.json:
        a.json.write_text(json.dumps({"capture": str(a.capture_dir), "scenario": note, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
