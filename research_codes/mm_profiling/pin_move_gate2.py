#!/usr/bin/env python3
"""Gate 2 verdict: measured move cost vs modelled DRAM saving, per pin size.

Inputs: generated/lmhead_l1_profile/pin_move_cost/move_cost_*.json (test_pin_move_cost.py, one per process) and the
8x8-scenario rows of pin_move_feasibility.py for a capture. Per iteration the moves are:
8 target global SDPAs (64 -> 46 cores) + 3 drafter global SDPAs (32 -> 78 cores), each out and
back, using the faster mover (1 or 2 RISCs) per case and size; from 52,800 columns also 3
single-core concats, charged at the measured 64 B (fixed) cost of the target case as a floor.
GO if some pin size is faster by more than 50 us/iteration (dt < -50; negative = faster) (about the §5.5.13 paired noise floor).

    python pin_move_gate2.py CAPTURE_DIR
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pin_move_feasibility import evaluate, invocations, regrid_full_grid_sdpa  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/pin_move_cost"
GO_THRESHOLD_US = 50.0


def main():
    cap = Path(sys.argv[1])
    parts = [json.loads(f.read_text()) for f in sorted(ROOT.glob("move_cost_*.json"))]
    mc = {"pin_overlap_B": parts[0]["pin_overlap_B"], "rows": [r for p in parts for r in p["rows"]]}
    overlap = {int(k): v for k, v in mc["pin_overlap_B"].items()}
    best = {}
    for r in mc["rows"]:
        key = (r["case"], r["bytes_per_src_core"])
        t = r["out_us"] + r["back_us"]
        if key not in best or t < best[key][0]:
            best[key] = (t, r["riscs"])
    invs = invocations(json.loads((cap / "iteration.graph.json").read_text()))
    regrid_full_grid_sdpa(invs, 8, 8)
    model = {r["pin_cols"]: r for r in evaluate(invs, range(11, 17))}
    fixed = best[("target_8x8", 64)][0]
    rows = []
    print(
        f"{'pin cols':>8} | {'B/core':>8} | {'target move':>12} {'drafter move':>13} | {'cost/iter':>10} | "
        f"{'saved*':>7} | {'dt/iter':>8}"
    )
    for cols, b in sorted(overlap.items()):
        t_t, r_t = best[("target_8x8", b)]
        t_d, r_d = best[("drafter_8x4", b)]
        concat = 3 * fixed if model[cols]["colliding_by_op"].get("Concat") else 0.0
        cost = 8 * t_t + 3 * t_d + concat
        gain = model[cols]["gain_if_moves_free_us"]
        rows.append(
            {
                "pin_cols": cols,
                "overlap_B_per_core": b,
                "target_out_back_us": t_t,
                "target_riscs": r_t,
                "drafter_out_back_us": t_d,
                "drafter_riscs": r_d,
                "concat_floor_us": concat,
                "move_cost_us_per_iter": cost,
                "saved_if_free_us": gain,
                "delta_us_per_iter": cost - gain,
            }
        )
        print(
            f"{cols:>8,} | {b:>8,} | {t_t:>9.1f} us ({r_t}R) {t_d:>8.1f} us ({r_d}R) | {cost:>7.1f} us | "
            f"{gain:>5.0f}us | {cost - gain:>+6.1f}us"
        )
    best = min(rows, key=lambda r: r["delta_us_per_iter"])
    verdict = "GO" if best["delta_us_per_iter"] < -GO_THRESHOLD_US else "NO-GO"
    print(
        f"* DRAM time saved if moves were free (model); dt = cost - saved, negative = faster. Best dt "
        f"{best['delta_us_per_iter']:+.1f} us/iter at {best['pin_cols']:,} cols -> {verdict} "
        f"(GO needs dt < -{GO_THRESHOLD_US:.0f} us/iter)"
    )
    (ROOT / "gate2.json").write_text(
        json.dumps({"capture": str(cap), "rows": rows, "verdict": verdict, "threshold_us": GO_THRESHOLD_US}, indent=1)
        + "\n"
    )


if __name__ == "__main__":
    main()
