#!/usr/bin/env python3
"""Fixed cost of an L1-fed fat-K matmul: kernel time vs K on the same 32 cores (gemma4-12b record §9.2, item 6a).

Input: the cells of item6/run_l1_fixed_cost.sh. In each cell one fat-K weight is pinned L1 WIDTH_SHARDED on the
tuner's 32 cores (per_core_N 1): o_proj layer 0 (K 4096, Kt 128), pre_projection (K 7680, Kt 240) or down_proj
layer 0 (K 8192, Kt 256), on the 1D multicast or the gather_in0 ring. With no DRAM read, kernel time = a + b * Kt
per delivery: the intercept a is the fixed per-call cost (launch, pipeline fill / drain, output write), the slope b
the per-K-tile cost of the in0 delivery and compute. Reshards around ring calls are separate ops and are listed, not
fitted.

    python item6/analyze_l1_fixed_cost.py <campaign_dir> -> <campaign_dir>/l1_fixed_cost.{json,md}
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_resource_profile import label_session, load  # noqa: E402

LAYER_TYPES = ["sliding", "sliding", "sliding", "global"]
# pin suffix -> (op label, layer, Kt)
PINS = {
    "pino_proj@0": ("o_proj (sliding)", 0, 128),
    "pinpre_projection": ("pre_projection", None, 240),
    "pindown_proj@0": ("down_proj (sliding)", 0, 256),
}


def fit(points):
    """Least-squares a + b * x over [(x, y)]; with 2 points, the line through them."""
    xs, ys = zip(*points)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    times = defaultdict(list)  # (delivery, Kt) -> per-call kernel us
    reshard = defaultdict(list)
    for cell in sorted(d for d in args.root.iterdir() if (d / "ops_perf.csv").exists()):
        pin = cell.name.split("+")[-1]
        ring = pin.endswith("r")
        key = pin[:-1] if ring else pin
        if key not in PINS:
            continue
        label, layer, kt = PINS[key]
        for ops in load(cell / "ops_perf.csv").values():
            for o, (_, name, lay) in zip(ops, label_session(ops, LAYER_TYPES)):
                if name == label and (layer is None or lay == layer) and "L1" in o["row"].get("INPUT_1_MEMORY", ""):
                    times[("ring" if ring else "mcast", kt)].append(o["dur"])
                elif (
                    ring
                    and name.startswith(("to sharded: ", "to interleaved: "))
                    and label.split(" (")[0] in name
                    and lay == layer
                ):
                    reshard[(kt, name.split(":")[0])].append(o["dur"])
    out = {"points": {}, "fit": {}, "reshards": {}}
    lines = ["L1-fed fat-K matmul, kernel time vs Kt on 32 cores (traced time mode, mean per call)", ""]
    lines += ["| delivery | Kt | calls | kernel us |", "|---|--:|--:|--:|"]
    for (dlv, kt), v in sorted(times.items()):
        out["points"][f"{dlv}_{kt}"] = statistics.mean(v)
        lines.append(f"| {dlv} | {kt} | {len(v)} | {statistics.mean(v):.2f} |")
    lines += [
        "",
        "| delivery | intercept a, us (fixed per call) | slope b, us per K tile | points |",
        "|---|--:|--:|--:|",
    ]
    for dlv in ("mcast", "ring"):
        pts = [(kt, statistics.mean(v)) for (d, kt), v in times.items() if d == dlv]
        if len(pts) >= 2:
            a, b = fit(pts)
            out["fit"][dlv] = {"a_us": a, "b_us_per_kt": b, "points": len(pts)}
            lines.append(f"| {dlv} | {a:.2f} | {b * 1e3:.1f} ns | {len(pts)} |")
    for (kt, kind), v in sorted(reshard.items()):
        out["reshards"][f"{kt}_{kind}"] = statistics.mean(v)
    if reshard:
        lines += [
            "",
            "Ring reshards (separate ops, not fitted): "
            + ", ".join(f"Kt {kt} {k} {statistics.mean(v):.2f} us" for (kt, k), v in sorted(reshard.items())),
        ]
    (args.root / "l1_fixed_cost.json").write_text(json.dumps(out, indent=1) + "\n")
    (args.root / "l1_fixed_cost.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
