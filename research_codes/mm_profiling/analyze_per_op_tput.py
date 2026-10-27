#!/usr/bin/env python3
"""Per-op effective throughput of the drafter's weight matmuls per arm (gemma4-12b record §9.2e, per_op_ep0).

From time cells of run_traced_opprof_arms.py: per arm, each op label's µs per 3-step iteration (mean over every replay
of every round; analyze_t3_vc.replays) and the weight bytes it reads per iteration (shape x calls, bf16), giving
effective GB/s = bytes / time. Effective rates include each op's fixed costs; they are not DRAM bandwidths.

    python research_codes/mm_profiling/analyze_per_op_tput.py generated/lmhead_l1_profile/per_op_ep0
        -> <root>/per_op_tput.{json,md}
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_t3_vc import replays  # noqa: E402

B = 2  # bf16
MB = {  # weight bytes read per 3-step iteration (shape x calls; record §9.2d shapes)
    "pre_projection": 7680 * 1024 * B * 3,
    "o_proj (sliding)": 4096 * 1024 * B * 9,
    "o_proj (global)": 8192 * 1024 * B * 3,
    "down_proj (sliding)": 8192 * 1024 * B * 9,
    "down_proj (global)": 8192 * 1024 * B * 3,
    "qkv_proj (sliding)": 1024 * 8192 * B * 9,
    "qkv_proj (global)": 1024 * 9216 * B * 3,
    "gate_proj (sliding)": 1024 * 8192 * B * 9,
    "gate_proj (global)": 1024 * 8192 * B * 3,
    "up_proj (sliding)": 1024 * 8192 * B * 9,
    "up_proj (global)": 1024 * 8192 * B * 3,
    "post_projection": 1024 * 3840 * B * 3,
    "lm_head (whole, DRAM)": 1024 * 262144 * B * 3,
    "lm_head DRAM tail": 1024 * 202304 * B * 3,
}
FAMILY = {
    "fat-K": ("pre_projection", "o_proj (sliding)", "o_proj (global)", "down_proj (sliding)", "down_proj (global)"),
    "thin-K": tuple(k for k in MB if k.split(" ")[0] in ("qkv_proj", "gate_proj", "up_proj", "post_projection")),
    "lm_head (DRAM part)": ("lm_head (whole, DRAM)", "lm_head DRAM tail"),
}


def main(root):
    arms = defaultdict(list)
    for d in sorted(root.iterdir()):
        if (d / "ops_perf.csv").exists():
            arms[d.name.split("_", 2)[2]].extend(replays(d))
    res = {}
    for arm, rs in arms.items():
        us = {k: statistics.mean(r.get(k, 0.0) for r in rs) for k in set().union(*rs)}
        row = {"replays": len(rs), "span": us["span"]}
        for k in MB:
            if us.get(k):
                row[k] = (us[k], MB[k] / (us[k] * 1e3))
        for fam, keys in FAMILY.items():
            t = sum(us.get(k, 0.0) for k in keys)
            b = sum(MB[k] for k in keys if us.get(k))
            if t:
                row[fam] = (t, b / (t * 1e3))
        if us.get("lm_head L1 slice"):
            row["lm_head L1 slice"] = (us["lm_head L1 slice"], None)
        res[arm] = row
    order = sorted(res, key=lambda a: res[a]["span"], reverse=True)
    cols = list(MB) + ["lm_head L1 slice"] + list(FAMILY)
    lines = ["| op | " + " | ".join(f"`{a}`" for a in order) + " |", "|---|" + "--:|" * len(order)]
    lines.append("| span µs/iter | " + " | ".join(f"{res[a]['span']:.0f}" for a in order) + " |")
    for k in cols:
        cells = []
        for a in order:
            v = res[a].get(k)
            cells.append("—" if v is None else (f"{v[0]:.0f} µs" if v[1] is None else f"{v[1]:.0f} ({v[0]:.0f} µs)"))
        if any(c != "—" for c in cells):
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
    text = "Effective GB/s per op (µs per 3-step iteration)\n\n" + "\n".join(lines)
    (root / "per_op_tput.md").write_text(text + "\n")
    (root / "per_op_tput.json").write_text(json.dumps(res, indent=1))
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
