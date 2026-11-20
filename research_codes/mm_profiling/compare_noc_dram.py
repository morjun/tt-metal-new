"""Baseline vs pinned arm: per-op DRAM read / write (NoC traces) and device time (time cells), by role (record §8).

Inputs, all from run_opprof.py cells under <root>:
  noc_dram.json   analyze_noc_dram.py --json over the two noc cells (bytes per iteration = 3 drafter steps)
  <index>_time_<arm>/ops_perf.csv   device kernel durations of 5 replays (averaged here)
Roles follow the drafter's weight shapes (hidden 1024; sliding heads 256, global heads 512):
the per-label rows of analyze_noc_dram.py are summed into the roles below.

    python compare_noc_dram.py <root> <noc cell A> <noc cell B> <time cell A> <time cell B>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze_noc_dram  # noqa: E402

ROLES = [  # (role, predicate on the analyze_noc_dram label)
    ("lm_head, all 262,144 columns from DRAM", lambda n: n == "Matmul 1024x262144"),
    ("lm_head, L1-pinned 59,840 columns", lambda n: n == "Matmul 1024x59840"),
    ("lm_head, DRAM tail 202,304 columns", lambda n: n == "Matmul 1024x202304"),
    ("gate_up (fused), 12", lambda n: n == "Matmul 1024x16384"),
    ("down 12 + global o_proj 3", lambda n: n == "Matmul 8192x1024"),
    ("sliding qkv, 9", lambda n: n == "Matmul 1024x8192"),
    ("sliding o_proj, 9", lambda n: n == "Matmul 4096x1024"),
    ("global qkv, 3", lambda n: n == "Matmul 1024x9216"),
    ("pre_projection, 3", lambda n: n == "Matmul 7680x1024"),
    ("post_projection, 3", lambda n: n == "Matmul 1024x3840"),
    ("SDPA sliding, 9", lambda n: n == "SDPA (sliding)"),
    ("SDPA global, 3", lambda n: n == "SDPA (global)"),
    (
        "logits untilize + row join",
        lambda n: n.startswith("UntilizeWithUnpadding") or n.startswith("GenericOp after Matmul 1024x202304"),
    ),
    (
        "DRISC reload trigger + wait",
        lambda n: n in ("GenericOp after Matmul 1024x9216 #0", "GenericOp after Matmul 8192x1024 #0"),
    ),
    ("ArgMax", lambda n: n.startswith("ArgMax")),
    ("other non-matmul ops", lambda n: True),
]


def role(name):
    return next(r for r, f in ROLES if f(name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("noc", nargs=2)
    p.add_argument("time", nargs=2)
    p.add_argument("--replays", type=int, default=5)
    args = p.parse_args()
    noc = json.loads((args.root / "noc_dram.json").read_text())
    arms = []
    for nc, tc in zip(args.noc, args.time):
        agg = {r: {"read": 0.0, "write": 0.0, "us": 0.0, "n": 0.0} for r, _ in ROLES}
        for name, a in noc[nc]["agg"].items():
            g = agg[role(name)]
            g["read"] += a.get("read", 0)
            g["write"] += a.get("write", 0)
        for name, a in analyze_noc_dram.cell(args.root / tc)["agg"].items():
            g = agg[role(name)]
            g["us"] += a["us"] / args.replays
            g["n"] += a["count"] / args.replays
        arms.append(agg)
    out = [
        "| role | calls / iter (A / B) | A read MB | A write MB | A µs | B read MB | B write MB | B µs | Δ read + write MB | Δ µs |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    tot = [{"read": 0.0, "write": 0.0, "us": 0.0} for _ in arms]
    for r, _ in ROLES:
        a, b = (x[r] for x in arms)
        if not (a["n"] or b["n"]):
            continue
        for t, x in zip(tot, (a, b)):
            for k in t:
                t[k] += x[k]
        cell = lambda x: f"{x['read'] / 1e6:,.2f} | {x['write'] / 1e6:,.2f} | {x['us']:,.1f}" if x["n"] else "— | — | —"
        d_mb = (b["read"] + b["write"] - a["read"] - a["write"]) / 1e6
        out.append(
            f"| {r} | {a['n']:.0f} / {b['n']:.0f} | {cell(a)} | {cell(b)} | {d_mb:+,.2f} | {b['us'] - a['us']:+,.1f} |"
        )
    a, b = tot
    out.append(
        f"| **total** | | {a['read'] / 1e6:,.2f} | {a['write'] / 1e6:,.2f} | {a['us']:,.1f} | {b['read'] / 1e6:,.2f} | "
        f"{b['write'] / 1e6:,.2f} | {b['us']:,.1f} | {(b['read'] + b['write'] - a['read'] - a['write']) / 1e6:+,.2f} | {b['us'] - a['us']:+,.1f} |"
    )
    (args.root / "compare_noc_dram.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
