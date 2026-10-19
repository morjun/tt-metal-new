#!/usr/bin/env python3
"""Space-time L1 budget map of the standalone drafter's K=3 fused trace (gemma4-12b record §9.1).

Inputs: one address capture (run_spacetime_capture.py) and the `time` cells of the same arm from
run_traced_opprof_arms.py. Every device op of the capture is matched, in order, to the traced replay's op, so
each op gets its program-local CB footprint (per core and global), the L1 buffers live while it runs, its
label (analyze_resource_profile.label_step) and its mean traced duration + op-to-op gap.

Derived:
  clash(a)   the ops whose program-local CBs end above address a. Under LOCKSTEP the host CB check compares
             each program's CB end with the lowest L1 buffer address on ANY bank (program.cpp:1674, :1767),
             so an item at a must be bracketed (freed + reloaded) around every op in clash(a).
  per core   the max CB end per core over the step, with and without the bracketed ops (the HYBRID view:
             there the check is per bank, program.cpp:1750-1764).
  windows    the drafter's weight consumers (matmuls) in time order, the time between consecutive ones, and
             the bytes a paced DRISC push moves in that time.

    python analyze_spacetime_map.py <capture_dir> <profile_root> --arm B+logitsl1
        -> <capture_dir>/analysis/spacetime.{json,md}
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_resource_profile import load, label_session
from lm_head_l1_cb_ceiling import invocations

L1_TOP = 1_572_864  # MEM_L1_SIZE, record §5.0
GRID = [(x, y) for y in range(10) for x in range(11)]
THRESHOLDS = [458_752, 512_896, 589_824, 653_568, 720_896, 801_024, 851_968]


def op_code(name):
    return name.split("::")[-1].replace("DeviceOperation", "")


def live_buffers(graph, before, invs):
    """L1 buffers live at the start of each invocation: [(address, bytes per bank, num_cores, layout)]."""
    base = before["base"]
    live = {}
    for b in before["blocks"]:
        if b.get("allocated") == "yes":
            a = base + int(b["address"])
            info = before["buffers"].get(str(a), {})
            live[a] = (a, int(b["size"]), None, info.get("layout", ""))
    starts = {inv["start"]: k for k, inv in enumerate(invs)}
    out = [None] * len(invs)
    for i, node in enumerate(graph):
        if i in starts:
            out[starts[i]] = sorted(live.values(), key=lambda b: b[0])
        p = node.get("params") or {}
        if node["node_type"] == "buffer_allocate" and p.get("type") == "L1":
            a = int(p["address"])
            live[a] = (a, int(p["max_size_per_bank"]), int(p.get("num_cores", 0)), p.get("layout", ""))
        elif node["node_type"] == "buffer_deallocate" and p.get("type") == "L1":
            live.pop(int(p["address"]), None)
    return out


def profile_ops(root, arm):
    """Mean traced duration / gap / DRAM bytes and labels per op position, over every replay of the arm's cells."""
    sessions = []
    for d in sorted(root.iterdir()):
        if d.name.endswith(f"_time_{arm}") and (d / "ops_perf.csv").exists():
            for ops in load(d / "ops_perf.csv").values():
                labels = label_session(ops, ["sliding", "sliding", "sliding", "global"])
                sessions.append([dict(o, label=lab) for o, lab in zip(ops, labels)])
    assert sessions, f"no time cells for {arm} under {root}"
    seq = {tuple(o["op"] for o in s) for s in sessions}
    assert len(seq) == 1, f"{len(seq)} different op sequences across replays"
    out = []
    for i, o in enumerate(sessions[0]):
        out.append(
            {
                "op": o["op"],
                "kind": o["label"][0],
                "name": o["label"][1],
                "layer": o["label"][2],
                "cores": o["cores"],
                "us": statistics.mean(s[i]["dur"] for s in sessions),
                "gap_us": statistics.mean(s[i]["gap"] for s in sessions) if i else 0.0,
                "dram_bytes": statistics.mean(s[i]["bytes"] for s in sessions),
            }
        )
    return out, len(sessions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture_dir", type=Path)
    p.add_argument("profile_root", type=Path, help="run_traced_opprof_arms.py output holding <i>_time_<arm> cells")
    p.add_argument("--arm", required=True, help="cell arm name in profile_root (e.g. B+logitsl1)")
    p.add_argument("--pinned-addr", type=int, default=0, help="bottom of the arm's pinned region (S); 0 = no pin")
    p.add_argument("--pace-gbps", type=float, default=60.0, help="hidden DRISC push rate for the window table")
    p.add_argument(
        "--bracketed",
        default="",
        help="comma-separated op labels the arm brackets (freed + reloaded around), excluded from the "
        "no-bracket ceilings; e.g. 'SDPA (global),rotary embedding (global),concat split logits'",
    )
    args = p.parse_args()
    root = args.capture_dir
    graph = json.loads((root / "iteration.graph.json").read_text())
    before = json.loads((root / "before_iteration.l1.json").read_text())
    base = before["base"]
    invs, _ = invocations(graph)
    prof, n_sessions = profile_ops(args.profile_root, args.arm)

    cap_seq = [op_code(i["op"]) for i in invs]
    prof_seq = [o["op"] for o in prof]
    if cap_seq != prof_seq:
        for k, (a, b) in enumerate(zip(cap_seq, prof_seq)):
            if a != b:
                raise SystemExit(f"op {k}: capture {a} vs profile {b} ({len(cap_seq)} vs {len(prof_seq)} ops)")
        raise SystemExit(f"op count: capture {len(cap_seq)} vs profile {len(prof_seq)}")
    lives = live_buffers(graph, before, invs)
    bracketed = {s.strip() for s in args.bracketed.split(",") if s.strip()}

    ops, t = [], 0.0
    for k, (inv, o, live) in enumerate(zip(invs, prof, lives)):
        local = [cb for cb in inv["cbs"] if not cb["global"]]
        per_core = defaultdict(int)
        for cb in local:
            for c in cb["cores"]:
                per_core[tuple(c)] = max(per_core[tuple(c)], cb["address"] + cb["size"])
        t += o["gap_us"]
        ops.append(
            {
                "index": k,
                "step": None,
                "op": o["op"],
                "name": o["name"],
                "layer": o["layer"],
                "cores": o["cores"],
                "start_us": t,
                "us": o["us"],
                "dram_bytes": o["dram_bytes"],
                "cb_end": max(per_core.values(), default=base),
                "cb_cores": len(per_core),
                "per_core_cb_end": {f"{x},{y}": e for (x, y), e in sorted(per_core.items())},
                "live_l1": [{"address": a, "bytes_per_bank": s, "num_cores": n, "layout": l} for a, s, n, l in live],
                "bracketed": o["name"] in bracketed,
            }
        )
        t += o["us"]
    step = 0
    for o in ops:
        o["step"] = step
        step += o["op"] == "ArgMax"
    span_us = t

    # clash(a): ops whose CBs reach above a
    clash = {}
    for a in THRESHOLDS:
        hit = [o for o in ops if o["cb_end"] > a]
        clash[a] = {
            "ops": len(hit),
            "per_step": sorted({o["name"] for o in hit}),
            "us": sum(o["us"] for o in hit),
        }
    ceiling_all = max(o["cb_end"] for o in ops)
    ceiling_nb = max(o["cb_end"] for o in ops if not o["bracketed"])

    # per-core max CB end over the step (HYBRID view)
    core_max, core_max_nb = defaultdict(lambda: base), defaultdict(lambda: base)
    for o in ops:
        for c, e in o["per_core_cb_end"].items():
            core_max[c] = max(core_max[c], e)
            if not o["bracketed"]:
                core_max_nb[c] = max(core_max_nb[c], e)

    # transients: L1 buffers below the pinned region (allocated top-down under it), per op. An op's demand is its
    # CB end plus the transient bytes live during it; a pin of P bytes per bank fits (LOCKSTEP, pins allocated
    # first at the top) iff demand + P <= top for every op that is not bracketed.
    pin = args.pinned_addr or L1_TOP
    for o in ops:
        below = [b["address"] for b in o["live_l1"] if b["address"] < pin]
        o["transient_bytes"] = pin - min(below) if below else 0
        o["demand"] = o["cb_end"] + o["transient_bytes"]
    bracket_sets = {"none": set()}
    acc = set()
    for name in [s.strip() for s in args.bracketed.split(",") if s.strip()]:
        acc = acc | {name}
        bracket_sets[" + ".join(sorted(acc))] = set(acc)
    budgets = {}
    for key, names in bracket_sets.items():
        worst = max((o for o in ops if o["name"] not in names), key=lambda o: o["demand"])
        budgets[key] = {
            "max_demand": worst["demand"],
            "by": worst["name"],
            "pin_bytes_per_bank": L1_TOP - worst["demand"],
        }

    # weight consumers (matmuls) and the windows between them, step 0
    consumers = [o for o in ops if o["op"] == "Matmul"]
    windows = []
    for prev, cur in zip(consumers, consumers[1:]):
        gap = cur["start_us"] - (prev["start_us"] + prev["us"])
        windows.append(
            {
                "after": f"s{prev['step']} {prev['name']}"
                + (f" L{prev['layer']}" if prev["layer"] is not None else ""),
                "before": f"s{cur['step']} {cur['name']}" + (f" L{cur['layer']}" if cur["layer"] is not None else ""),
                "between_us": gap,
                "consumer_us": cur["us"],
                "push_mb_between": gap * args.pace_gbps / 1e3,
                "push_mb_incl_prev_matmul": (gap + prev["us"]) * args.pace_gbps / 1e3,
            }
        )

    result = {
        "capture": str(root),
        "profile": f"{args.profile_root} ({args.arm}, {n_sessions} replays)",
        "base": base,
        "l1_top": L1_TOP,
        "span_us": span_us,
        "ops": ops,
        "clash": {str(k): v for k, v in clash.items()},
        "ceiling_all": ceiling_all,
        "ceiling_no_bracketed": ceiling_nb,
        "pinned_addr": args.pinned_addr,
        "max_transient_bytes": max(o["transient_bytes"] for o in ops),
        "pin_budget": budgets,
        "core_max_cb_end": dict(core_max),
        "core_max_cb_end_no_bracketed": dict(core_max_nb),
        "windows": windows,
    }
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    (out / "spacetime.json").write_text(json.dumps(result, indent=1) + "\n")

    md = [f"# Space-time L1 map: {root.name}", ""]
    md.append(f"- ops {len(ops)} (3 steps), traced span {span_us:.1f} us, {n_sessions} replays")
    md.append(f"- L1 base {base:,}, top {L1_TOP:,}")
    md.append(f"- max CB end: all ops {ceiling_all:,}; without bracketed ops {ceiling_nb:,}")
    md.append(f"- pinned region from {pin:,}; max transient L1 below it {result['max_transient_bytes']:,} B/bank")
    md += ["", "## LOCKSTEP pin budget (bytes per bank) by bracketed op set", ""]
    md.append("| bracketed | max demand (CB end + transients) | set by | pin budget B/bank | x110 MB |")
    md.append("|---|--:|---|--:|--:|")
    for key, b in budgets.items():
        md.append(
            f"| {key} | {b['max_demand']:,} | {b['by']} | {b['pin_bytes_per_bank']:,} | "
            f"{b['pin_bytes_per_bank'] * 110 / 1e6:.1f} |"
        )
    md += ["", "## Ops whose CBs reach above address a (they clash with an item placed at a)", ""]
    md.append("| a | ops (3 steps) | their kernel us | op labels |")
    md.append("|--:|--:|--:|---|")
    for a in THRESHOLDS:
        c = clash[a]
        md.append(f"| {a:,} | {c['ops']} | {c['us']:.1f} | {'; '.join(c['per_step'])} |")
    md += ["", "## Highest CB end per op label (max over steps)", ""]
    md.append("| op label | op | cores | max CB end | kernel us (mean) |")
    md.append("|---|---|--:|--:|--:|")
    by = defaultdict(list)
    for o in ops:
        by[o["name"]].append(o)
    for name, v in sorted(by.items(), key=lambda kv: -max(o["cb_end"] for o in kv[1])):
        md.append(
            f"| {name} | {v[0]['op']} | {max(o['cb_cores'] for o in v)} | {max(o['cb_end'] for o in v):,} | "
            f"{statistics.mean(o['us'] for o in v):.1f} |"
        )
    for title, cm in (("all ops", core_max), ("without bracketed ops", core_max_nb)):
        md += ["", f"## Per-core max CB end over the step, {title} (KiB; free above = top - this)", ""]
        md.append("| y \\ x | " + " | ".join(str(x) for x in range(11)) + " |")
        md.append("|--:|" + "--:|" * 11)
        for y in range(10):
            md.append(f"| {y} | " + " | ".join(f"{cm.get(f'{x},{y}', base) // 1024}" for x in range(11)) + " |")
    md += ["", f"## Windows between weight consumers (step 0; push at {args.pace_gbps:g} GB/s)", ""]
    md.append("| after | before | between us | consumer us | push MB between | push MB incl. prev matmul |")
    md.append("|---|---|--:|--:|--:|--:|")
    for w in windows:
        if w["after"].startswith("s0"):
            md.append(
                f"| {w['after']} | {w['before']} | {w['between_us']:.1f} | {w['consumer_us']:.1f} | "
                f"{w['push_mb_between']:.2f} | {w['push_mb_incl_prev_matmul']:.2f} |"
            )
    (out / "spacetime.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
