#!/usr/bin/env python3
"""Per-op attribution of a head-split layout against F0 (gemma4-12b record §9.4, H1 vs F0).

Reads the time-mode cells of run_traced_opprof_arms.py (ops_perf.csv, trace replays only) and splits every drafter
step into: up_proj L0, the other backbone matmuls, the head's L1 slice A (the L1 head weight on the most cores), slice B (other
L1 head weights), the DRAM tail, post_projection, untilize, concat, argmax, the reload's generic ops, every other op,
and the op-to-op gaps. Matmuls are placed by order (pre_projection, then 5 per layer, then the head parts, then
post_projection last); head parts by their weight's buffer type. Values are µs per 3-step draft, mean over replays.

    python research_codes/mm_profiling/analyze_p4_attr.py generated/lmhead_l1_profile/opprof_p4_attr
"""

import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_resource_profile import load  # noqa: E402

ROLES = ["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
KEYS = [
    "up_proj L0",
    "other backbone matmuls",
    "head slice A",
    "head slice B",
    "head DRAM tail",
    "post_projection",
    "untilize",
    "concat",
    "argmax",
    "reload generic ops",
    "other ops",
    "op-to-op gaps",
    "span",
]


def width(o):
    x = o["row"].get("INPUT_1_X_PAD[LOGICAL]", "")
    return int(x.split("[")[1].rstrip("]")) if "[" in x else None


def step_parts(step):
    acc = defaultdict(float)
    mms = [o for o in step if o["op"] == "Matmul"]
    head = mms[21:-1]
    in_l1 = [o for o in head if "L1" in o["row"].get("INPUT_1_MEMORY", "")]
    main = max(in_l1, key=lambda o: o["cores"]) if in_l1 else None  # slice A spans all 110 cores
    for k, o in enumerate(mms):
        if 1 <= k <= 20 and ROLES[(k - 1) % 5] == "up_proj" and (k - 1) // 5 == 0:
            acc["up_proj L0"] += o["dur"]
        elif k <= 20:
            acc["other backbone matmuls"] += o["dur"]
        elif o is mms[-1]:
            acc["post_projection"] += o["dur"]
        elif o is main:
            acc["head slice A"] += o["dur"]
        elif o in in_l1:
            acc["head slice B"] += o["dur"]
        else:
            acc["head DRAM tail"] += o["dur"]
    for o in step:
        if o["op"] == "Matmul":
            continue
        c = o["op"]
        key = (
            "untilize"
            if "Untilize" in c
            else "concat"
            if "Concat" in c
            else "argmax"
            if c == "ArgMax"
            else "reload generic ops"
            if "Generic" in c
            else "other ops"
        )
        acc[key] += o["dur"]
    acc["op-to-op gaps"] += sum(o["gap"] for o in step[1:])
    return acc


def cell(path):
    per = []
    for ops in load(path / "ops_perf.csv").values():
        acc, step = defaultdict(float), []
        for o in ops:
            step.append(o)
            if o["op"] == "ArgMax":
                for k, v in step_parts(step).items():
                    acc[k] += v
                step = []
        acc["span"] = ops[0]["dur"] + sum(o["dur"] + o["gap"] for o in ops[1:])
        per.append(acc)
    return {k: statistics.mean(r.get(k, 0.0) for r in per) for k in KEYS}, len(per)


def main(root):
    cells = sorted(d for d in Path(root).iterdir() if (d / "ops_perf.csv").exists())
    res = []
    for d in cells:
        v, n = cell(d)
        res.append((d.name.split("_", 2)[2], v, n))
    print("| part (µs per draft) | " + " | ".join(f"{chr(97 + i)}" for i in range(len(res))) + " |")
    print("|---|" + "--:|" * len(res))
    for k in KEYS:
        print(f"| {k} | " + " | ".join(f"{v[k]:.1f}" for _, v, _ in res) + " |")
    for i, (arm, _, n) in enumerate(res):
        print(f"{chr(97 + i)}: {arm} ({n} replays)")


if __name__ == "__main__":
    main(sys.argv[1])
