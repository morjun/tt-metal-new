#!/usr/bin/env python3
"""M1 and M2 of gemma4-12b record §9.2b ("Reader windows and residuals", staged measurement plan).

Reads the traced time-mode cells of run_1102_m1m2.sh (<root>/<index>_time_<arm>/: ops_perf.csv, and niu_samples.csv
for +niusample arms); index // 6 is the round. Every value is µs per 3-step draft (one replay), first per replay, then
mean / SD / SE over the replays of an arm and mode.

M1 (no head in L1): base vs up L0 pin, per part (analyze_p4_attr.step_parts) and per matmul instance (labelled by
order: pre_projection, then qkv / o / gate / up / down per layer).

M2 (F0 vs H1, NIU sampled and unsampled), per replay:
  R_tail, R_up      reader windows (latest final sample - earliest initial sample across cores) of the DRAM tail and,
                    in F0, of up L0; NIU bytes per call (phantom reads included)
  T_up, T_A, T_B, T_tail   kernel times
  Q_F0 = T_up + T_A + T_tail - R_tail - R_up,  Q_H1 = T_up + T_A + T_tail - R_tail
  G_read = mean R_tail,H1 - mean (R_tail + R_up),F0;  dT_shared = mean (T_up + T_A + T_tail) difference;
  P_final = mean Q_H1 - mean Q_F0 = dT_shared - G_read
SE of a difference of arm means = sqrt(SD_H1^2 / n_H1 + SD_F0^2 / n_F0): F0 and H1 are different processes (not
paired); the terms within a replay are summed before the variance, so their covariance is included.

    python research_codes/mm_profiling/analyze_m12.py generated/lmhead_l1_profile/m12_1102 \
        [--standalone generated/lmhead_l1_profile/standalone_m12_1102/results.json] -> stdout + <root>/m12.json
"""

import argparse
import bisect
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "item6"))
from analyze_niu_samples import MHZ, WORD_BYTES, read_samples  # noqa: E402
from analyze_p4_attr import KEYS, step_parts  # noqa: E402
from analyze_resource_profile import load  # noqa: E402

ROLES = ["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def steps(ops):
    out, cur = [], []
    for o in ops:
        cur.append(o)
        if o["op"] == "ArgMax":
            out.append(cur)
            cur = []
    assert not cur
    return out


def mm_label(k, n):
    if k == 0:
        return "pre_projection"
    if k <= 20:
        return f"{ROLES[(k - 1) % 5]} L{(k - 1) // 5}"
    return "post_projection" if k == n - 1 else None


def instances(step):
    """{label: duration} for every op of one step; head matmuls labelled by weight placement."""
    out = defaultdict(float)
    mms = [o for o in step if o["op"] == "Matmul"]
    for k, o in enumerate(mms):
        lab = mm_label(k, len(mms))
        if lab is None:
            lab = "head L1 matmul" if "L1" in o["row"].get("INPUT_1_MEMORY", "") else "head DRAM matmul"
        out[lab] += o["dur"]
    for i, o in enumerate(step):
        if o["op"] != "Matmul":
            out[f"op {o['op']}"] += o["dur"]
    out["op-to-op gaps"] += sum(o["gap"] for o in step[1:])
    return out


def niu_windows(cell, sessions):
    """Per replay: {'R_tail', 'R_up', 'B_tail', 'B_up'} summed over the 3 steps (µs, bytes per call averaged)."""
    samples = read_samples(cell / "niu_samples.csv")
    ops = []  # (fw_start, fw_end, session, role)
    for sid, ops_s in sessions.items():
        for st in steps(ops_s):
            mms = [o for o in st if o["op"] == "Matmul"]
            for k, o in enumerate(mms):
                dram = "DRAM" in o["row"].get("INPUT_1_MEMORY", "")
                role = None
                if k == 4 and dram:
                    role = "up"
                elif 21 <= k < len(mms) - 1 and dram:
                    role = "tail"
                ops.append((o["fw_start"], o["fw_end"], sid, role))
    ops.sort()
    starts = [o[0] for o in ops]
    tol = 2.0 * MHZ
    per = defaultdict(lambda: defaultdict(list))
    for t, core, words in samples:
        i = bisect.bisect_right(starts, t + tol) - 1
        if i > 0 and t <= ops[i - 1][1]:
            i -= 1
        if i >= 0 and t <= ops[i][1] + tol:
            per[i][core].append((t, words))
    res = defaultdict(lambda: defaultdict(float))
    calls = defaultdict(lambda: defaultdict(int))
    for i, cores in per.items():
        _, _, sid, role = ops[i]
        if role is None:
            continue
        cores = {c: s for c, s in cores.items() if len(s) >= 2}
        if not cores:
            continue
        first = min(s[0][0] for s in cores.values())
        last = max(s[-1][0] for s in cores.values())
        words = sum((b - a) % (1 << 32) for s in cores.values() for (_, a), (_, b) in zip(s, s[1:]))
        res[sid][f"R_{role}"] += (last - first) / MHZ
        res[sid][f"B_{role}"] += words * WORD_BYTES
        calls[sid][role] += 1
    for sid in res:
        for role, n in calls[sid].items():
            res[sid][f"B_{role}"] /= n  # bytes per call
            res[sid][f"n_{role}"] = n
    return res


def stats(xs):
    n = len(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return {"mean": statistics.mean(xs), "sd": sd, "se": sd / math.sqrt(n), "n": n}


def diff(a, b):
    """b - a for two independent samples of replays."""
    return {"mean": b["mean"] - a["mean"], "se": math.sqrt(a["se"] ** 2 + b["se"] ** 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--standalone", type=Path)
    args = p.parse_args()
    cells = sorted(d for d in args.root.iterdir() if (d / "ops_perf.csv").exists())
    data = defaultdict(list)  # (arm, sampled) -> [per-replay dict]
    for d in cells:
        idx, arm = re.match(r"(\d+)_time_(.*)", d.name).groups()
        rnd = int(idx) // 6
        sampled = "+niusample" in arm
        key = arm.replace("+niusample", "")
        token = re.findall(r"K=3 output: token=\[(\d+)\] hidden_sha256=(\w+)", (d / "run.log").read_text())
        sessions = load(d / "ops_perf.csv")
        niu = niu_windows(d, sessions) if sampled else {}
        for sid, ops in sessions.items():
            rec = {"cell": d.name, "round": rnd, "token": token[-1] if token else None}
            parts, inst = defaultdict(float), defaultdict(float)
            for st in steps(ops):
                for k, v in step_parts(st).items():
                    parts[k] += v
                for k, v in instances(st).items():
                    inst[k] += v
            parts["span"] = ops[0]["dur"] + sum(o["dur"] + o["gap"] for o in ops[1:])
            rec["parts"], rec["inst"] = dict(parts), dict(inst)
            rec.update(niu.get(sid, {}))
            data[(key, sampled)].append(rec)
    out = {"cells": [d.name for d in cells], "tokens": sorted({str(r["token"]) for v in data.values() for r in v})}

    arms = sorted({k for k, _ in data})
    mb = next(a for a in arms if a.startswith("B") and "+pin" not in a)
    mp = next(a for a in arms if a.startswith("B") and "+pin" in a)
    f0 = next(a for a in arms if a.startswith("R59840"))
    h1 = next(a for a in arms if a.startswith("R49280"))

    # ---- M1
    base, pin = data[(mb, False)], data[(mp, False)]
    m1 = {"parts": {}, "inst": {}, "rounds": {}}
    for k in KEYS:
        a, b = stats([r["parts"].get(k, 0.0) for r in base]), stats([r["parts"].get(k, 0.0) for r in pin])
        m1["parts"][k] = {"base": a, "pin": b, "delta": diff(a, b)}
    for k in sorted({k for r in base + pin for k in r["inst"]}):
        a, b = stats([r["inst"].get(k, 0.0) for r in base]), stats([r["inst"].get(k, 0.0) for r in pin])
        m1["inst"][k] = {"base": a["mean"], "pin": b["mean"], "delta": b["mean"] - a["mean"], "se": diff(a, b)["se"]}
    for rnd in sorted({r["round"] for r in base}):
        sb = statistics.mean(r["parts"]["span"] for r in base if r["round"] == rnd)
        sp = statistics.mean(r["parts"]["span"] for r in pin if r["round"] == rnd)
        m1["rounds"][rnd] = sp - sb
    out["M1"] = m1
    print("\n## M1: no head in L1, up L0 pin vs base (profiled, µs per draft, mean ± SE over replays)")
    print("| part | base | pin | Δ |\n|---|--:|--:|--:|")
    for k, v in m1["parts"].items():
        print(
            f"| {k} | {v['base']['mean']:.1f} | {v['pin']['mean']:.1f} | {v['delta']['mean']:+.1f} ± {v['delta']['se']:.1f} |"
        )
    print("span Δ per round:", {k: round(v, 1) for k, v in m1["rounds"].items()})
    print("\n| instance (|Δ| ≥ 1) | base | pin | Δ ± SE |\n|---|--:|--:|--:|")
    for k, v in sorted(m1["inst"].items(), key=lambda x: -abs(x[1]["delta"])):
        if abs(v["delta"]) >= 1:
            print(f"| {k} | {v['base']:.1f} | {v['pin']:.1f} | {v['delta']:+.1f} ± {v['se']:.1f} |")

    # ---- M2
    m2 = {}
    for arm, name in ((f0, "F0"), (h1, "H1")):
        for sampled in (True, False):
            recs = data[(arm, sampled)]
            tag = f"{name} {'sampled' if sampled else 'unsampled'}"
            row = {
                "T_up": stats([r["parts"]["up_proj L0"] for r in recs]),
                "T_A": stats([r["parts"]["head slice A"] for r in recs]),
                "T_B": stats([r["parts"].get("head slice B", 0.0) for r in recs]),
                "T_tail": stats([r["parts"]["head DRAM tail"] for r in recs]),
                "T_shared": stats(
                    [r["parts"]["up_proj L0"] + r["parts"]["head slice A"] + r["parts"]["head DRAM tail"] for r in recs]
                ),
                "span": stats([r["parts"]["span"] for r in recs]),
                "rest": stats(
                    [
                        r["parts"]["span"]
                        - r["parts"]["up_proj L0"]
                        - r["parts"]["head slice A"]
                        - r["parts"].get("head slice B", 0.0)
                        - r["parts"]["head DRAM tail"]
                        for r in recs
                    ]
                ),
                "parts": {k: stats([r["parts"].get(k, 0.0) for r in recs]) for k in KEYS},
                "round_means_T_shared": {
                    rnd: statistics.mean(
                        r["parts"]["up_proj L0"] + r["parts"]["head slice A"] + r["parts"]["head DRAM tail"]
                        for r in recs
                        if r["round"] == rnd
                    )
                    for rnd in sorted({r["round"] for r in recs})
                },
            }
            if sampled:
                row["R_tail"] = stats([r["R_tail"] for r in recs])
                row["B_tail_per_call"] = statistics.mean(r["B_tail"] for r in recs)
                row["n_tail_calls"] = statistics.mean(r["n_tail"] for r in recs)
                if name == "F0":
                    row["R_up"] = stats([r["R_up"] for r in recs])
                    row["B_up_per_call"] = statistics.mean(r["B_up"] for r in recs)
                    row["Q"] = stats(
                        [
                            r["parts"]["up_proj L0"]
                            + r["parts"]["head slice A"]
                            + r["parts"]["head DRAM tail"]
                            - r["R_tail"]
                            - r["R_up"]
                            for r in recs
                        ]
                    )
                    row["R_read"] = stats([r["R_tail"] + r["R_up"] for r in recs])
                else:
                    row["Q"] = stats(
                        [
                            r["parts"]["up_proj L0"]
                            + r["parts"]["head slice A"]
                            + r["parts"]["head DRAM tail"]
                            - r["R_tail"]
                            for r in recs
                        ]
                    )
                    row["R_read"] = stats([r["R_tail"] for r in recs])
                row["round_means_Q"] = {
                    rnd: statistics.mean(
                        r["parts"]["up_proj L0"]
                        + r["parts"]["head slice A"]
                        + r["parts"]["head DRAM tail"]
                        - r["R_tail"]
                        - (r.get("R_up", 0.0) if name == "F0" else 0.0)
                        for r in recs
                        if r["round"] == rnd
                    )
                    for rnd in sorted({r["round"] for r in recs})
                }
            m2[tag] = row
    fs, hs, fu, hu = m2["F0 sampled"], m2["H1 sampled"], m2["F0 unsampled"], m2["H1 unsampled"]
    res = {
        "G_read": diff(fs["R_read"], hs["R_read"]),
        "dT_shared": diff(fs["T_shared"], hs["T_shared"]),
        "P_final": diff(fs["Q"], hs["Q"]),
        "dT_up": diff(fs["T_up"], hs["T_up"]),
        "dT_A": diff(fs["T_A"], hs["T_A"]),
        "dT_tail": diff(fs["T_tail"], hs["T_tail"]),
        "dR_tail": diff(fs["R_tail"], hs["R_tail"]),
        "T_B": hs["T_B"],
        "d_rest_sampled": diff(fs["rest"], hs["rest"]),
        "d_span_sampled": diff(fs["span"], hs["span"]),
        "dT_shared_unsampled": diff(fu["T_shared"], hu["T_shared"]),
        "T_B_unsampled": hu["T_B"],
        "d_rest_unsampled": diff(fu["rest"], hu["rest"]),
        "d_span_unsampled": diff(fu["span"], hu["span"]),
        "perturbation": {
            arm: {
                k: diff(m2[f"{arm} unsampled"][k], m2[f"{arm} sampled"][k])
                for k in ("T_up", "T_A", "T_B", "T_tail", "span")
            }
            for arm in ("F0", "H1")
        },
        "P_final_by_round": {
            rnd: hs["round_means_Q"][rnd] - fs["round_means_Q"][rnd]
            for rnd in fs["round_means_Q"]
            if rnd in hs["round_means_Q"]
        },
    }
    out["M2"] = {"arms": m2, "result": res}
    print("\n## M2: F0 vs H1 endpoints (µs per draft; mean ± SE over replays)")
    print("| | F0 sampled | H1 sampled | F0 unsampled | H1 unsampled |\n|---|--:|--:|--:|--:|")
    for k in ("T_up", "T_A", "T_B", "T_tail", "T_shared", "rest", "span"):
        print(
            f"| {k} | "
            + " | ".join(
                f"{m2[t][k]['mean']:.1f} ± {m2[t][k]['se']:.1f}"
                for t in ("F0 sampled", "H1 sampled", "F0 unsampled", "H1 unsampled")
            )
            + " |"
        )
    for k in ("R_tail", "R_up"):
        print(
            f"| {k} | "
            + " | ".join(
                f"{m2[t][k]['mean']:.1f} ± {m2[t][k]['se']:.1f}" if k in m2[t] else "—"
                for t in ("F0 sampled", "H1 sampled")
            )
            + " | — | — |"
        )
    print(
        f"NIU bytes per call: F0 tail {fs['B_tail_per_call']/1e6:.3f} MB, F0 up {fs['B_up_per_call']/1e6:.3f} MB, H1 tail {hs['B_tail_per_call']/1e6:.3f} MB"
    )
    for k, v in res.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"{k:22s} {v['mean']:+8.2f} ± {v['se']:.2f}")
    print("P_final by round:", {k: round(v, 2) for k, v in res["P_final_by_round"].items()})
    print(
        "perturbation (sampled - unsampled):",
        json.dumps({a: {k: round(v["mean"], 2) for k, v in d.items()} for a, d in res["perturbation"].items()}),
    )
    print("tokens:", out["tokens"])
    if args.standalone and args.standalone.exists():
        out["standalone"] = json.loads(args.standalone.read_text())
    (args.root / "m12.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
