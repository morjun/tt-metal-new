#!/usr/bin/env python3
"""Resource profile of the traced standalone drafter, per arm (gemma4-12b record §8).

Reads the cells of run_traced_opprof_arms.py (<root>/<index>_<mode>_<arm>/ops_perf.csv, legacy post-processor).
Every replayed op is labelled at two levels:
  kind       from the op code (Matmul, Attention, Normalization, Layout conversion, Eltwise, Data movement,
             Reduction, Reload);
  specific   matmuls by their ordinal within the drafter step (pre_projection, 4 x qkv/o/gate/up/down,
             lm_head [L1 slice + DRAM tail | whole], post_projection), the other ops by the matmul they follow
             and their position there (e.g. the LayerNorm after o_proj is post_attention_layernorm).
Per arm, from the `time` cells: µs per iteration (mean over replays) of DEVICE KERNEL DURATION (first start to
last end over the op's cores; ops run one after another, so these plus the op-to-op gaps tile the iteration),
PER CORE MAX, op-to-op gaps, DRAM bytes and GB/s, grid occupancy, data-movement vs compute RISC time. The
`fpu` / `unpack_pack` / `noc` cells add their counter / tt-npe columns, joined by op position.

    python analyze_resource_profile.py <root> [--wall A=9.515,B59=8.618] -> <root>/resource_profile.{json,md}
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

AICLK_GHZ = 1.35
DRAM_PEAK_GBPS = 444.7  # measured, record §5.4
ALIAS = {"R59840_64_8_rope_drisc60c2": "B59", "R59840_64_8_rope_drisc60c2+logitsl1": "B59L"}
KIND = {
    "Matmul": "Matmul",
    "SdpaDecode": "Attention",
    "RotaryEmbedding": "Attention",
    "NLPCreateQKVHeadsDecode": "Attention",
    "NLPConcatHeadsDecode": "Attention",
    "LayerNorm": "Normalization",
    "InterleavedToSharded": "Layout conversion",
    "ShardedToInterleaved": "Layout conversion",
    "TilizeWithValPadding": "Layout conversion",
    "UntilizeWithUnpadding": "Layout conversion",
    "Untilize": "Layout conversion",  # untilize of an L1 input keeps the 32-row pad (GEMMA4_LMHEAD_LOGITS_L1)
    "Copy": "Layout conversion",
    "BinaryNg": "Eltwise",
    "Unary": "Eltwise",
    "Slice": "Data movement",
    "Concat": "Data movement",
    "Embeddings": "Data movement",
    "ArgMax": "Reduction",
    "GenericOp": "Reload",
}
LAYER_ROLES = ["qkv_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# non-matmul ops by (segment = role of the matmul they follow, op code, occurrence in that segment)
SEGMENT = {
    "start": {
        ("Embeddings", 0): "target embedding lookup",
        ("BinaryNg", 0): "embedding scale",
        ("TilizeWithValPadding", 0): "tilize embedding",
        ("Concat", 0): "concat embedding + target hidden",
    },
    "pre_projection": {("LayerNorm", 0): "input_layernorm"},
    "qkv_proj": {
        ("Copy", 0): "qkv to L1",
        ("NLPCreateQKVHeadsDecode", 0): "split q/k/v heads",
        ("LayerNorm", 0): "q_norm",
        ("Embeddings", 0): "rope cos lookup",
        ("Embeddings", 1): "rope sin lookup",
        ("RotaryEmbedding", 0): "rotary embedding",
        ("SdpaDecode", 0): "SDPA",
        ("GenericOp", 0): "DRISC reload trigger",
        ("NLPConcatHeadsDecode", 0): "concat heads",
        ("Slice", 0): "slice concat heads",
    },
    "o_proj": {
        ("LayerNorm", 0): "post_attention_layernorm",
        ("BinaryNg", 0): "attention residual add",
        ("LayerNorm", 1): "pre_feedforward_layernorm",
    },
    "gate_proj": {("Unary", 0): "GELU"},
    "up_proj": {("BinaryNg", 0): "GELU(gate) x up"},
    "down_proj": {
        ("LayerNorm", 0): "post_feedforward_layernorm",
        ("BinaryNg", 0): "MLP residual add",
        ("BinaryNg", 1): "layer scalar",
        ("LayerNorm", 1): "next input_layernorm / final norm",
        ("GenericOp", 0): "lm_head reload (Tensix) / DRISC wait",
    },
    "post_projection": {
        ("UntilizeWithUnpadding", 0): "untilize logits (L1 slice or whole)",
        ("UntilizeWithUnpadding", 1): "untilize logits (DRAM tail)",
        ("Untilize", 0): "untilize logits (L1 slice or whole)",
        ("Untilize", 1): "untilize logits (DRAM tail)",
        ("Concat", 0): "concat split logits",
        ("ArgMax", 0): "argmax",
    },
}
DTYPE_BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "BFLOAT8_B": 1088 / 1024}


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x  # NaN: no value


def tensor_bytes(r, p):
    if not r.get(f"{p}_MEMORY") or "DRAM" not in r[f"{p}_MEMORY"]:
        return 0
    n = 1
    for d in "WZYX":
        m = re.match(r"(\d+)", r.get(f"{p}_{d}_PAD[LOGICAL]") or "")
        if not m:
            return 0
        n *= int(m.group(1))
    return n * DTYPE_BYTES.get(r[f"{p}_DATATYPE"], 2)


def dram_bytes(r, op):
    ins = [tensor_bytes(r, f"INPUT_{i}") for i in range(5)]
    outs = [tensor_bytes(r, f"OUTPUT_{i}") for i in range(3)]
    if op == "Embeddings":  # reads one row per index, not the table
        return ins[0] + 2 * sum(outs)
    return sum(ins) + sum(outs)


def label_step(ops, layer_types):
    """ops of one drafter step in order -> (kind, specific, layer) per op."""
    mms = [i for i, o in enumerate(ops) if o["op"] == "Matmul"]
    head = ["lm_head L1 slice", "lm_head DRAM tail", "post_projection"] if len(mms) == 24 else None
    head = head or (["lm_head (whole, DRAM)", "post_projection"] if len(mms) == 23 else None)
    assert head, f"{len(mms)} matmuls in a step"
    role = {}
    for k, i in enumerate(mms):
        if k == 0:
            role[i] = ("pre_projection", None)
        elif k <= 20:
            role[i] = (LAYER_ROLES[(k - 1) % 5], (k - 1) // 5)
        else:
            role[i] = (head[k - 21], None)
    out, seg, layer, occ = [], "start", None, defaultdict(int)
    for i, o in enumerate(ops):
        if i in role:
            seg, layer = role[i]
            occ = defaultdict(int)
            name = seg
        else:
            c = o["op"]
            key = (c, occ[c])
            occ[c] += 1
            if c in ("InterleavedToSharded", "ShardedToInterleaved"):
                name = None  # named after the norm / heads op it wraps, below
            else:
                name = SEGMENT.get(seg, {}).get(key, f"{c} after {seg} #{key[1]}")
                if name == "lm_head reload (Tensix) / DRISC wait":
                    name = "lm_head reload (Tensix)" if o["cores"] > 1 else "DRISC reload wait"
                if name == "next input_layernorm / final norm":
                    name = "final norm" if layer == len(layer_types) - 1 else "input_layernorm"
        out.append([KIND.get(o["op"], "Other"), name, layer, seg])
    for i, o in enumerate(ops):  # reshards: wrap the next (I2S) / previous (S2I) non-reshard op
        if out[i][1] is None:
            j = i + 1 if o["op"] == "InterleavedToSharded" else i - 1
            while out[j][1] is None:
                j += 1 if o["op"] == "InterleavedToSharded" else -1
            out[i][1] = ("to sharded: " if o["op"] == "InterleavedToSharded" else "to interleaved: ") + out[j][1]
    for lab in out:  # layer type where the op differs between sliding and global layers
        if lab[2] is not None and (lab[0] in ("Matmul",) or lab[1] in ("SDPA", "rotary embedding")):
            lab[1] += f" ({layer_types[lab[2]]})"
    return [tuple(x[:2]) + (x[2],) for x in out]


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if (r.get("METAL TRACE REPLAY SESSION ID") or "").strip()]
    sess = defaultdict(list)
    for r in rows:
        op = r["OP CODE"].replace("DeviceOperation", "")
        sess[int(r["METAL TRACE REPLAY SESSION ID"])].append(
            {
                "op": op,
                "cores": int(r["CORE COUNT"]),
                "dur": num(r["DEVICE KERNEL DURATION [ns]"]) / 1e3,
                "pcmax": num(r["DEVICE KERNEL DURATION PER CORE MAX [ns]"]) / 1e3,
                "gap": (num(r["OP TO OP LATENCY [ns]"]) or 0) / 1e3,
                "fw_start": int(r["DEVICE FW START CYCLE"]),
                "fw_end": int(r["DEVICE FW END CYCLE"]),
                "dm": max(num(r[f"DEVICE {k} KERNEL DURATION [ns]"]) or 0 for k in ("BRISC", "NCRISC")) / 1e3,
                "compute": max(num(r[f"DEVICE TRISC{k} KERNEL DURATION [ns]"]) or 0 for k in range(3)) / 1e3,
                "bytes": dram_bytes(r, op),
                "row": r,
            }
        )
    return {k: sorted(v, key=lambda o: o["fw_start"]) for k, v in sorted(sess.items())}


def label_session(ops, layer_types):
    """Split one replay into drafter steps (each ends at its ArgMax) and label every op."""
    labels, step = [], []
    for o in ops:
        step.append(o)
        if o["op"] == "ArgMax":
            labels += label_step(step, layer_types)
            step = []
    assert not step, "replay does not end with an ArgMax"
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--wall", default="", help="unprofiled ms per iteration per arm, e.g. A=9.515,B59=8.618")
    p.add_argument("--layer-types", default="sliding,sliding,sliding,global")
    p.add_argument("--arms", default="A,B59", help="baseline,best for the markdown tables")
    p.add_argument(
        "--extra-cols",
        type=lambda x: [c for c in x.split(",") if c],
        default=[],
        help="counter / tt-npe columns for the per-op resource table",
    )
    p.add_argument("--reload-bytes", type=float, default=32 * 344064, help="DRAM bytes per reload (32 cores)")
    args = p.parse_args()
    layer_types = args.layer_types.split(",")
    wall = {k: float(v) for k, v in (x.split("=") for x in args.wall.split(",") if x)}

    cells = defaultdict(lambda: defaultdict(list))  # arm -> mode -> [session ops with labels]
    sequences = defaultdict(set)
    for d in sorted(args.root.iterdir()):
        m = re.fullmatch(r"\d+_(time|fpu|unpack_pack|instrn|noc)_(.+)", d.name)
        if not m or not (d / "ops_perf.csv").exists():
            continue
        mode, arm = m.group(1), ALIAS.get(m.group(2), m.group(2))
        for sid, ops in load(d / "ops_perf.csv").items():
            labels = label_session(ops, layer_types)
            for o, lab in zip(ops, labels):
                o["kind"], o["name"], o["layer"] = lab
            if mode == "time":  # counter / NoC cells may lose a few ops to profiler buffer overflow
                sequences[arm].add(tuple(o["name"] for o in ops))
            cells[arm][mode].append({"cell": d.name, "session": sid, "ops": ops})
    for arm, seqs in sequences.items():
        assert len(seqs) == 1, f"{arm}: {len(seqs)} different labelled op sequences across replays / cells"

    report = {}
    for arm, modes in cells.items():
        reps = modes["time"]
        n = len(reps)
        agg = defaultdict(lambda: defaultdict(float))
        for rep in reps:
            for i, o in enumerate(rep["ops"]):
                a = agg[(o["kind"], o["name"])]
                a["count"] += 1 / n
                a["us"] += o["dur"] / n
                a["pcmax_us"] += o["pcmax"] / n
                a["gap_us"] += (o["gap"] if i else 0) / n
                a["core_us"] += o["cores"] * o["dur"] / n
                a["dm_us"] += o["dm"] / n
                a["compute_us"] += o["compute"] / n
                a["bytes"] += o["bytes"] / n
                a["cores"] = max(a["cores"], o["cores"])
        spans = [(r["ops"][-1]["fw_end"] - r["ops"][0]["fw_start"]) / AICLK_GHZ / 1e3 for r in reps]
        per_rep_us = [sum(o["dur"] for o in r["ops"]) for r in reps]
        per_rep_gap = [sum(o["gap"] for o in r["ops"][1:]) for r in reps]
        # counter / NoC columns, duration-weighted per specific op
        extra = defaultdict(lambda: defaultdict(list))
        for mode in ("fpu", "unpack_pack", "instrn", "noc"):
            for rep in modes.get(mode, []):
                for o in rep["ops"]:
                    for col, v in o["row"].items():
                        x = num(v)
                        if x is None or col.startswith(("INPUT", "OUTPUT", "DEVICE", "HOST", "OP ", "PM ")):
                            continue
                        if col in ("GLOBAL CALL COUNT", "DEVICE ID", "CORE COUNT", "AVAILABLE WORKER CORE COUNT"):
                            continue
                        if col.startswith(("METAL", "PROGRAM", "TENSIX", "ACTIVE", "IDLE", "SUB DEVICE")):
                            continue
                        extra[(o["kind"], o["name"])][col].append((x, o["dur"]))
        rows = []
        for key, a in agg.items():
            row = {"kind": key[0], "name": key[1], **a}
            for col, vals in extra.get(key, {}).items():
                w = sum(d for _, d in vals)
                row[col] = sum(x * d for x, d in vals) / w if w else statistics.mean(x for x, _ in vals)
            rows.append(row)
        reload_dram = 0
        if any(r["name"].startswith("lm_head L1 slice") for r in rows):
            reload_dram = 3 * args.reload_bytes  # 3 steps, Tensix op or DRISC push: same bytes, not in any tensor
        ref = len(reps[0]["ops"])
        report[arm] = {
            "replays": n,
            "extra_cells_ops_missing": {
                f"{rep['cell']}/{rep['session']}": ref - len(rep["ops"])
                for mode in ("fpu", "unpack_pack", "instrn", "noc")
                for rep in modes.get(mode, [])
            },
            "cells": sorted({r["cell"] for r in reps}),
            "span_us": spans,
            "kernel_us": per_rep_us,
            "gap_us": per_rep_gap,
            "wall_ms": wall.get(arm),
            "reload_dram_bytes": reload_dram,
            "rows": rows,
        }
    (args.root / "resource_profile.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({a: {k: v for k, v in r.items() if k != "rows"} for a, r in report.items()}, indent=1))
    if len(args.arms.split(",")) == 2:
        (args.root / "resource_profile.md").write_text(tables(report, *args.arms.split(","), args.extra_cols))


def tables(report, base, best, extra_cols):
    """Markdown: iteration budget, level-1 (kind) and level-2 (specific op) times, per-op resources."""
    out = []
    budget = [
        ("unprofiled wall (µs per iteration)", lambda r: r["wall_ms"] * 1e3 if r["wall_ms"] else None),
        ("profiled iteration span (µs)", lambda r: statistics.mean(r["span_us"])),
        ("Σ op kernel time (µs)", lambda r: statistics.mean(r["kernel_us"])),
        ("Σ op-to-op gaps (µs)", lambda r: statistics.mean(r["gap_us"])),
        ("DRAM bytes per iteration (MB)", lambda r: dram_total(r) / 1e6),
        (
            "avg DRAM GB/s over the unprofiled wall",
            lambda r: dram_total(r) / r["wall_ms"] / 1e6 if r["wall_ms"] else None,
        ),
        ("DRAM-bound floor: bytes / 444.7 GB/s (µs)", lambda r: dram_total(r) / DRAM_PEAK_GBPS / 1e3),
        ("grid occupancy Σ(cores × kernel) / (110 × span)", lambda r: occupancy(r)),
    ]
    out.append(f"| quantity | {base} | {best} |\n|---|--:|--:|")
    for name, f in budget:
        a, b = f(report[base]), f(report[best])
        out.append(f"| {name} | {fmt(a)} | {fmt(b)} |")
    kinds = sorted({r["kind"] for arm in (base, best) for r in report[arm]["rows"]})
    by = {arm: {(r["kind"], r["name"]): r for r in report[arm]["rows"]} for arm in (base, best)}
    span = {arm: statistics.mean(report[arm]["span_us"]) for arm in (base, best)}
    hdr = f"| {{}} | n / iter | {base} µs | {base} % | {best} µs | {best} % | Δ µs |\n|---|--:|--:|--:|--:|--:|--:|"
    out.append("\n**Level 1, op kind** (µs per iteration, kernel time; % of the profiled span)\n")
    out.append(hdr.format("kind"))
    lvl1 = {}
    for k in kinds + ["(op-to-op gaps)"]:
        if k == "(op-to-op gaps)":
            v = {arm: (0, statistics.mean(report[arm]["gap_us"])) for arm in (base, best)}
        else:
            v = {
                arm: (
                    sum(r["count"] for key, r in by[arm].items() if key[0] == k),
                    sum(r["us"] for key, r in by[arm].items() if key[0] == k),
                )
                for arm in (base, best)
            }
        lvl1[k] = v
    for k, v in sorted(lvl1.items(), key=lambda kv: -kv[1][base][1]):
        n = max(v[base][0], v[best][0])
        out.append(row(k, n, v, base, best, span))
    out.append(
        "\n**Time above the DRAM floor, per op kind** (µs per iteration; floor = the kind's DRAM bytes at "
        "444.7 GB/s; the reload's bytes are counted where they move: the Tensix reload op, or no op for the "
        "DRISC push)\n"
    )
    out.append(
        f"| kind | {base} µs | {base} floor | {base} above | {best} µs | {best} floor | {best} above | "
        f"{best} share of above |\n|---|--:|--:|--:|--:|--:|--:|--:|"
    )
    above = {}
    for arm in (base, best):
        above[arm] = {}
        for k in kinds:
            us = sum(r["us"] for r in report[arm]["rows"] if r["kind"] == k)
            floor = sum(r["bytes"] for r in report[arm]["rows"] if r["kind"] == k) / DRAM_PEAK_GBPS / 1e3
            above[arm][k] = (us, floor, us - floor)
        above[arm]["(op-to-op gaps)"] = (
            statistics.mean(report[arm]["gap_us"]),
            0,
            statistics.mean(report[arm]["gap_us"]),
        )
    tot = sum(v[2] for v in above[best].values())
    for k in sorted(above[best], key=lambda k: -above[best][k][2]):
        a, b = above[base].get(k, (0, 0, 0)), above[best][k]
        out.append(
            f"| {k} | {a[0]:.1f} | {a[1]:.1f} | {a[2]:.1f} | {b[0]:.1f} | {b[1]:.1f} | {b[2]:.1f} | "
            f"{100 * b[2] / tot:.1f}% |"
        )
    out.append(
        f"| **total** | {sum(v[0] for v in above[base].values()):.1f} | {sum(v[1] for v in above[base].values()):.1f} "
        f"| {sum(v[2] for v in above[base].values()):.1f} | {sum(v[0] for v in above[best].values()):.1f} "
        f"| {sum(v[1] for v in above[best].values()):.1f} | {tot:.1f} | 100% |"
    )
    out.append("\n**Level 2, specific op** (µs per iteration; n = instances per iteration in either arm)\n")
    out.append(hdr.format("kind / op"))
    for k, _ in sorted(lvl1.items(), key=lambda kv: -kv[1][base][1]):
        if k == "(op-to-op gaps)":
            continue
        names = sorted(
            {key[1] for arm in (base, best) for key in by[arm] if key[0] == k},
            key=lambda nm: -max(by[arm].get((k, nm), {}).get("us", 0) for arm in (base, best)),
        )
        out.append(f"| **{k}** | | | | | | |")
        for nm in names:
            v = {
                arm: (by[arm].get((k, nm), {}).get("count", 0), by[arm].get((k, nm), {}).get("us", 0))
                for arm in (base, best)
            }
            out.append(row(f"&nbsp;&nbsp;{nm}", max(v[base][0], v[best][0]), v, base, best, span))
    for arm in (base, best):
        out.append(
            f"\n**Resources per specific op, {arm}** (per iteration; GB/s = DRAM bytes / kernel time; "
            "tt-npe and counter columns duration-weighted means)\n"
        )
        cols = ["cores", "µs", "PER CORE MAX µs", "DRAM MB", "GB/s", "% of 444.7", "DM / compute RISC µs"] + extra_cols
        out.append("| kind / op | " + " | ".join(cols) + " |\n|---|" + "--:|" * len(cols))
        for r in sorted(report[arm]["rows"], key=lambda r: -r["us"]):
            gbps = r["bytes"] / r["us"] / 1e3 if r["us"] else 0
            vals = [
                r["cores"],
                r["us"],
                r["pcmax_us"],
                r["bytes"] / 1e6,
                gbps,
                100 * gbps / DRAM_PEAK_GBPS,
                f"{r['dm_us']:.0f} / {r['compute_us']:.0f}",
            ] + [r.get(c) for c in extra_cols]
            out.append(f"| {r['kind']} / {r['name']} | " + " | ".join(fmt(x) for x in vals) + " |")
    return "\n".join(out) + "\n"


def dram_total(r):
    return sum(x["bytes"] for x in r["rows"]) + r["reload_dram_bytes"]


def occupancy(r):
    return sum(x["core_us"] for x in r["rows"]) / (110 * statistics.mean(r["span_us"]))


def row(name, n, v, base, best, span):
    a, b = v[base][1], v[best][1]
    return (
        f"| {name} | {fmt(n)} | {a:.1f} | {100 * a / span[base]:.1f} | {b:.1f} | {100 * b / span[best]:.1f} "
        f"| {b - a:+.1f} |"
    )


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, int) or abs(x - round(x)) < 1e-6 and abs(x) >= 1:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}" if abs(x) < 10 else f"{x:,.1f}"


if __name__ == "__main__":
    main()
