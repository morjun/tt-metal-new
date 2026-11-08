#!/usr/bin/env python3
"""Analysis + figure of test_l1_fed_fixed_cost.py (gemma4-12b record §9.2b, "What an L1-fed matmul's time is made of").

Reads <dir>/cells.json (cell i = trace i) and <dir>/ops_perf.csv (legacy post-processor). Per cell: median, min, max
of the device kernel duration over the traced calls, and the median per-RISC kernel durations. Fits (least squares):
time vs per_core_N per (feed, iw) on the W cells; time vs K-steps per per_core_N on the L1-fed iw cells. Writes
<dir>/l1fc.json and, with --fig <stem>, <stem>.png / .pdf.

    python research_codes/mm_profiling/analyze_l1_fed_fixed_cost.py generated/lmhead_l1_profile/l1_fed_fixed_cost \
        --fig ../lab-meeting-notes/documents/gemma4-specdec/gemma4-12b/figures/fig32_l1_fed_fixed_cost
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

K_TILES = 32
TILE_MB = 32 * 1024 * 2 / 1e6  # one 32-column tile column of K = 1024, bf16


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return a, b, max(abs(y - (a + b * x)) for x, y in zip(xs, ys))


def load(d):
    cells = json.loads((d / "cells.json").read_text())
    by_trace = defaultdict(list)
    for r in csv.DictReader(open(d / "ops_perf.csv")):
        if "Matmul" not in r["OP CODE"] or not (r.get("METAL TRACE ID") or "").strip():
            continue
        by_trace[int(r["METAL TRACE ID"])].append(r)
    traces = sorted(by_trace)
    if len(traces) != len(cells):
        raise SystemExit(f"{len(traces)} traces for {len(cells)} cells")
    for c, t in zip(cells, traces):
        rows = by_trace[t]
        dur = [num(r["DEVICE KERNEL DURATION [ns]"]) / 1e3 for r in rows]
        c["calls"] = len(dur)
        c["samples"] = dur
        c["med"], c["min"], c["max"] = statistics.median(dur), min(dur), max(dur)
        c["cores_profiled"] = int(rows[0]["CORE COUNT"])
        for k in ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"):
            c[k] = statistics.median(num(r[f"DEVICE {k} KERNEL DURATION [ns]"]) or 0 for r in rows) / 1e3
        c["ksteps"] = K_TILES // c["iw"]
        c["mb"] = c["cores"] * c["pcn"] * TILE_MB if c["group"] != "U" else 256 * TILE_MB
    return cells


def pick(cells, **kw):
    return [c for c in cells if all(c[k] == v for k, v in kw.items())]


def report(cells):
    out = {"fits": {}}
    print(
        "| group | feed | cores | per_core_N | iw | K-steps | output | MB | median µs | min–max | BRISC | NCRISC | TRISC0 | TRISC1 | TRISC2 |"
    )
    print("|---|---|--:|--:|--:|--:|---|--:|--:|---|--:|--:|--:|--:|--:|")
    for c in cells:
        print(
            f"| {c['group']} | {c['feed']} | {c['cores']} | {c['pcn']} | {c['iw']} | {c['ksteps']} | {c['out']} | "
            f"{c['mb']:.1f} | {c['med']:.2f} | {c['min']:.2f}–{c['max']:.2f} | {c['BRISC']:.2f} | {c['NCRISC']:.2f} | "
            f"{c['TRISC0']:.2f} | {c['TRISC1']:.2f} | {c['TRISC2']:.2f} |"
        )
    for feed in ("l1", "dram"):
        for iw in (2, 8):
            w = sorted(pick(cells, group="W", feed=feed, iw=iw), key=lambda c: c["pcn"])
            a, b, r = fit([c["pcn"] for c in w], [c["med"] for c in w])
            out["fits"][f"W {feed} iw{iw}"] = {"intercept": a, "per_tile_per_core": b, "max_resid": r}
            print(f"fit W {feed} iw{iw}: {a:.2f} + {b:.3f} x per_core_N  (max residual {r:.2f})")
    for pcn in (1, 17):
        ks = [c for c in cells if c["feed"] == "l1" and c["out"] == "l1" and c["cores"] == 110 and c["pcn"] == pcn]
        ks = sorted({c["iw"]: c for c in ks}.values(), key=lambda c: c["ksteps"])
        a, b, r = fit([c["ksteps"] for c in ks], [c["med"] for c in ks])
        out["fits"][f"K l1 pcN{pcn}"] = {"intercept": a, "per_kstep": b, "max_resid": r, "iws": [c["iw"] for c in ks]}
        print(f"fit K pcN{pcn}: {a:.2f} + {b:.3f} x K-steps  (max residual {r:.2f}; iw {[c['iw'] for c in ks]})")
    out["cells"] = [{k: v for k, v in c.items() if k != "samples"} for c in cells]
    return out


def figure(cells, stem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID = "#1f2328", "#59636e", "#e6e8eb"
    L1, DRAM, MARK = "#2a78d6", "#eb6834", "#1f2328"
    # gemma4-12b drafter, traced profile (record §9.2b): slice A at iw2 on 110 cores; up L0 L1-fed / DRAM-fed
    slice_a = [(1, 26.0), (5, 28.1), (14, 30.8), (17, 31.7)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for a in (ax, bx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(INK2)
        a.tick_params(colors=INK2)
        a.yaxis.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)

    def series(a, xs, cs, color, ls, label, width):
        a.boxplot(
            [c["samples"] for c in cs],
            positions=xs,
            widths=width,
            showfliers=False,
            whis=(0, 100),
            patch_artist=True,
            boxprops=dict(facecolor=color, edgecolor=color, lw=1),
            medianprops=dict(color="white", lw=1),
            whiskerprops=dict(color=color, lw=1),
            capprops=dict(color=color, lw=1),
            manage_ticks=False,
        )
        a.plot(xs, [c["med"] for c in cs], color=color, ls=ls, lw=2, label=label)

    for feed, color in (("dram", DRAM), ("l1", L1)):
        for iw, ls in ((2, "-"), (8, "--")):
            cs = sorted(pick(cells, group="W", feed=feed, iw=iw), key=lambda c: c["pcn"])
            name = "weight in L1" if feed == "l1" else "weight in DRAM"
            series(ax, [c["mb"] for c in cs], cs, color, ls, f"{name}, in0_block_w {iw}", 2.2)
            last = cs[-1]
            ax.annotate(
                f"{name}, iw{iw}",
                (last["mb"], last["med"]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                color=INK,
            )
    ax.plot(
        [p * 110 * TILE_MB for p, _ in slice_a],
        [t for _, t in slice_a],
        ls="none",
        marker="o",
        ms=9,
        mfc="none",
        mec=MARK,
        mew=1.5,
        label="lm_head slice A inside the drafter (iw2)",
    )
    top = ax.secondary_xaxis("top", functions=(lambda mb: mb / (110 * TILE_MB), lambda p: p * 110 * TILE_MB))
    top.set_xlabel("weight tiles per core (110 cores)", color=INK2, fontsize=10)
    top.tick_params(colors=INK2)
    top.set_xticks([1, 3, 6, 8, 11, 14, 17])
    ax.set_xlabel("weight read by the matmul (MB)", color=INK2, fontsize=10)
    ax.set_ylabel("device time per call (µs)", color=INK2, fontsize=10)
    ax.set_xlim(0, 17 * 110 * TILE_MB * 1.22)
    ax.set_ylim(0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    for pcn, color, ls in ((17, L1, "-"), (1, L1, ":")):
        ks = [c for c in cells if c["feed"] == "l1" and c["out"] == "l1" and c["cores"] == 110 and c["pcn"] == pcn]
        ks = sorted({c["iw"]: c for c in ks}.values(), key=lambda c: c["ksteps"])
        series(
            bx,
            [c["ksteps"] for c in ks],
            ks,
            color,
            ls,
            f"weight in L1, {pcn} tile{'s' if pcn > 1 else ''} per core",
            0.9,
        )
        for c in ks:
            if c["iw"] == 16:
                continue  # 1 K-step from iw32: its label would sit on iw32's
            off = (0, 7) if pcn == 17 else (8, -12)
            bx.annotate(
                f"iw{c['iw']}",
                (c["ksteps"], c["med"]),
                xytext=off,
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK2,
            )
    bx.set_xlabel("K-steps per call (32 / in0_block_w)", color=INK2, fontsize=10)
    bx.set_ylabel("device time per call (µs)", color=INK2, fontsize=10)
    bx.set_ylim(0)
    bx.set_xlim(-1, 34)
    bx.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--fig")
    args = p.parse_args()
    cells = load(args.dir)
    out = report(cells)
    (args.dir / "l1fc.json").write_text(json.dumps(out, indent=1) + "\n")
    if args.fig:
        figure(cells, args.fig)


if __name__ == "__main__":
    main()
