#!/usr/bin/env python3
"""fig33 of gemma4-12b record §9.2b ("The output write"): up_proj L0 per core, weight in DRAM (c) vs in L1 (d).

Same data as fig34 (plot_write_heatmap.py): every call of every marked M3 / M3b cell of the arm (16 cells, 240 calls).
Every time is the per-core median over those calls of (marker - the call's first core start). The write is the median
of (writes complete - last pack), i.e. fig34's value, drawn from the median last pack. The activation arrival and
which input a core waited for last come from the cells that mark the arrival (masks 3 and 33; 4 cells, 60 calls).

Left: three receiver cores, every K-block's weight read (c), activation arrival, compute (inputs available -> packed)
and the output write. c: the cores with the earliest, median and latest last weight block; d: the earliest, median
and latest to finish. The core with 1 output tile (ONE_TILE) is not picked. Right: every core, sorted by its last
pack: the last pack (marker: which input it waited for last in most calls) and its write. Dashed line: the median
call end. Prints the selected cores.

    python research_codes/mm_profiling/plot_pack_spread.py generated/lmhead_l1_profile \
        --fig ../lab-meeting-notes/documents/gemma4-specdec/gemma4-12b/figures/fig33_pack_spread
"""

import argparse
import statistics
from pathlib import Path

from collections import defaultdict

from analyze_output_segments import calls
from plot_write_heatmap import marked_cells

INK, INK2, GRID = "#1f2328", "#59636e", "#e6e8eb"
WEIGHT, COMPUTE, OUTPUT, ACT = "#5d6b7e", "#e69f00", "#0072b2", "#009e73"
NB = 4
# 256 output tiles at 3 per core: the 86th core holds 1 (it writes 1 tile; 1 output-wait exit in the mask-65 cells)
ONE_TILE = (13, 9)


def aggregate(root, l1):
    """Per core: median over all calls of each marker's time from the call start; write; share of calls it waited
    last for the activation. Also the median call end and the number of calls."""
    t, wr, gate, ends = defaultdict(lambda: defaultdict(list)), defaultdict(list), defaultdict(list), []
    for cell in marked_cells(root, l1):
        for m in calls(cell):
            ok = {c: v for c, v in m.items() if (0x17, 0) in v and all((0x34, b) in v for b in range(NB))}
            if len(ok) < 80:
                continue
            t0 = min(v[(0x10, 0)] for v in ok.values())
            ends.append(max(v[(0x17, 0)] for v in ok.values()) - t0)
            for c, v in ok.items():
                for k, x in v.items():
                    t[c][k].append(x - t0)
                wr[c].append(v[(0x17, 0)] - v[(0x34, NB - 1)])
                if (0x29, NB - 1) in v:
                    gate[c].append(l1 or v[(0x29, NB - 1)] > v[(0x13, NB - 1)])
    med = {c: {k: statistics.median(x) for k, x in v.items()} for c, v in t.items()}
    for c in med:
        med[c]["write"] = statistics.median(wr[c])
        med[c]["end"] = med[c][(0x34, NB - 1)] + med[c]["write"]
        med[c]["gate"] = statistics.mean(gate[c]) if gate[c] else None
    return med, statistics.median(ends), len(ends)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--fig", required=True)
    a = p.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), gridspec_kw={"width_ratios": [1.55, 1], "height_ratios": [1, 1]})
    for row, l1 in enumerate((False, True)):
        rel, end, n = aggregate(a.root, l1)
        recv = {c: v for c, v in rel.items() if (0x29, NB - 1) in v}
        three = [c for c in recv if c != ONE_TILE]
        act_last = statistics.median(v[(0x29, NB - 1)] for v in recv.values())
        key = (lambda c: recv[c]["end"]) if l1 else (lambda c: recv[c][(0x13, NB - 1)])
        order = sorted(three, key=key)
        pick = [("slowest", order[-1]), ("median", order[len(order) // 2]), ("fastest", order[0])]
        print(
            f"{'d' if l1 else 'c'}: medians over {n} calls, median call end {end:.2f} µs; "
            f"cores {[(k, c, round(recv[c]['write'], 2)) for k, c in pick]}"
        )

        ax = axes[row][0]
        for y, (name, c) in enumerate(pick):
            v = recv[c]
            if l1:
                ax.broken_barh([(v[(0x10, 0)], v[(0x14, 0)] - v[(0x10, 0)] + 0.05)], (y + 0.08, 0.26), color=WEIGHT)
            else:
                for b in range(NB):
                    s, e = v[(0x12, b)], v[(0x13, b)]
                    ax.broken_barh([(s, e - s)], (y + 0.08, 0.26), color=WEIGHT, edgecolor="white", lw=1)
                    if y == 2 or b == NB - 1:
                        ax.text((s + e) / 2, y + 0.21, f"W{b}", ha="center", va="center", fontsize=7.5, color="white")
            for b in range(NB):
                s, e = v[(0x32, b)], v[(0x34, b)]
                ax.broken_barh([(s, e - s)], (y - 0.3, 0.26), color=COMPUTE, edgecolor="white", lw=0.6)
                ax.plot([v[(0x29, b)]] * 2, [y - 0.36, y + 0.04], color=ACT, lw=2)
            s, e = v[(0x34, NB - 1)], v["end"]
            ax.broken_barh([(s, e - s)], (y - 0.3, 0.26), color=OUTPUT)
            ax.text(e + 0.15, y - 0.17, f"write {e - s:.2f}", va="center", fontsize=8, color=INK)
        ax.axvline(act_last, color=ACT, ls=":", lw=1.2)
        ax.text(
            act_last,
            2.55,
            "last activation block\nreaches every core",
            color=INK2,
            fontsize=8,
            ha="center",
            va="bottom",
        )
        ax.axvline(end, color=INK2, ls="--", lw=1)
        ax.set_yticks(range(3))
        lab = "finish" if l1 else "last weight block"
        ax.set_yticklabels([f"{n} {lab}\ncore ({c[0]}, {c[1]})" for n, c in pick], fontsize=8.5, color=INK)
        ax.set_ylim(-0.6, 3.15)
        ax.set_xlim(0, end + 2.2)
        ax.set_title(
            "weight read from DRAM" if not l1 else "weight already in SRAM", loc="left", fontsize=10.5, color=INK
        )
        ax.set_xlabel(f"µs from the matmul's start (median over {n} calls)", fontsize=9.5, color=INK2)

        bx = axes[row][1]
        allc = sorted(rel, key=lambda c: rel[c][(0x34, NB - 1)])
        for y, c in enumerate(allc):
            v = rel[c]
            p_, e_ = v[(0x34, NB - 1)], v["end"]
            bx.plot([p_, e_], [y, y], color=OUTPUT, lw=1.6, solid_capstyle="butt")
            act_bound = v["gate"] is not None and v["gate"] > 0.5
            if (0x29, NB - 1) not in v:
                bx.plot(p_, y, marker="D", ms=4, color=INK, ls="none")
            elif c == ONE_TILE:
                bx.plot(p_, y, marker="x", ms=5, color=INK, ls="none")
            else:
                bx.plot(p_, y, marker="o" if act_bound else "s", ms=3.6, color=ACT if act_bound else WEIGHT, ls="none")
        bx.axvline(act_last, color=ACT, ls=":", lw=1.2)
        bx.axvline(end, color=INK2, ls="--", lw=1)
        lo = act_last - (1.0 if l1 else 2.0)
        bx.set_xlim(lo, end + 0.6)
        bx.set_ylim(-1, len(allc))
        bx.set_yticks([])
        bx.set_ylabel(f"all {len(allc)} cores, sorted by last pack", fontsize=9, color=INK2)
        bx.set_xlabel("µs from the matmul's start", fontsize=9.5, color=INK2)
        nA = sum(1 for c in recv if recv[c]["gate"] > 0.5)
        bx.set_title(
            f"{nA} cores usually waited last for the activation, {len(recv) - nA} for their own weight",
            loc="left",
            fontsize=9.5,
            color=INK,
        )
        for a_ in (ax, bx):
            for s in ("top", "right"):
                a_.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a_.spines[s].set_color(INK2)
            a_.tick_params(colors=INK2, labelsize=8.5)
            a_.xaxis.grid(True, color=GRID, lw=0.8)
            a_.set_axisbelow(True)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(color=WEIGHT, label="weight block read (DRAM) / shard handed over (SRAM)"),
        Line2D([], [], color=ACT, lw=2, label="activation block arrives"),
        Patch(color=COMPUTE, label="compute of one K-block (inputs ready → packed)"),
        Patch(color=OUTPUT, label="output write (last pack → all writes complete)"),
        Line2D([], [], marker="o", color=ACT, ls="none", label="last pack, waited last for the activation"),
        Line2D([], [], marker="s", color=WEIGHT, ls="none", label="last pack, waited last for its own weight"),
        Line2D([], [], marker="D", color=INK, ls="none", label="last pack, the activation sender core"),
        Line2D([], [], marker="x", color=INK, ls="none", label="last pack, the core with 1 output tile"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.fig}.{ext}", dpi=200)


if __name__ == "__main__":
    main()
