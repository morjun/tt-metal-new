#!/usr/bin/env python3
"""fig34 of gemma4-12b record §9.2b ("The output write"): up_proj L0's per-core write time on the chip, weight in
DRAM (c) and in L1 (d), and the per-row numbers behind it.

Per core: median over every call of every marked M3 / M3b cell of the arm of the output segment (last block packed ->
all writes acknowledged). Coordinates are physical NoC0 coordinates (the device profiler's core_x / core_y; this
P150 has Tensix columns 7 and 10 harvested, so the 11-wide grid is x 1-6, 11-15; column 8 holds the ARC at (8, 0)
and NoC routers only). Core i (row-major) holds output tiles 3i .. 3i+2 (the last core 255 only); tile t goes to DRAM
bank t mod 8, written by BRISC on NoC0 to that bank's NoC0 worker endpoint (soc descriptor dram_views: b0 (0,11),
b1 (0,2), b2 (0,9), b3 (0,5), b4 (9,11), b5 (9,2), b6 (9,9), b7 (9,5)); ep0 (TT_MM1D_IN1_BANK0_XY) moves only the
weight reads of bank 0, to (0,0).

Prints, per row: the write with little other traffic in c (<= 10 other cores writing and <= 5 reading during it), in
d (every core writing, no reads), and in c for the cores that wait last for the activation (mask-3 cells, which mark
it); and the static check: per-core rank correlation of the write with the row and with the busiest link on its
NoC0 routes (+x first, then +y, wrapping; load = tile writes crossing the link).

    python research_codes/mm_profiling/plot_write_heatmap.py generated/lmhead_l1_profile \
        --fig ../lab-meeting-notes/documents/gemma4-specdec/gemma4-12b/figures/fig34_write_heatmap
"""

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_output_segments import calls

W, H = 17, 12
XS = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]  # NoC0 x of logical columns 0-10
HARVESTED = (7, 10)
EP = {0: (0, 11), 1: (0, 2), 2: (0, 9), 3: (0, 5), 4: (9, 11), 5: (9, 2), 6: (9, 9), 7: (9, 5)}
ONE_TILE = (13, 9)


def core_of(i):
    return (XS[i % 11], 2 + i // 11)


TILES = {core_of(i): [t for t in (3 * i, 3 * i + 1, 3 * i + 2) if t < 256] for i in range(86)}


def noc0(s, d):
    x, y = s
    out = []
    while x != d[0]:
        nx = (x + 1) % W
        out.append(((x, y), (nx, y)))
        x = nx
    while y != d[1]:
        ny = (y + 1) % H
        out.append(((x, y), (x, ny)))
        y = ny
    return out


def link_load():
    load = defaultdict(int)
    for c, ts in TILES.items():
        for t in ts:
            for link in noc0(c, EP[t % 8]):
                load[link] += 1
    return load


def marked_cells(root, l1):
    return [
        c
        for d in ("m3_1103", "m3b_1103")
        for c in sorted((root / d).iterdir())
        if c.is_dir() and "timeline" in c.name and (("pinup" in c.name) == l1)
    ]


def per_core_write(root, l1):
    pc = defaultdict(list)
    for cell in marked_cells(root, l1):
        for m in calls(cell):
            for c, v in m.items():
                if (0x17, 0) in v and (0x34, 3) in v:
                    pc[c].append(v[(0x17, 0)] - v[(0x34, 3)])
    return {c: statistics.median(v) for c, v in pc.items()}


def per_row(root):
    quiet, act, dall = defaultdict(list), defaultdict(list), defaultdict(list)
    for l1 in (False, True):
        for cell in marked_cells(root, l1):
            for m in calls(cell):
                ok = {c: v for c, v in m.items() if (0x17, 0) in v and (0x34, 3) in v and c != ONE_TILE}
                if len(ok) < 80:
                    continue
                for c, v in ok.items():
                    P, E = v[(0x34, 3)], v[(0x17, 0)]
                    if l1:
                        dall[c[1]].append(E - P)
                        continue
                    nw = sum(1 for c2, v2 in ok.items() if c2 != c and v2[(0x34, 3)] < E and v2[(0x17, 0)] > P)
                    nr = sum(
                        1
                        for c2, v2 in ok.items()
                        if c2 != c
                        and any(
                            (0x12, b) in v2 and (0x13, b) in v2 and v2[(0x12, b)] < E and v2[(0x13, b)] > P
                            for b in range(4)
                        )
                    )
                    if nw <= 10 and nr <= 5:
                        quiet[c[1]].append(E - P)
                    if (0x29, 3) in v and v[(0x29, 3)] > v[(0x13, 3)]:
                        act[c[1]].append(E - P)
    return quiet, dall, act


def spearman(a, b):
    ks = list(a)
    ra = {k: r for r, k in enumerate(sorted(ks, key=lambda k: a[k]))}
    rb = {k: r for r, k in enumerate(sorted(ks, key=lambda k: b[k]))}
    m = (len(ks) - 1) / 2
    num = sum((ra[k] - m) * (rb[k] - m) for k in ks)
    return num / math.sqrt(sum((ra[k] - m) ** 2 for k in ks) * sum((rb[k] - m) ** 2 for k in ks))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--fig", required=True)
    a = p.parse_args()

    load = link_load()
    arms = [
        ("weight read from DRAM", per_core_write(a.root, False)),
        ("weight already in SRAM", per_core_write(a.root, True)),
    ]
    for name, wt in arms:
        ks = [c for c in wt if c != ONE_TILE]
        bott = {c: max(load[l] for t in TILES[c] for l in noc0(c, EP[t % 8])) for c in ks}
        print(
            f"{name}: spearman(write, row) {spearman({c: c[1] for c in ks}, {c: wt[c] for c in ks}):.2f}; "
            f"(write, busiest link on the core's routes) {spearman(bott, {c: wt[c] for c in ks}):.2f}"
        )
    quiet, dall, act = per_row(a.root)
    f = lambda v: f"{statistics.median(v):.2f} (n {len(v)})" if v else "-"
    print(
        "row | c, little other traffic | d, every core writing | c, cores waiting last for the activation | busiest link on the row's routes (mean)"
    )
    for y in range(2, 10):
        cs = [c for c in TILES if c[1] == y]
        bl = statistics.mean(max(load[l] for l in noc0(c, EP[t % 8])) for c in cs for t in TILES[c])
        print(f"{y} | {f(quiet[y])} | {f(dall[y])} | {f(act[y])} | {bl:.0f}")

    INK, INK2, EDGE = "#1f2328", "#59636e", "#d55e00"
    cmap = LinearSegmentedColormap.from_list("w", ["#eef4fb", "#9cc3e6", "#2a78d6", "#0b3a75"])
    vmax = max(max(w.values()) for _, w in arms)
    norm = Normalize(0, math.ceil(vmax))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.0))
    for ax, (name, wt) in zip(axes, arms):
        for x in range(W):
            for y in range(H):
                c = (x, y)
                if c in wt:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color=cmap(norm(wt[c])), ec="white", lw=1))
                    col = "white" if norm(wt[c]) > 0.5 else INK
                    ax.text(x, y - 0.08, f"{wt[c]:.1f}", ha="center", va="center", fontsize=8, color=col)
                    if c == ONE_TILE:
                        ax.text(x, y + 0.28, "1 tile", ha="center", va="center", fontsize=5.5, color=col)
                elif x in (0, 9):
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="#f3e8d6", ec="white", lw=1))
                elif x in HARVESTED or x == 8 or y < 2:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="#e6e8eb", ec="white", lw=1))
                else:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="#f6f8fa", ec="white", lw=1))
        for b, (x, y) in EP.items():
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec=EDGE, lw=1.6))
            ax.text(x, y, f"b{b}", ha="center", va="center", fontsize=8.5, color=INK, weight="bold")
        for x, lab in ((7, "disabled"), (10, "disabled"), (8, "no cores (routers only)"), (0, "DRAM"), (9, "DRAM")):
            ax.text(
                x, 6.5 if x not in (0, 9) else 7.5, lab, rotation=90, ha="center", va="center", fontsize=7, color=INK2
            )
        ax.text(12, 0.5, "Ethernet, PCIe, ARC rows", ha="center", va="center", fontsize=7, color=INK2)
        ax.text(5, 10.5, "cores not used by this matmul", ha="center", va="center", fontsize=7, color=INK2)
        ax.set_xlim(-0.6, W - 0.4)
        ax.set_ylim(H - 0.4, -0.6)
        ax.set_aspect("equal")
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("x on the chip (NoC coordinate)", fontsize=9, color=INK2)
        ax.set_ylabel("row on the chip (NoC coordinate)", fontsize=9, color=INK2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=fig.add_axes([0.925, 0.17, 0.012, 0.68]))
    cb.set_label("output write per call, median (µs)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fill=False, ec=EDGE, lw=1.6, label="DRAM bank endpoint the writes go to (b0-b7)")
        ],
        loc="lower center",
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.91, top=0.94, bottom=0.08, wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.fig}.{ext}", dpi=200)


if __name__ == "__main__":
    main()
