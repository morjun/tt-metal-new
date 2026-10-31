#!/usr/bin/env python3
"""fig26: why tiles added to the lm_head DRAM tail cost more than the tail's average tile (gemma4-12b record §9.4).

Tail time T(w) at width w tiles (vc1 + ep0 time-mode profiles, the cells of plot_p4_attr.TAIL_CELLS, medians), and
its average cost per tile a(w) = T(w) / w. Going from w0 to w1 tiles adds

    T(w1) - T(w0) = (w1 - w0) × a(w1)   [the added tiles at the new average]
                  +  w0 × (a(w1) - a(w0))   [every tile already in the tail gets slower]

Left: a(w) against w for both tail configs. Right: the step cost per added tile, split into those two terms, with
up_proj L0's DRAM cost per tile for reference. A tile = 32 columns × K 1,024 × bf16 = 65,536 B.

    python research_codes/mm_profiling/plot_tail_width.py <figure path stem>
"""

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_p4_attr import cell  # noqa: E402
from plot_p4_attr import GRID, INK, INK2, ROOT, SURFACE, TAIL_CELLS, style, tail_time  # noqa: E402

TILE = 65536
CFG = {"66 / ow6": ("o", "#0072b2"), "75 / ow5": ("s", "#e69f00")}
NAMES = {6322: "F0", 6580: "H1", 6652: "b, c, d", 7354: "H4", 7378: "R4", 8192: "whole head"}
# (label, tail config, narrower width, wider width): width steps measured within one tail config
STEPS = [
    ("F0 → b\n+330 tiles (3 per core × 110 cores)", "66 / ow6", 6322, 6652),
    ("c → H4\n+702 tiles", "75 / ow5", 6652, 7354),
    ("R4 → whole head\n+814 tiles", "75 / ow5", 7378, 8192),
]
OWN, SLOW = "#56b4e9", "#d55e00"


def main(stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    pts = {}
    for path, cfg in TAIL_CELLS:
        t, us = tail_time(ROOT / path)
        pts.setdefault(cfg, {}).setdefault(t, []).append(us)
    pts = {cfg: {t: statistics.median(v) for t, v in sorted(d.items())} for cfg, d in pts.items()}
    avg = {cfg: {t: us / t * 1e3 for t, us in d.items()} for cfg, d in pts.items()}  # ns per tile per call
    up_ns = cell(next((ROOT / "opprof_p4_attr").glob("02_*")))[0]["up_proj L0"] / 3 / 256 * 1e3

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 6.8), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.42})
    fig.subplots_adjust(left=0.06, right=0.975, top=0.9, bottom=0.2)

    # left: average cost per tile against width
    style(ax)
    for cfg, d in avg.items():
        m, c = CFG[cfg]
        ax.plot(list(d), list(d.values()), marker=m, ms=8, color=c, lw=2, markeredgecolor=SURFACE)
    for t, name in NAMES.items():
        cfg = "66 / ow6" if t == 6322 else "75 / ow5"
        below = t in (6580, 7378)
        ax.annotate(
            name,
            (t, avg[cfg][t]),
            xytext=(8, -14) if below else (-8, 8),
            textcoords="offset points",
            ha="left" if below else "right",
            fontsize=9.5,
            color=INK2,
        )
    a0, a1 = avg["66 / ow6"][6322], avg["66 / ow6"][6652]
    ax.annotate(
        "",
        xy=(6652, a1),
        xytext=(6652, a0),
        arrowprops={"arrowstyle": "->", "color": SLOW, "lw": 1.8},
    )
    ax.plot([6322, 6652], [a0, a0], ls=":", color=SLOW, lw=1.4)
    ax.text(
        6690,
        (a0 + a1) / 2,
        f"+{a1 - a0:.1f} ns on each tile\nalready in the tail",
        fontsize=9.5,
        color=SLOW,
        va="center",
    )
    ax.set_xlim(6150, 8350)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("DRAM-tail width, tiles (64 KiB = 32 columns each)", color=INK2, fontsize=10.5)
    ax.set_ylabel("tail time per call ÷ width, ns per tile (axis not from 0)", color=INK2, fontsize=10.5)
    ax.set_title(
        "A wider tail is slower per tile, on every tile", loc="left", fontsize=13, fontweight="bold", color=INK
    )
    ax.legend(
        [Line2D([], [], marker=CFG[k][0], color=CFG[k][1], lw=2) for k in CFG],
        [f"tail config {k}" for k in CFG],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )

    # right: what one added tile costs, per step
    style(bx)
    rows = []
    for i, (label, cfg, w0, w1) in enumerate(STEPS):
        own = avg[cfg][w1]
        slow = w0 * (avg[cfg][w1] - avg[cfg][w0]) / (w1 - w0)
        dt = pts[cfg][w1] - pts[cfg][w0]
        y = len(STEPS) - 1 - i
        bx.barh(y, own, height=0.56, color=OWN, edgecolor=SURFACE, lw=2)
        bx.barh(y, slow, left=own, height=0.56, color=SLOW, edgecolor=SURFACE, lw=2)
        bx.text(own / 2, y, f"{own:.1f}", ha="center", va="center", fontsize=10, color=INK)
        bx.text(
            max(own + slow, up_ns) + 4,
            y,
            f"+{slow:.1f} = {own + slow:.1f} ns per added tile\n"
            f"({(w1 - w0) * own / 1e3:.1f} + {(w1 - w0) * slow / 1e3:.1f} = {dt:.1f} µs per call)",
            ha="left",
            va="center",
            fontsize=9.5,
            color=INK,
        )
        rows.append((label, own, slow, dt))
    bx.axvline(up_ns, color="#a34a00", ls=":", lw=1.6)
    bx.text(
        up_ns + 2, len(STEPS) - 0.42, f"up_proj L0 from DRAM: {up_ns:.1f} ns per tile", fontsize=9.5, color="#a34a00"
    )
    bx.set_yticks(range(len(STEPS)))
    bx.set_yticklabels([s[0] for s in reversed(STEPS)], fontsize=10, color=INK)
    bx.set_xlim(0, 270)
    bx.set_ylim(-0.5, len(STEPS) - 0.2)
    bx.xaxis.grid(True, color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("tail time added per call ÷ tiles added, ns per tile", color=INK2, fontsize=10.5)
    bx.set_title("What one tile added to the tail costs", loc="left", fontsize=13, fontweight="bold", color=INK)
    bx.legend(
        [plt.Rectangle((0, 0), 1, 1, color=OWN), plt.Rectangle((0, 0), 1, 1, color=SLOW)],
        [
            "its own share: the tail's average cost per tile at the new width",
            "the slowdown it adds to the tiles already in the tail",
        ],
        loc="upper center",
        bbox_to_anchor=(0.4, -0.12),
        ncol=1,
        frameon=False,
        fontsize=9.5,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    for cfg, d in pts.items():
        for t, us in d.items():
            print(f"{cfg} {t:5d} tiles {us:8.1f} µs/call  {avg[cfg][t]:6.2f} ns/tile  {t * TILE / us / 1e3:5.1f} GB/s")
    for label, own, slow, dt in rows:
        print(f"{label.splitlines()[0]}: own {own:.1f} + slowdown {slow:.1f} ns per added tile; {dt:.1f} µs/call")
    print(f"up_proj L0 {up_ns:.1f} ns per tile from DRAM")


if __name__ == "__main__":
    main(sys.argv[1])
