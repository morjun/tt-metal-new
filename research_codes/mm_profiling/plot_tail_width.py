#!/usr/bin/env python3
"""fig26: the lm_head DRAM tail's time against its width (gemma4-12b record §9.4).

Left: µs per call of the DRAM tail matmul at every profiled width (vc1 + ep0 time-mode profiles, the cells of
plot_p4_attr.TAIL_CELLS), with lines through the origin at F0's average per-tile rate and at up_proj L0's DRAM rate.
Right: the rate behind each width step (column-tile bytes added ÷ time added, between neighbouring widths of one tail
config) next to the average rate at each width; up_proj L0's DRAM rate and the whole head in DRAM before the
access-point swap (B+logitsl1, no ep0) for reference. A column tile = 32 columns × K 1,024 × bf16 = 65,536 B.

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
PRE_EP0 = [  # whole head in DRAM, B+logitsl1 without the access-point swap (time-mode profiles)
    "phase2_opprof/00_time_B+logitsl1",
    "phase2_vc/00_time_B+logitsl1",
    "phase2_vc/07_time_B+logitsl1",
    "phase2_gridwide/00_time_B+logitsl1",
    "phase2_gridwide/05_time_B+logitsl1",
    "phase2_blk/00_time_B+logitsl1",
    "phase2_blk/05_time_B+logitsl1",
]
CFG = {"66 / ow6": ("o", "#0072b2"), "75 / ow5": ("s", "#e69f00")}
NAMES = {6322: "F0", 6580: "H1", 6652: "b, c, d", 7354: "H4", 7378: "R4", 8192: "whole head"}


def gbs(tiles, us):
    return tiles * TILE / us / 1e3


def main(stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    pts = {}
    for path, cfg in TAIL_CELLS:
        t, us = tail_time(ROOT / path)
        pts.setdefault(cfg, {}).setdefault(t, []).append(us)
    pts = {cfg: {t: statistics.median(v) for t, v in sorted(d.items())} for cfg, d in pts.items()}
    up_us = cell(next((ROOT / "opprof_p4_attr").glob("02_*")))[0]["up_proj L0"] / 3
    up_rate = gbs(256, up_us)
    f0 = pts["66 / ow6"][6322]
    per_tile = f0 / 6322
    pre = statistics.median(tail_time(ROOT / p)[1] for p in PRE_EP0)
    pre_rate = gbs(8192, pre)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.25})
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.2)

    # left: time vs width
    style(ax)
    xs = [6150, 8350]
    ax.plot(xs, [per_tile * x for x in xs], ls="--", color=INK2, lw=1.2)
    ax.plot(xs, [x * TILE / up_rate / 1e3 for x in xs], ls=":", color="#a34a00", lw=1.6)
    for cfg, d in pts.items():
        m, c = CFG[cfg]
        ax.plot(list(d), list(d.values()), marker=m, ms=8, color=c, lw=2, markeredgecolor=SURFACE)
    for t, name in NAMES.items():
        cfg = "66 / ow6" if t == 6322 else "75 / ow5"
        below = t in (6580, 7378)
        ax.annotate(
            name,
            (t, pts[cfg][t]),
            xytext=(10, -14) if below else (-10, 8),
            textcoords="offset points",
            ha="left" if below else "right",
            fontsize=9,
            color=INK2,
        )
    ax.set_xlim(*xs)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("DRAM-tail width, tiles (64 KiB = 32 columns each)", color=INK2, fontsize=10.5)
    ax.set_ylabel("tail matmul time per call, µs", color=INK2, fontsize=10.5)
    ax.set_title("Tail time against width", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.legend(
        [Line2D([], [], marker=CFG[k][0], color=CFG[k][1], lw=2) for k in CFG]
        + [Line2D([], [], ls="--", color=INK2), Line2D([], [], ls=":", color="#a34a00", lw=1.6)],
        [f"tail config {k}" for k in CFG]
        + [
            f"proportional at F0's average ({gbs(6322, f0):.0f} GB/s)",
            f"proportional at up_proj L0's DRAM rate ({up_rate:.0f} GB/s)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )

    # right: rate per width step vs average rate
    style(bx)
    steps = []
    for cfg, d in pts.items():
        ts = list(d)
        for t0, t1 in zip(ts, ts[1:]):
            if t1 - t0 < 50:  # 7,354 -> 7,378: 24 tiles, under 1 µs apart; skipped (measurement noise dominates)
                continue
            r = gbs(t1 - t0, d[t1] - d[t0])
            steps.append((t0, t1, r, cfg))
            bx.plot([t0, t1], [r, r], color=CFG[cfg][1], lw=3.5, solid_capstyle="butt")
            bx.text((t0 + t1) / 2, r - 9, f"{r:.0f}", ha="center", va="top", fontsize=9, color=INK)
    for cfg, d in pts.items():
        m, c = CFG[cfg]
        bx.plot(
            list(d),
            [gbs(t, us) for t, us in d.items()],
            marker=m,
            ls="",
            ms=7,
            markerfacecolor="white",
            markeredgecolor=c,
            mew=1.6,
        )
    bx.axhline(up_rate, color="#a34a00", ls=":", lw=1.6)
    bx.text(6170, up_rate + 3, f"up_proj L0 from DRAM, {up_rate:.0f} GB/s", fontsize=9, color="#a34a00", va="bottom")
    bx.plot(8192, pre_rate, marker="D", ms=8, color=INK2, ls="")
    bx.annotate(
        f"whole head, before the access-point swap: {pre_rate:.0f} GB/s",
        (8192, pre_rate),
        xytext=(-12, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9,
        color=INK2,
    )
    bx.set_xlim(6150, 8350)
    bx.set_ylim(340, 510)
    bx.xaxis.grid(True, color=GRID, lw=0.8)
    bx.yaxis.grid(True, color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    bx.set_xlabel("DRAM-tail width, tiles (64 KiB = 32 columns each)", color=INK2, fontsize=10.5)
    bx.set_ylabel("GB/s", color=INK2, fontsize=10.5)
    bx.set_title("Average rate vs the rate of each width step", loc="left", fontsize=13, fontweight="bold", color=INK)
    bx.legend(
        [Line2D([], [], color=CFG[k][1], lw=3.5) for k in CFG]
        + [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, mew=1.6)],
        [f"rate of the step between two widths, {k}" for k in CFG]
        + ["average rate at that width (tail bytes ÷ tail time)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    for cfg, d in pts.items():
        for t, us in d.items():
            print(
                f"{cfg} {t:5d} tiles {us:8.1f} µs/call  avg {gbs(t, us):5.1f} GB/s  above F0 rate {us - per_tile * t:+5.1f}"
            )
    for t0, t1, r, cfg in steps:
        print(f"step {cfg} {t0}->{t1}: {r:.0f} GB/s")
    print(f"up_proj L0 {up_us:.2f} µs/call = {up_rate:.0f} GB/s; whole head pre-ep0 {pre:.1f} µs = {pre_rate:.0f} GB/s")


if __name__ == "__main__":
    main(sys.argv[1])
