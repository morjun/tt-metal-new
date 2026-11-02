#!/usr/bin/env python3
"""fig28: time added per added core in the isolated lm_head DRAM tail (gemma4-12b record §9.2b / §9.4).

Data: test_tail_width_sweep.py on build_codex_current (vc1 + ep0, in0_block_w 2), µs per call, median over rounds.
  - single-core steps: per core count, the median time over widths not ≡ 0 or 4 mod 8 (66 tiles per core: fine_codex
    + prod_codex, 96-102 cores; 75 tiles per core: core99_codex, one width per core count, 96-101 cores); step = the
    difference between consecutive core counts
  - multi-core steps: the full-core coarse sweep (prod_codex: 66 x 80, 82, ... 110; 75 x 80, 83, ... 107), step =
    time difference / cores added, drawn over the cores it spans. At 66 tiles per core every coarse width is ≡ 0 or
    4 mod 8, so each such step also carries the change in the width penalty
  - the dashed line: the slope of a line fitted to the coarse full-core points x tiles per core (the "average" step)

    python research_codes/mm_profiling/plot_tail_core_steps.py <figure path stem>
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_p4_attr import GRID, INK, INK2, ROOT, SURFACE, style  # noqa: E402

SW = ROOT / "tail_width_sweep"
SINGLE, UP, DOWN, MIXED, HI = "#0072b2", "#d55e00", "#e69f00", "#8c8b86", "#cc79a7"


def rows(name, pcn):
    return [r for r in json.loads((SW / f"{name}.json").read_text())["rows"] if r["per_core_N"] == pcn]


def cores(t, pcn):
    return -(-t // pcn)


def single_steps(names, pcn):
    pts = {}
    for n in names:
        for r in rows(n, pcn):
            pts.setdefault(r["tiles"], []).extend(r["us"])
    by = {}
    for t, v in pts.items():
        if t % 4:
            by.setdefault(cores(t, pcn), []).append(statistics.median(v))
    med = {c: statistics.median(v) for c, v in by.items()}
    cs = sorted(c for c in med if c - 1 in med)
    return {c: med[c] - med[c - 1] for c in cs}


def coarse(pcn):
    full = sorted((r["tiles"], r["median_us"]) for r in rows("prod_codex", pcn) if r["tiles"] % pcn == 0)
    n = len(full)
    mx = statistics.mean(t for t, _ in full)
    my = statistics.mean(u for _, u in full)
    slope = sum((t - mx) * (u - my) for t, u in full) / sum((t - mx) ** 2 for t, _ in full)
    segs = []
    for (t0, u0), (t1, u1) in zip(full, full[1:]):
        c0, c1 = t0 // pcn, t1 // pcn
        segs.append((c0, c1, (u1 - u0) / (c1 - c0), t0 % 8, t1 % 8))
    return segs, slope * pcn, n


def panel(ax, pcn, names, title, shade=None):
    style(ax)
    segs, avg, n = coarse(pcn)
    single = single_steps(names, pcn)
    if shade:
        ax.axvspan(shade[0] + 0.5, shade[1] + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.text((shade[0] + shade[1]) / 2 + 0.5, 4.6, "F0 → b\n(96 → 101 cores)", ha="center", fontsize=9, color=INK2)
    ax.axhline(avg, ls="--", color=INK2, lw=1.2, zorder=1)
    ax.text(
        80.6,
        avg + 0.25,
        f"average step {avg:.2f} µs\n(slope of a line through the coarse points)",
        fontsize=8.5,
        color=INK2,
    )
    for c0, c1, s, r0, r1 in segs:
        if pcn == 66:
            color = UP if (r0, r1) == (4, 0) else DOWN
        else:
            color = MIXED
        ax.plot([c0 + 0.5, c1 + 0.5], [s, s], color=color, lw=3.2, solid_capstyle="butt", zorder=2)
    xs = sorted(single)
    ax.plot(xs, [single[c] for c in xs], color=SINGLE, lw=1.4, marker="o", ms=6, mec=SURFACE, mew=1.0, zorder=3)
    for i, c in enumerate(xs):
        v = single[c]
        near_prev = i > 0 and abs(single[xs[i - 1]] - v) < 0.8
        near_next = i + 1 < len(xs) and abs(single[xs[i + 1]] - v) < 0.8
        ha, dx = ("right", -0.15) if near_next and not near_prev else ("left", 0.15) if near_prev else ("center", 0)
        dy = 0.45 if v >= avg else -0.85
        ax.text(c + dx, v + dy, f"{v:.1f}", ha=ha, fontsize=8.5, color=INK)
    if 99 in single:
        ax.plot([99], [single[99]], ls="", marker="o", ms=11, mfc="none", mec=HI, mew=2.0, zorder=4)
    ax.set_xlim(80, 111)
    ax.set_ylim(4, 19.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("core count after the step (cores 1-110 fill the grid row by row)", color=INK2, fontsize=10)
    ax.set_ylabel("tail time added per added core, µs per call", color=INK2, fontsize=10)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    return segs, single, avg


def main(stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"wspace": 0.18})
    fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.25)
    sa = panel(a, 66, ("fine_codex", "prod_codex"), "66 tiles per core", shade=(96, 101))
    sb = panel(b, 75, ("core99_codex",), "75 tiles per core")
    handles = [
        Line2D([], [], color=SINGLE, marker="o", lw=1.4),
        Line2D([], [], color=UP, lw=3.2),
        Line2D([], [], color=DOWN, lw=3.2),
        Line2D([], [], color=MIXED, lw=3.2),
        Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec=HI, mew=2.0),
        Line2D([], [], ls="--", color=INK2),
    ]
    labels = [
        "one core added (widths not ≡ 0 or 4 mod 8)",
        "2 cores added, width ≡ 4 → ≡ 0 mod 8 (penalty grows: step reads high)",
        "2 cores added, width ≡ 0 → ≡ 4 mod 8 (penalty shrinks: step reads low)",
        "3 cores added, mixed width residues",
        "the 99th core",
        "average step (line fit)",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    for name, (segs, single, avg) in (("66", sa), ("75", sb)):
        print(f"{name} tiles per core: average step {avg:.2f}")
        print("  single:", {c: round(v, 1) for c, v in single.items()})
        print("  coarse:", [(c0, c1, round(s, 2), f"{r0}->{r1}") for c0, c1, s, r0, r1 in segs])


if __name__ == "__main__":
    main(sys.argv[1])
