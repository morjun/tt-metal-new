#!/usr/bin/env python3
"""fig26: what the lm_head DRAM tail's time depends on (gemma4-12b record §9.4, tail slowdown profiling).

Isolated tail matmul at the drafter's config (test_tail_width_sweep.py on build_codex_current; vc1 + ep0, 66 / ow6,
in0_block_w 2), µs per call, median over rounds: every width 6,600-6,680, every 11th width 6,281-6,699, and the
coarse sweep's widths in that range. Left: time against width, coloured by width mod 8, with the proportional line
(134.92 ns per tile read, fitted on the coarse sweep). Right: the F0 -> b step split into the proportional part,
the core-count steps above nominal (mostly the 99th core) and b's width-mod-8 penalty.

    python research_codes/mm_profiling/plot_tail_width.py <figure path stem>
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

PCN = 66
NS_PER_TILE = 0.13492  # µs per tile read, fit over the coarse 66 / ow6 sweep (record §9.4 stage 1)
CLASS = {0: ("width ≡ 0 mod 8", "#d55e00"), 4: ("width ≡ 4 mod 8", "#e69f00"), None: ("other widths", "#0072b2")}
LO, HI = 6275, 6710


def load():
    pts = {}
    for name in ("fine_codex", "prod_codex"):
        for r in json.loads((ROOT / "tail_width_sweep" / f"{name}.json").read_text())["rows"]:
            if r["per_core_N"] == PCN and LO <= r["tiles"] <= HI:
                pts.setdefault(r["tiles"], []).extend(r["us"])
    return {t: statistics.median(v) for t, v in sorted(pts.items())}


def cores(t):
    return -(-t // PCN)


def main(stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    pts = load()
    base = {}  # per core count: median over widths not 0 / 4 mod 8
    for t, us in pts.items():
        if t % 4:
            base.setdefault(cores(t), []).append(us)
    base = {c: statistics.median(v) for c, v in base.items()}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 6.8), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.3})
    fig.subplots_adjust(left=0.055, right=0.98, top=0.9, bottom=0.2)

    # left: time against width
    style(ax)
    for c in range(cores(LO), cores(HI) + 1):
        x0 = (c - 1) * PCN
        if LO < x0 < HI:
            ax.axvline(x0 + 0.5, color=GRID, lw=1.0, zorder=0)
        mid = max(LO, x0) + (min(HI, c * PCN) - max(LO, x0)) / 2
        ax.text(mid, 838, f"{c} cores", ha="center", fontsize=8.5, color=INK2)
    ax.plot([LO, HI], [NS_PER_TILE * LO, NS_PER_TILE * HI], ls="--", color=INK2, lw=1.2)
    for key, (label, color) in CLASS.items():
        xs = [t for t in pts if (t % 8 == key if key is not None else t % 4)]
        ax.plot(xs, [pts[t] for t in xs], ls="", marker="o", ms=4.5, color=color, markeredgecolor=SURFACE, mew=0.6)
    s99 = 98 * PCN
    ax.annotate(
        "99th core added:\n+14.2 µs instead of +8.9",
        xy=(s99 + 0.5, (base[98] + base[99]) / 2),
        xytext=(s99 - 190, base[99] + 12),
        fontsize=9.5,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK2, "lw": 1.1},
    )
    for t, name in ((6322, "F0"), (6652, "b")):
        ax.annotate(
            name,
            (t, pts[t]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=INK,
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
        )
    ax.set_xlim(LO, HI)
    ax.set_ylim(832, 922)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("DRAM-tail width, tiles (64 KiB = 32 columns each); 66 tiles per core", color=INK2, fontsize=10.5)
    ax.set_ylabel("tail matmul time per call, µs (axis not from 0)", color=INK2, fontsize=10.5)
    ax.set_title(
        "Time steps with each added core, plus a penalty at widths ≡ 0 or 4 mod 8",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    ax.legend(
        [Line2D([], [], ls="", marker="o", color=CLASS[k][1]) for k in CLASS] + [Line2D([], [], ls="--", color=INK2)],
        [CLASS[k][0] for k in CLASS] + [f"proportional: {NS_PER_TILE * 1e3:.1f} ns per tile read"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )

    # right: F0 -> b decomposition
    style(bx)
    f0, b = pts[6322], pts[6652]
    prop = (6652 - 6322) * NS_PER_TILE
    steps = base[cores(6652)] - base[cores(6322)] - prop
    pen = b - base[cores(6652)]
    other = (b - f0) - prop - steps - pen
    parts = [
        ("330 tiles at the proportional cost", prop, "#56b4e9"),
        ("5 added cores, above nominal\n(mostly the 99th core)", steps, "#009e73"),
        ("b's width (6,652) ≡ 4 mod 8", pen, "#e69f00"),
    ]
    left = 0.0
    for label, v, color in parts:
        bx.barh(0, v, left=left, height=0.5, color=color, edgecolor=SURFACE, lw=2)
        bx.text(left + v / 2, 0.36, f"{v:.1f}", ha="center", va="bottom", fontsize=10.5, color=INK)
        left += v
    bx.text(left + 1, 0, f"= {left:.1f} µs per call\n(measured {b - f0:.1f})", va="center", fontsize=10.5, color=INK)
    bx.set_xlim(0, 75)
    bx.set_ylim(-0.45, 0.65)
    bx.set_yticks([0])
    bx.set_yticklabels(["F0 → b\n6,322 → 6,652 tiles\n96 → 101 cores"], fontsize=10, color=INK)
    bx.tick_params(axis="y", length=0)
    bx.xaxis.grid(True, color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    bx.set_xlabel("added tail time per call, µs", color=INK2, fontsize=10.5)
    bx.set_title("Where the F0 → b step goes", loc="left", fontsize=12.5, fontweight="bold", color=INK)
    bx.legend(
        [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts],
        [p[0] for p in parts],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"F0 {f0:.1f}  b {b:.1f}  step {b - f0:.1f} = prop {prop:.1f} + core steps {steps:.1f} + penalty {pen:.1f}")
    print(
        f"unexplained {other:+.2f}; per-core-count bases " + ", ".join(f"{c}: {v:.1f}" for c, v in sorted(base.items()))
    )


if __name__ == "__main__":
    main(sys.argv[1])
