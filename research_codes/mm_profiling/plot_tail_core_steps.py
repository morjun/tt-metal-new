#!/usr/bin/env python3
"""fig28: lm_head DRAM-tail time against core count (gemma4-12b record §9.2b / §9.4).

Isolated tail matmul at 66 tiles per core (test_tail_width_sweep.py on build_codex_current; vc1 + ep0, in0_block_w 2),
µs per call. Every measured width (fine_codex + prod_codex; each width = median of its rounds) is grouped by its core
count ceil(width / 66). Within a core count the time does not depend on how full the last core is (its phantom reads
fill it), only on whether the width is ≡ 0 or 4 mod 8. One box per core count with ≥ 3 widths; one dot per width,
coloured by width mod 8; the line joins the per-core-count medians.

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

PCN = 66
CLASS = {0: ("width ≡ 0 mod 8", "#d55e00"), 4: ("width ≡ 4 mod 8", "#e69f00"), None: ("other widths", "#0072b2")}


def load():
    pts = {}
    for name in ("fine_codex", "prod_codex"):
        for r in json.loads((ROOT / "tail_width_sweep" / f"{name}.json").read_text())["rows"]:
            if r["per_core_N"] == PCN:
                pts.setdefault(r["tiles"], []).extend(r["us"])
    by = {}
    for t, v in pts.items():
        by.setdefault(-(-t // PCN), []).append((t, statistics.median(v)))
    return {c: sorted(v) for c, v in sorted(by.items()) if 80 <= c <= 110}


def cls(t):
    return t % 8 if t % 4 == 0 else None


def main(stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    by = load()
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.17)
    style(ax)
    cs = list(by)
    med = {c: statistics.median(u for _, u in by[c]) for c in cs}
    boxed = [c for c in cs if len(by[c]) >= 3]
    ax.boxplot(
        [[u for _, u in by[c]] for c in boxed],
        positions=boxed,
        widths=0.6,
        showfliers=False,
        manage_ticks=False,
        whis=(0, 100),
        medianprops={"color": INK, "lw": 1.4},
        boxprops={"color": INK2},
        whiskerprops={"color": INK2},
        capprops={"color": INK2},
        zorder=2,
    )
    ax.plot(cs, [med[c] for c in cs], color=INK2, lw=1.3, zorder=1)
    for c in cs:
        for t, u in by[c]:
            ax.plot(
                c,
                u,
                ls="",
                marker="o",
                ms=4,
                color=CLASS[cls(t)][1],
                mec=SURFACE,
                mew=0.5,
                zorder=3,
            )
    for c, t, name in ((96, 6322, "F0"), (101, 6652, "b")):
        u = dict(by[c])[t]
        ax.annotate(
            name,
            (c, u),
            xytext=(-26, 14),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=INK,
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
        )
    ax.set_xlim(79, 111)
    ax.set_xticks(range(80, 111, 2))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "cores used by the tail (66 weight tiles per core; cores fill the 11 x 10 grid row by row)", color=INK2
    )
    ax.set_ylabel("DRAM-tail time per call, µs (axis not from 0)", color=INK2)
    ax.set_title("DRAM-tail time against core count", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.legend(
        [Line2D([], [], ls="", marker="o", color=CLASS[k][1]) for k in CLASS] + [Line2D([], [], color=INK2, lw=1.3)],
        [CLASS[k][0] for k in CLASS] + ["median per core count (box: all widths at that core count)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    for c in cs:
        print(c, len(by[c]), round(med[c], 1), [(t, round(u, 1)) for t, u in by[c]][:4])


if __name__ == "__main__":
    main(sys.argv[1])
