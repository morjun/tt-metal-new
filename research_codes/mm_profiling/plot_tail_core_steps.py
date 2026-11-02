#!/usr/bin/env python3
"""fig28: lm_head DRAM-tail time against core count (gemma4-12b record §9.2b / §9.4).

Isolated tail matmul at 66 tiles per core (test_tail_width_sweep.py on build_codex_current; vc1 + ep0, in0_block_w 2),
µs per call. Every measured width (fine_codex + prod_codex; each width = median of its rounds) is grouped by its core
count ceil(width / 66). Within a core count the time does not depend on how full the last core is (its phantom reads
fill it), only on the width's class mod 8 (≡ 0, ≡ 4, other). At 96-102 cores (several widths each): one line per class
through the per-core-count median of that class's widths. At 80-94 and 104-110 (one full-core width 66 x cores each,
always ≡ 0 or 4 mod 8): one grey line, joined to the class line of the full-core width's class at 96 and 102.

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


INNER = range(96, 103)


def main(stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    by = load()
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.17)
    style(ax)
    # outside 96-102: one full-core width (66 x cores) per core count
    outer = {c: v[0][1] for c, v in by.items() if c not in INNER}
    assert all(len(by[c]) == 1 and by[c][0][0] == PCN * c for c in outer)
    # inside 96-102: one line per width class, each point = median over that class's widths at that core count
    lines = {k: {} for k in CLASS}
    for c in INNER:
        for k in CLASS:
            us = [u for t, u in by[c] if cls(t) == k]
            if us:
                lines[k][c] = statistics.median(us)
    lo, hi = min(INNER), max(INNER)
    left, right = sorted(c for c in outer if c < lo), sorted(c for c in outer if c > hi)
    join_lo, join_hi = lines[cls(PCN * lo)][lo], lines[cls(PCN * hi)][hi]
    for xs, ys in (
        (left + [lo], [outer[c] for c in left] + [join_lo]),
        ([hi] + right, [join_hi] + [outer[c] for c in right]),
    ):
        ax.plot(xs, ys, color=INK2, lw=1.3, zorder=1)
    for c in outer:
        ax.plot(c, outer[c], ls="", marker="o", ms=6, color=CLASS[cls(PCN * c)][1], mec=SURFACE, mew=0.6, zorder=3)
    for k, pts in lines.items():
        xs = sorted(pts)
        ax.plot(xs, [pts[c] for c in xs], color=CLASS[k][1], lw=1.8, marker="o", ms=6, mec=SURFACE, mew=0.6, zorder=2)
    for c, t, name in ((96, 6322, "F0"), (101, 6652, "b")):
        u = lines[cls(t)][c]
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
        [Line2D([], [], color=CLASS[k][1], lw=1.8, marker="o") for k in CLASS] + [Line2D([], [], color=INK2, lw=1.3)],
        [f"{CLASS[k][0]} (96-102 cores: median over those widths)" for k in CLASS]
        + ["80-94, 104-110 cores: full-core width (66 x cores), dot coloured by its class"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    for k, pts in lines.items():
        print(CLASS[k][0], {c: round(u, 1) for c, u in pts.items()})
    print("outer", {c: round(u, 1) for c, u in outer.items()})


if __name__ == "__main__":
    main(sys.argv[1])
