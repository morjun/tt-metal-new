#!/usr/bin/env python3
"""fig29: up_proj L0's slowest core, weight from DRAM (arm c) vs weight in SRAM (arm d) (gemma4-12b record §9.2b, M3).

From analyze_timeline.py's m3.json (minimal marker set, mask 1, the only set inside the 1% perturbation limit in both
arms). Activation deliveries: M3's mask 3 for c (inside the limit) and M3b's receiver-only mask 33 for d (+1.5%, over it).
The weight in SRAM has no read; its one CB push takes 0.06 µs and is not drawn. Median over calls of the slowest core's
event times, µs from the op's first kernel start.

    python research_codes/mm_profiling/plot_m3_timeline.py <m3.json> <m3b.json> <figures dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK, INK2, GRID, SURFACE = "#1f2328", "#59636e", "#e6e8eb", "#ffffff"
WEIGHT, COMPUTE, OUTPUT, ACT = "#5d6b7e", "#e69f00", "#0072b2", "#009e73"


def main(m3, m3b, figdir):
    s = json.loads(Path(m3).read_text())["summary"]
    sb = json.loads(Path(m3b).read_text())["summary"] if Path(m3b).exists() else {}
    # activation deliveries: c from M3's mask 3 (inside the 1% limit), d from M3b's receiver-only mask 33 (+1.5%)
    rows = [
        ("weight from DRAM", s["c mask 1"], s["c mask 3"]),
        ("weight in SRAM", s["d mask 1"], sb.get("d mask 33", {})),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.86, bottom=0.27)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK2)
    ax.tick_params(colors=INK2)
    h = 0.22
    for i, (label, v, vb) in enumerate(rows):
        y = i
        # weight lane
        if v.get("b0_weight_issue") is not None:
            for b in range(4):
                a, e = v[f"b{b}_weight_issue"], v[f"b{b}_weight_avail"]
                ax.barh(y - h, e - a, left=a, height=h * 0.9, color=WEIGHT, edgecolor=SURFACE, linewidth=1.5)
        # compute lane
        for b in range(4):
            a = v[f"b{b}_inputs_ready"]
            ax.barh(y, v[f"b{b}_compute"], left=a, height=h * 0.9, color=COMPUTE, edgecolor=SURFACE, linewidth=1.0)
        # output lane
        last = v["b3_inputs_ready"] + v["b3_compute"]
        ax.barh(y + h, v["slow_end"] - last, left=last, height=h * 0.9, color=OUTPUT, edgecolor=SURFACE)
        # activation deliveries (receivers' median)
        for b in range(4):
            t = vb.get(f"r{b}_delivered_median")
            if t is not None:
                ax.plot([t, t], [y - 1.6 * h, y + 1.6 * h], color=ACT, lw=1.6)
        ax.text(
            v["slow_end"] + 0.4, y, f"{v['slow_end']:.1f} µs", va="center", fontsize=10, color=INK, fontweight="bold"
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10.5, color=INK)
    ax.set_ylim(len(rows) - 0.4, -0.55)
    ax.set_xlim(0, 42)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("µs from the matmul's start (one call; median over 30 calls)", color=INK2, fontsize=10)
    ax.set_title(
        "up_proj layer 0, slowest core: what each block waits for",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    ax.legend(
        handles=[
            Patch(color=WEIGHT, label="weight block read from DRAM (issue → complete)"),
            Patch(color=COMPUTE, label="compute of one block (inputs available → packed)"),
            Patch(color=OUTPUT, label="output write (last block packed → written)"),
            plt.Line2D([0], [0], color=ACT, lw=1.6, label="activation block delivered"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.42, -0.17),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    out = Path(figdir) / "fig29_up_l0_timeline"
    fig.savefig(out.with_suffix(".png"), dpi=150, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
