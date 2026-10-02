#!/usr/bin/env python3
"""Whole drafter lm_head time vs out_block_w, unpinned and pinned (fig12).

Reads generated/lmhead_l1_profile/out_block_w/whole_head{,_w1}.json, written by
test_lmhead_out_block_w.py::test_drafter_whole_head_unpinned_vs_pinned. Style, palette and the
public-figure checks come from plot_contributions.py.

    python research_codes/mm_profiling/plot_out_block_w.py [--out DIR]
"""

import argparse
import json
import statistics
from pathlib import Path

from plot_contributions import BLUE, DEFAULT_OUT, INK, INK2, MUTED, SURFACE, plt, save

DATA = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/out_block_w/whole_head.json"


def medians(d):
    return {k: statistics.median(v) for k, v in d.items()}


def load():
    """Main run plus the width-1 run (same test, its own auto arms as anchors), merged."""
    runs = [json.loads((DATA.parent / n).read_text()) for n in ("whole_head.json", "whole_head_w1.json")]
    for d in runs:
        assert all(all(v.values()) for v in d["bit_identical"].values()), "an arm was not bit-identical"
    unpinned, pinned = {}, {}
    for d in runs:  # unpinned measured before and after the pinned arms in each run: pool both passes
        for k in d["unpinned_A"]:
            unpinned.setdefault(k, []).extend(d["unpinned_A"][k] + d["unpinned_B"][k])
        for k, v in d["pinned"].items():
            pinned.setdefault(k, []).extend(v)
    return runs[0], medians(unpinned), medians(pinned)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    d, unpinned, pinned = load()
    auto_u, auto_p = d["auto_resolves_to"]["unpinned"], d["auto_resolves_to"]["pinned_tail"]
    pct = 100 * d["pinned_cols"] / d["N"]
    series = {}
    for name, ser in (("unpinned", unpinned), ("pinned", pinned)):
        xs = sorted(int(k) for k in ser if k != "auto")
        series[name] = (xs, [ser[str(x)] for x in xs])

    # Broken y-axis: width 1 sits ~600-900 us above everything else and would flatten the rest.
    fig, (top, ax) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7.2), gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.06}
    )
    for a in (top, ax):
        for name, color in (("unpinned", MUTED), ("pinned", BLUE)):
            xs, ys = series[name]
            a.plot(xs, ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, clip_on=True)
    w1 = [series[n][1][0] for n in series]
    top.set_ylim(min(w1) - 120, max(w1) + 120)
    rest = [y for n in series for x, y in zip(*series[n]) if x != 1]
    ax.set_ylim(min(rest) - 60, max(rest) + 70)
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False)
    kw = dict(
        marker=[(-1, -0.5), (1, 0.5)], markersize=10, linestyle="none", color=INK2, mec=INK2, mew=1.2, clip_on=False
    )
    top.plot([0], [0], transform=top.transAxes, **kw)
    ax.plot([0], [1], transform=ax.transAxes, **kw)
    for name in ("unpinned", "pinned"):
        xs, ys = series[name]
        top.annotate(
            f"{ys[0]:,.0f} μs",
            (xs[0], ys[0]),
            xytext=(-12, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=INK,
            fontsize=11,
        )

    # Direct labels at the wide (right) end of each line.
    ax.text(77, unpinned["75"], "  whole head in DRAM", va="center", ha="left", color=INK, fontsize=12)
    ax.text(
        66,
        pinned["64"],
        f"  {pct:.1f}% of columns pinned in L1,\n  rest in DRAM",
        va="center",
        ha="left",
        color=INK,
        fontsize=12,
    )

    # ttnn's automatic choice (what runs today) and the fastest width, per line.
    for name, ser, auto_w in (("unpinned", unpinned, auto_u), ("pinned", pinned, auto_p)):
        ya = ser[str(auto_w)]
        ax.scatter([auto_w], [ya], s=260, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=4)
        ax.annotate(
            f"ttnn default\n{ya:,.0f} μs",
            (auto_w, ya),
            xytext=(0, 22),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=11,
        )
        xs, ys = series[name]
        best = min(range(len(ys)), key=ys.__getitem__)
        ax.annotate(
            f"fastest\n{ys[best]:,.0f} μs",
            (xs[best], ys[best]),
            xytext=(0, -26),
            textcoords="offset points",
            ha="center",
            va="top",
            color=INK,
            fontsize=11,
            fontweight="semibold",
        )

    ax.set_xscale("log")
    ticks = sorted(set(series["unpinned"][0]) | set(series["pinned"][0]))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    for t, lab in zip(ticks, ax.get_xticklabels()):  # 15 and 16 sit almost on top of each other
        if t == 15:
            lab.set_horizontalalignment("right")
        elif t == 16:
            lab.set_horizontalalignment("left")
    ax.minorticks_off()
    ax.set_xlim(0.6, 150)
    ax.set_xlabel("output columns computed per pass (out_block_w, in 32-column tiles)")
    fig.supylabel("whole lm_head, one draft step (μs)", x=0.035, fontsize=12, color=INK2)
    top.set_title("Drafter lm_head time vs output block width, with and without the L1-pinned slice")

    u_auto, p_auto = unpinned[str(auto_u)], pinned[str(auto_p)]
    u_best, p_best = min(series["unpinned"][1]), min(series["pinned"][1])
    desc = f"""
Time for the drafter's whole lm_head (1024 x 262,144, bf16) in one draft step, as the output block
width varies. Each core computes its share of the 262,144 output columns in passes of
out_block_w 32-column tiles; a narrower block means more passes per core. The y-axis is broken:
the top band holds only the width-1 points.

Grey: the whole weight is read from DRAM by one matmul. Blue: {d["pinned_cols"]:,} columns
({pct:.1f}%) are pinned in on-chip SRAM and read by a second matmul with its own fixed config; the
other {d["N"] - d["pinned_cols"]:,} columns are read from DRAM, and the x-axis is that DRAM part's block
width. Both matmuls are timed together as one head.

Why the two lines have points at different widths: ttnn only accepts a block width that divides the
number of output tiles each core owns. With 110 cores that is 75 tiles for the whole head
(75 = 3 x 5 x 5, so 1, 3, 5, 15, 25, 75) and 64 tiles for the DRAM part of the pinned head
(1, 2, 4, 8, 16, 32, 64). The grey line has no point at 2, 4, 8, 16, 32 or 64 because those widths
are not allowed for it; no data is missing.

Circled: what ttnn picks automatically today ({auto_u} unpinned, {auto_p} pinned; ttnn shrinks the
block when SRAM is occupied). Unpinned {u_auto:,.1f} us, pinned {p_auto:,.1f} us
({p_auto - u_auto:+.1f} us). At each line's fastest width: unpinned {u_best:,.1f} us, pinned
{p_best:,.1f} us ({p_best - u_best:+.1f} us). From width 4 up, the pinned head is faster than the
unpinned head at any width. At 2 and 1 it is not: very narrow blocks are slow because the activation
is re-sent to every core on each pass, and at width 1 the pinned head ({pinned["1"]:,.1f} us) is
even slower than the unpinned one ({unpinned["1"]:,.1f} us).

Every point is bit-identical to the unpinned default (the pinned head's two outputs joined). Both
effects act on a weight-read-bound op: down to width 3, every point streams its DRAM weight at
86-89% of the measured DRAM peak; the block width changes how efficiently the read is issued,
pinning removes bytes from it.

Op-level timing: 20 calls captured in one trace, replayed 20 times; medians of 3 counterbalanced
runs (pinned) and of 6 runs from two passes before and after the pinned runs (unpinned; the passes
agree within 0.5 us). The width-1 points come from a second run that repeated ttnn's default
widths as anchors; those agreed with the first run within 0.4 us. Synthetic weights.
"""
    save(fig, args.out, "fig12_lm_head_out_block_w", desc)


if __name__ == "__main__":
    main()
