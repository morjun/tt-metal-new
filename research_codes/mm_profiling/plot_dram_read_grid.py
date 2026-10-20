#!/usr/bin/env python3
"""fig15 / fig16: where on the chip the DRAM weight reads go fast or slow (gemma4-12b record §9.2).

Inputs are the two noc cells of generated/lmhead_l1_profile/phase2_noc_timeline (one profiled replay each, every
NoC transfer of every op in npe_viz/). Read rates come from the MEASURED issue time of each DRAM read; tt-npe's
link demand is its congestion MODEL over the same transfers.

  fig15  per-core DRAM read rate on the physical grid: gate_proj while all 64 cores read and its per-core peak,
         gate_proj read in 16 blocks instead of 4, down_proj (32 cores), lm_head (110 cores)
  fig16  tt-npe's modelled demand per NoC link for the same gate_proj: map of the busiest link at each router,
         and over time the grid average against the busiest link

    python research_codes/mm_profiling/plot_dram_read_grid.py [--out DIR]
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from analyze_dram_read_timeline import LAYER_TYPES, MHZ
from analyze_resource_profile import label_session, load
from plot_contributions import AXIS, DEFAULT_OUT, GRID, INK, INK2, MUTED, ORANGE, SURFACE, BLUE, plt, save

CELLS = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/phase2_noc_timeline"
A0, W2 = CELLS / "00_noc_B+logitsl1", CELLS / "01_noc_B+logitsl1+blkwide2"
ROWS, COLS, DRAM_COLS = 12, 17, (0, 9)
# reference sequential ramps (dataviz palette.md): blue for read rate, orange for modelled link demand
BLUES = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("o", ["#fbe3d6", "#f5b394", "#eb6834", "#b8461c", "#7a2c10"])


def npeviz(cell, label, layer=None, nth=0):
    """The complete (unsplit) tt-npe file of the nth call of a labelled matmul."""
    (session,) = load(cell / "ops_perf.csv").values()
    calls = [
        int(o["row"]["GLOBAL CALL COUNT"]) // 1024
        for o, (_, name, lay) in zip(session, label_session(session, LAYER_TYPES))
        if name == label and (layer is None or lay == layer)
    ]
    (f,) = [
        f
        for f in (cell / "npe_viz").glob(f"MatmulDeviceOperation_ID{calls[nth]}_traceID*.npeviz.zst")
        if "_split_" not in f.name
    ]
    return json.loads(subprocess.run(["zstd", "-dc", str(f)], capture_output=True, check=True).stdout)


def dram_reads(d):
    """[(issue us, bytes, (row, col) of the reading core)] and the DRAM endpoints that served them."""
    reads = [t for t in d["noc_transfers"] if t["noc_event_type"] == "READ"]
    dram = {tuple(t["src"]) for t in reads} - {tuple(x) for t in reads for x in t["dst"]}
    out = [(t["start_cycle"] / MHZ, t["total_bytes"], tuple(t["dst"][0][1:])) for t in reads if tuple(t["src"]) in dram]
    used = defaultdict(int)
    for t in reads:
        if tuple(t["src"]) in dram:
            used[tuple(t["src"][1:])] += t["total_bytes"]
    return out, {k for k, v in used.items() if v > 0.05 * max(used.values())}


def per_core_rate(reads, t_from=None, t_to=None):
    """GB/s per core: bytes issued in [t_from, t_to), or over the core's own first-to-last read."""
    b, first, last = defaultdict(int), {}, {}
    for t, n, c in reads:
        first[c], last[c] = min(first.get(c, t), t), max(last.get(c, t), t)
        if (t_from is None or t >= t_from) and (t_to is None or t < t_to):
            b[c] += n
    if t_from is None:
        return {c: b[c] / (last[c] - first[c]) / 1e3 for c in b}
    return {c: b[c] / (t_to - t_from) / 1e3 for c in first}


def per_core_rate_alive(reads, t_from, t_to):
    """GB/s per core in [t_from, t_to), only for cores still reading at t_to (the others have finished)."""
    last = defaultdict(float)
    for t, _, c in reads:
        last[c] = max(last[c], t)
    rate = per_core_rate(reads, t_from, t_to)
    return {c: v for c, v in rate.items() if last[c] >= t_to}


def grid(ax, values, cmap, vmax, endpoints, title, fmt="{:.1f}", fs=6.5):
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.tick_params(length=0, labelsize=7, colors=MUTED)
    for r in range(ROWS):
        for c in range(COLS):
            if c in DRAM_COLS:
                served = (r, c) in endpoints
                ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color=INK2 if served else AXIS, lw=0))
                ax.text(c, r, "D", ha="center", va="center", fontsize=6, color=SURFACE)
            elif (r, c) in values:
                v = values[(r, c)]
                ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color=cmap(min(v / vmax, 1.0)), lw=0))
                ax.text(
                    c,
                    r,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=SURFACE if v / vmax > 0.45 else INK,
                )
            else:
                ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color=GRID, lw=0))
    ax.set_title(title, fontsize=11)


def colorbar(fig, ax, cmap, vmax, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, aspect=40)
    cb.set_label(label, color=INK2, fontsize=10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=MUTED)


def fig15(out):
    gate, ends = dram_reads(npeviz(A0, "gate_proj (sliding)", 0))
    gate2, _ = dram_reads(npeviz(W2, "gate_proj (sliding)", 0))
    down, ends_d = dram_reads(npeviz(A0, "down_proj (sliding)", 0))
    head, ends_h = dram_reads(npeviz(A0, "lm_head (whole, DRAM)"))
    panels = [
        (per_core_rate(gate, 0, 20), ends, "gate_proj, 64 cores: first 20 us, all cores reading"),
        (per_core_rate_alive(gate, 32, 44), ends, "gate_proj: 32-44 us, after rows 7-9 finished"),
        (per_core_rate(gate2), ends, "gate_proj read in 16 blocks instead of 4"),
        (per_core_rate(down), ends_d, "down_proj, 32 cores"),
        (per_core_rate(head), ends_h, "lm_head, 110 cores"),
    ]
    vmax = 12.0
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.6))
    for ax, (vals, eps, title) in zip(axs.flat, panels):
        grid(ax, vals, BLUES, vmax, eps, title)
    axs.flat[5].axis("off")
    colorbar(fig, axs.ravel().tolist(), BLUES, vmax, "DRAM read rate per core, GB/s (issued weight bytes per second)")
    desc = """
Where on the chip the weight reads from DRAM go fast or slow. Each square is one compute core on the
physical 17 x 12 grid, labelled with how fast it pulled its weights from DRAM (GB/s). "D" squares are
the two DRAM columns; the dark ones are the DRAM endpoints that served these reads. Grey squares are
cores the matmul does not use.

- gate_proj, first 20 us (top left): while all 64 cores read, the cores in the lower rows (8-9) get
  about 3x the rate of the cores in the upper rows (2-4). Together they pull about 380 GB/s.
- gate_proj, 32-44 us (top middle; only the cores still reading then): the upper rows are not
  capped. Once the lower rows have finished their share, the same upper-row cores read at 5-13 GB/s
  (median about 9), roughly three times their rate while every core competed. The slow rows get a smaller share of
  the bandwidth while everyone competes; they are not limited to it.
- gate_proj in 16 blocks (top right): with more, smaller steps the cores advance together, every
  core reads at about the slow rows' rate (5.8 GB/s), and all finish within a few microseconds of each
  other instead of 21-53 us. The matmul is about 9% faster.
- down_proj (bottom left): 32 cores, each reading its whole weight column from one DRAM bank, all at
  8.3-9.0 GB/s, in step with each other; about 270 GB/s in total.
- lm_head (bottom middle): 110 cores at 3.4 GB/s each, uniformly. Each core needs so little that
  none of them is the limit; the DRAM itself is (about 400 GB/s in total).

Rates are issued bytes per second from a per-transfer trace of one profiled run. Tracing slows some
matmuls (down_proj by about 24%), so the absolute values here are lower than in untraced runs; the
pattern is what the figure shows. The lm_head trace's timestamps run about 7% long, so its values are
about 7% low.
"""
    save(fig, out, "fig15_dram_read_rate_grid", desc)


def fig16(out):
    d = npeviz(A0, "gate_proj (sliding)", 0)
    _, ends = dram_reads(d)
    ts = sorted(d["timestep_data"], key=lambda t: t["start_cycle"])
    n = len(ts)
    mean = defaultdict(float)
    for t in ts:
        for _, r, c, kind, dem in t["link_demand"]:
            mean[(r, c, kind)] += dem / n
    busiest = {}
    for (r, c, kind), v in mean.items():
        if c not in DRAM_COLS or kind.endswith("EAST") or kind.endswith("WEST"):
            busiest[(r, c)] = max(busiest.get((r, c), 0.0), v)
    hot = max(mean, key=mean.get)
    us = np.array([t["start_cycle"] / MHZ for t in ts])
    avg = np.array([t["avg_link_util"] for t in ts])
    hot_t = np.array([next((x[4] for x in t["link_demand"] if (x[1], x[2], x[3]) == hot), 0.0) for t in ts])
    k = 32  # ~0.76 us smoothing, so the lines are readable

    def smooth(y):
        return np.convolve(y, np.ones(k) / k, mode="same")

    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0])
    vmax = 120.0
    grid(
        ax,
        {rc: v for rc, v in busiest.items() if rc[1] not in DRAM_COLS},
        ORANGES,
        vmax,
        ends,
        "Busiest NoC link at each router, average over the matmul",
        fmt="{:.0f}",
        fs=7,
    )
    colorbar(fig, ax, ORANGES, vmax, "modelled link demand, % of the link's bandwidth")
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(us, smooth(hot_t), color=ORANGE, lw=2)
    ax2.plot(us, smooth(avg), color=BLUE, lw=2)
    ax2.axhline(100, color=AXIS, lw=1, ls=(0, (4, 3)))
    ax2.text(us.max(), 103, "link fully used", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax2.text(
        us[int(n * 0.25)],
        smooth(hot_t).max() + 6,  # above the line's peaks, not on them
        f"busiest link (row {hot[0]}, column {hot[1]} → {hot[1] + 1})",
        fontsize=10,
        color=INK,
        ha="left",
    )
    ax2.text(
        us[int(n * 0.25)], smooth(avg)[int(n * 0.25)] + 6, "average over all links", fontsize=10, color=INK, ha="left"
    )
    ax2.set_xlabel("time since the matmul started, us")
    ax2.set_ylabel("modelled demand, %")
    ax2.set_ylim(0, 180)
    ax2.set_title("Average link versus the busiest link over time", fontsize=11)
    desc = f"""
What the NoC congestion model (tt-npe) says about the same gate_proj matmul (64 cores, weights read from
DRAM). The model replays every recorded transfer through the chip's network-on-chip and estimates how
much of each link's bandwidth it needs.

- Left: for every router, the busiest of its links, averaged over the matmul (% of that link's
  bandwidth). "D" squares are the DRAM columns; the dark ones are the endpoints that served the reads.
- Right: over time, the modelled demand on the single busiest link (row {hot[0]}, column {hot[1]} → {hot[1] + 1}:
  eastbound, leaving the DRAM column) against the average over all {len(mean)} links.

The averaged figure most profiling summaries report (about 11% here) says the network is nearly idle.
The same model puts the busiest link (row {hot[0]}, column {hot[1]} → {hot[1] + 1}) at 130-160% of its
bandwidth for most of the matmul, and several other links next to the DRAM columns at 50-100% on
average. So the network is not idle where the DRAM data enters it. This is a model of the recorded
transfers, not a measurement of the links.
"""
    save(fig, out, "fig16_npe_link_demand", desc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--only", default="fig15,fig16")
    args = p.parse_args()
    for name in args.only.split(","):
        globals()[name](args.out)


if __name__ == "__main__":
    main()
