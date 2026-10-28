#!/usr/bin/env python3
"""fig15 / fig16: where on the chip the DRAM weight reads go fast or slow (gemma4-12b record §9.2).

Inputs are the two noc cells of generated/lmhead_l1_profile/phase2_noc_timeline (one profiled replay each, every
NoC transfer of every op in npe_viz/). Read rates come from the MEASURED issue time of each DRAM read; tt-npe's
link demand is its congestion MODEL over the same transfers.

  fig15  per-core DRAM read rate on the physical grid: gate_proj while all 64 cores read and its per-core peak,
         gate_proj read in 16 blocks instead of 4, down_proj (32 cores), global-layer qkv (48 cores)
  fig16  tt-npe's modelled demand per NoC link for the same gate_proj: map of the busiest link at each router,
         and over time the grid average against the busiest link
  fig16b_split  fig16b's two runs as separate figures (fig16b1 NoC0 default, fig16b2 NoC1 swapped)
  fig16c fig16 before and after the access-point swap (bank 0 read through (0, 0); cells in npe_ep0)

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
GRIDWIDE = CELLS.parent / "phase2_gridwide"  # 06 A (64 cores), 07 wide grid (86 / 96), 08 wide grid + 16 K-steps
ROWS, COLS, DRAM_COLS = 12, 17, (0, 9)
# reference sequential ramps (dataviz palette.md): blue for read rate, orange for modelled link demand
BLUES = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("o", ["#fbe3d6", "#f5b394", "#eb6834", "#b8461c", "#7a2c10"])


def npeviz(cell, label, layer=None, nth=0, viz="npe_viz"):
    """The complete (unsplit) tt-npe file of the nth call of a labelled matmul (viz: the cell's npe_viz, or
    npe_sizefix.py's sizefix/fixed/npe_viz)."""
    (session,) = load(cell / "ops_perf.csv").values()
    calls = [
        int(o["row"]["GLOBAL CALL COUNT"]) // 1024
        for o, (_, name, lay) in zip(session, label_session(session, LAYER_TYPES))
        if name == label and (layer is None or lay == layer)
    ]
    (f,) = [
        f
        for f in (cell / viz).glob(f"MatmulDeviceOperation_ID{calls[nth]}_traceID*.npeviz.zst")
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
    q48, ends_q = dram_reads(npeviz(A0, "qkv_proj (global)"))
    panels = [
        (per_core_rate(gate, 0, 20), ends, "gate_proj, 64 cores: first 20 us, all cores reading"),
        (per_core_rate_alive(gate, 32, 44), ends, "gate_proj: 32-44 us, after rows 7-9 finished"),
        (per_core_rate(gate2), ends, "gate_proj read in 16 blocks instead of 4"),
        (per_core_rate(down), ends_d, "down_proj, 32 cores"),
        (per_core_rate(q48, 0, 20), ends_q, "global-layer qkv, 48 cores: 0-20 us"),
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
- global-layer qkv (bottom middle): 48 cores in their first 20 us, before any of them has finished.
  The same row-dependent share as gate_proj: the lower rows get about twice the rate of the upper rows,
  about 384 GB/s in total.

Rates are issued bytes per second from a per-transfer trace of one profiled run. They are effective
rates (bytes issued over time, including each core's waits), not a direct measurement of DRAM
bandwidth. Tracing slows some
matmuls (down_proj by about 24%), so the absolute values here are lower than in untraced runs; the
pattern is what the figure shows.
"""
    save(fig, out, "fig15_dram_read_rate_grid", desc)


# tt-npe's per-link timeline values are raw demand in bytes/cycle (npeDeviceModelUtils.hpp:135, npeStats.cpp:322);
# only its grid-wide statistics are scaled to %. Its Blackhole model gives every link 60.9 B/cycle
# (device_models/blackhole.hpp:246). NOC*_IN / NOC*_OUT entries are endpoint (NIU) demand, not links.
NPE_LINK_BW = 60.9
NPE_LINKS = {"NOC0_EAST": (0, 1), "NOC0_SOUTH": (1, 0), "NOC1_NORTH": (-1, 0), "NOC1_WEST": (0, -1)}


def fig16(out):
    d = npeviz(A0, "gate_proj (sliding)", 0)
    _, ends = dram_reads(d)
    ts = sorted(d["timestep_data"], key=lambda t: t["start_cycle"])
    n = len(ts)
    mean = defaultdict(float)  # (row, col, link) -> mean demand, % of the link's modelled bandwidth
    for t in ts:
        for _, r, c, kind, dem in t["link_demand"]:
            if kind in NPE_LINKS:
                mean[(r, c, kind)] += dem / NPE_LINK_BW * 100 / n
    # each link drawn on the router it leads INTO, so a link leaving a DRAM column shows on the core next to it
    busiest = {}
    for (r, c, kind), v in mean.items():
        dr, dc = NPE_LINKS[kind]
        rc = ((r + dr) % ROWS, (c + dc) % COLS)
        busiest[rc] = max(busiest.get(rc, 0.0), v)
    hot = max(mean, key=mean.get)
    over = sum(1 for v in mean.values() if v > 100)
    us = np.array([t["start_cycle"] / MHZ for t in ts])
    avg = np.array([t["avg_link_demand"] for t in ts])  # already % (grid average over every link)
    hot_t = np.array(
        [next((x[4] for x in t["link_demand"] if (x[1], x[2], x[3]) == hot), 0.0) / NPE_LINK_BW * 100 for t in ts]
    )
    k = 32  # ~0.76 us smoothing, so the lines are readable

    def smooth(y):
        return np.convolve(y, np.ones(k) / k, mode="same")

    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0])
    vmax = 200.0
    grid(
        ax,
        {rc: v for rc, v in busiest.items() if rc[1] not in DRAM_COLS},
        ORANGES,
        vmax,
        ends,
        "Busiest NoC link into each router, average over the matmul",
        fmt="{:.0f}",
        fs=7,
    )
    colorbar(fig, ax, ORANGES, vmax, "modelled link demand, % of the link's bandwidth")
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(us, smooth(hot_t), color=ORANGE, lw=2)
    ax2.plot(us, smooth(avg), color=BLUE, lw=2)
    ax2.axhline(100, color=AXIS, lw=1, ls=(0, (4, 3)))
    ax2.text(us[int(n * 0.02)], 103, "link fully used", ha="left", va="bottom", fontsize=9, color=MUTED)
    ax2.text(
        us[int(n * 0.25)],
        smooth(hot_t).max() + 8,  # above the line's peaks, not on them
        f"busiest link (row {hot[0]}, column {hot[1]} → {hot[1] + 1})",
        fontsize=10,
        color=INK,
        ha="left",
    )
    ax2.text(
        us[int(n * 0.25)], smooth(avg)[int(n * 0.25)] + 8, "average over all links", fontsize=10, color=INK, ha="left"
    )
    ax2.set_xlabel("time since the matmul started, us")
    ax2.set_ylabel("modelled demand, % of the link's bandwidth")
    ax2.set_ylim(0, 300)
    ax2.set_title("Average link versus the busiest link over time", fontsize=11)
    desc = f"""
What the NoC congestion model (tt-npe) says about the same gate_proj matmul (64 cores, weights read from
DRAM). The model replays every recorded transfer through the chip's network-on-chip in 32-cycle steps.
In each step, every transfer that is in flight adds the bandwidth it is trying to use to every link on
its route (routes go along the row first, then along the column). A link's "demand" is that sum, shown
as a percentage of the link's bandwidth in the model. Every core asks for the same number of bytes, but
a link next to a DRAM column, or on a row segment many routes share, carries the data of every core
routed through it, so its demand is many times that of a link at the edge of the grid. Above 100%, more
data wants the link than it can carry.

- Left: for every router, the busiest link leading into it, averaged over the matmul. "D" squares are
  the DRAM columns; the dark ones are the endpoints that served the reads.
- Right: over time, the demand on the single busiest link (row {hot[0]}, column {hot[1]} → {hot[1] + 1}: eastbound,
  leaving the DRAM column) against the average over all links.

The averaged figure most profiling summaries report (about 16% average demand and 11% average use, apart
from a brief spike when the last reads drain) says the network is nearly idle. The same model puts the
busiest link at {mean[hot]:.0f}% on average over the matmul, and {over} links in all above 100%. Every one of
them lies on the row of a DRAM endpoint, at most four columns east of it. So the network is not idle where the
DRAM data enters it. This is a model of the recorded transfers, not a measurement of the links.
"""
    save(fig, out, "fig16_npe_link_demand", desc)


def npe_link_means(d):
    """Mean modelled demand per link over the matmul (% of the link's bandwidth), the busiest link into each
    router, the time series, and the hottest link."""
    ts = sorted(d["timestep_data"], key=lambda t: t["start_cycle"])
    mean = defaultdict(float)
    for t in ts:
        for _, r, c, kind, dem in t["link_demand"]:
            if kind in NPE_LINKS:
                mean[(r, c, kind)] += dem / NPE_LINK_BW * 100 / len(ts)
    busiest = {}
    for (r, c, kind), v in mean.items():
        dr, dc = NPE_LINKS[kind]
        rc = ((r + dr) % ROWS, (c + dc) % COLS)
        busiest[rc] = max(busiest.get(rc, 0.0), v)
    hot = max(mean, key=mean.get)
    us = np.array([t["start_cycle"] / MHZ for t in ts])
    avg = np.array([t["avg_link_demand"] for t in ts])
    hot_t = np.array(
        [next((x[4] for x in t["link_demand"] if (x[1], x[2], x[3]) == hot), 0.0) / NPE_LINK_BW * 100 for t in ts]
    )
    return mean, busiest, hot, us, avg, hot_t


def npe_panel(fig, ax, busiest, ends, vmax, title):
    """Busiest modelled link into each router; links into a DRAM-column router are drawn on its "D" square."""
    grid(
        ax,
        {rc: v for rc, v in busiest.items() if rc[1] not in DRAM_COLS},
        ORANGES,
        vmax,
        ends,
        title,
        fmt="{:.0f}",
        fs=6.5,
    )
    for (r, c), v in busiest.items():
        if c in DRAM_COLS and v >= 5:
            # above grid()'s plain "D" label, which would otherwise show through next to this one
            ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color=ORANGES(min(v / vmax, 1.0)), lw=0, zorder=4))
            ax.text(
                c,
                r,
                f"D\n{v:.0f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=SURFACE if v / vmax > 0.45 else INK,
                zorder=5,
            )


T2 = CELLS.parent / "t2_noc"  # plan v2 T2 cells: W, W + fatk_dram_ring, W + dramshard


def fig18(out):
    """fat-K down_proj L0 under three deliveries on W: per-core read rate (top) and tt-npe link demand (bottom)."""
    runs = [
        ("DRAM interleaved, 1D multicast", T2 / "00_noc_B+logitsl1+gridwide+vc1"),
        ("DRAM interleaved, gather_in0 ring", T2 / "01_noc_B+logitsl1+gridwide+vc1+fatk_dram_ring"),
        # its 16 KB reads are recorded as 8,160 B (NoC trace size field); npe_sizefix.py restored them and re-ran tt-npe
        ("DRAM-sharded (8 bank workers)", T2 / "02_noc_B+logitsl1+gridwide+vc1+dramshard", "sizefix/fixed/npe_viz"),
    ]
    data = []
    for name, cell, *viz in runs:
        d = npeviz(cell, "down_proj (sliding)", 0, viz=viz[0] if viz else "npe_viz")
        reads, ends = dram_reads(d)
        mean, busiest, hot, *_ = npe_link_means(d)
        data.append((name, reads, ends, mean, busiest, hot))
    rate_max = max(max(per_core_rate(r).values()) for _, r, *_ in data)
    vmax_rate = float(np.ceil(rate_max / 10) * 10)
    vmax_npe = 250.0
    fig, axs = plt.subplots(2, 3, figsize=(18, 10.2))
    for i, (name, reads, ends, mean, busiest, hot) in enumerate(data):
        rate = per_core_rate(reads)
        total = sum(n for _, n, _ in reads) / (max(t for t, _, _ in reads) - min(t for t, _, _ in reads)) / 1e3
        grid(axs[0, i], rate, BLUES, vmax_rate, ends, f"{name}\n{len(rate)} reading cores, {total:.0f} GB/s issued")
        npe_panel(
            fig, axs[1, i], busiest, ends, vmax_npe, f"{name}\nbusiest link into each router; hottest {mean[hot]:.0f}%"
        )
    colorbar(
        fig,
        axs[0, :].tolist(),
        BLUES,
        vmax_rate,
        "DRAM read rate per reading core, GB/s (issued weight bytes per second)",
    )
    colorbar(fig, axs[1, :].tolist(), ORANGES, vmax_npe, "modelled link demand, % of the link's bandwidth")
    desc = """
One down-projection matmul of the drafter (8192 x 1024 weight, read from DRAM) under three ways of
delivering the weight, on the same chip configuration. Top row: each square is a compute core that reads
weights, labelled with how fast it pulled them from DRAM (GB/s, issued bytes over that core's own reading
time); "D" squares are the DRAM columns, the dark ones the endpoints that served the reads. Bottom row:
the network-on-chip congestion model's (tt-npe) busiest link into each router, averaged over the matmul,
as a percentage of the link's bandwidth; links into a DRAM-column router are drawn on its "D" square.

- Left, interleaved weight, activation multicast (the default): 32 cores each read one column of tiles,
  all at 8.4-8.6 GB/s. The multicast keeps every core within a couple of blocks of the slowest one, so
  they all move at one pace.
- Middle, interleaved weight, activation passed around a ring: the same 32 readers, no longer held
  together. Their rates spread from 9.2 to 17.5 GB/s by column, the column reading the first DRAM bank
  being the slowest.
- Right, weight sharded across the 8 DRAM banks: 8 cores, one next to each bank, each reading its bank's
  contiguous slice in large bursts at 52-53 GB/s, with far less load on the busiest links.

The trace records any transfer above 8,160 bytes as 8,160 bytes; the sharded reads (16 KB bursts) were
restored to their true size before the model was run on them. Rates are issued bytes per second from a
per-transfer trace of one profiled run (profiling slows the matmuls), not a direct measurement of DRAM
bandwidth; the link demand is a model of the recorded transfers, not a measurement.
"""
    save(fig, out, "fig18_fatk_delivery_read_rate_and_links", desc)


def fig19(out):
    """thin-K gate_proj L0: tt-npe busiest-link grids at 64 cores, 86 cores, and 86 cores + VC by row (W)."""
    cells = CELLS.parent / "phase2_vc"
    runs = [
        ("64 cores", cells / "08_noc_B+logitsl1"),
        ("86 cores", cells / "10_noc_B+logitsl1+gridwide"),
        ("86 cores, VC by row", cells / "11_noc_B+logitsl1+gridwide+vc1"),
    ]
    data = []
    for name, cell in runs:
        d = npeviz(cell, "gate_proj (sliding)", 0)
        _, ends = dram_reads(d)
        data.append((name, ends) + npe_link_means(d))
    k = 32

    def smooth(y):
        return np.convolve(y, np.ones(k) / k, mode="same")

    vmax = 250.0
    fig = plt.figure(figsize=(23, 7.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.0], wspace=0.25)
    for i, (name, ends, mean, busiest, hot, us, avg, hot_t) in enumerate(data):
        ax = fig.add_subplot(gs[i])
        over = sum(1 for v in mean.values() if v > 100)
        npe_panel(fig, ax, busiest, ends, vmax, f"{name}\nbusiest link {mean[hot]:.0f}%, {over} links above 100%")
        colorbar(fig, ax, ORANGES, vmax, "modelled link demand, % of the link's bandwidth")
    ax4 = fig.add_subplot(gs[3])
    for (name, _, mean, _, hot, us, avg, hot_t), color in zip(data, (BLUE, ORANGE, INK2)):
        ax4.plot(us, smooth(hot_t), color=color, lw=2, label=f"{name}, busiest link")
        ax4.plot(us, smooth(avg), color=color, lw=1.2, ls=(0, (4, 2)), label=f"{name}, average over all links")
    ax4.axhline(100, color=AXIS, lw=1, ls=(0, (1, 2)))
    ax4.set_ylim(0, 320)
    ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5, frameon=False)
    ax4.set_xlabel("time since the matmul started, us")
    ax4.set_ylabel("modelled demand, % of the link's bandwidth")
    ax4.set_title("Busiest link (solid) and average over all links (dashed)", fontsize=11)
    (_, _, m0, _, h0, *_), (_, _, m1, _, h1, *_), (_, _, m2, _, h2, *_) = data
    desc = f"""
The congestion model's view (tt-npe) of one gate-projection matmul of the drafter (1024 x 8192 weight,
read from DRAM) on three grid layouts: the default 64 cores, the same weight spread over 86 cores, and
86 cores with each row's read requests on its own virtual channel. Each map shows, for every router,
the busiest link leading into it, averaged over the matmul, as a percentage of the link's bandwidth in
the model; links into a DRAM-column router are drawn on its "D" square. All three maps use one scale.

- 64 cores: the hottest link is {m0[h0]:.0f}% on average, {sum(1 for v in m0.values() if v > 100)} links above 100%.
- 86 cores: {m1[h1]:.0f}%, {sum(1 for v in m1.values() if v > 100)} links above 100%. The extra cores in the right-hand columns
  add traffic on new links, and the hottest link stays where it was.
- 86 cores with a virtual channel per row: {m2[h2]:.0f}%, {sum(1 for v in m2.values() if v > 100)} links above 100%.

Measured on the chip, the 86-core layout pulls weights faster than the 64-core one even though the
model's hottest link does not get any less busy. This is a model of the recorded transfers, not a
measurement of the links.
"""
    save(fig, out, "fig19_thin_k_grid_width_link_demand", desc)


def fig16b(out):
    """fig16 for the NoC-swapped run: NoC0 (default) and NoC1 (in1 reader swapped) side by side, one scale."""
    cells = CELLS.parent / "phase2_nocswap"
    runs = [("NoC0 (default)", cells / "04_noc_B+logitsl1"), ("NoC1 (swapped)", cells / "05_noc_B+logitsl1+nocswap")]
    data = []
    for name, cell in runs:
        d = npeviz(cell, "gate_proj (sliding)", 0)
        _, ends = dram_reads(d)
        data.append((name, ends) + npe_link_means(d))
    k = 32

    def smooth(y):
        return np.convolve(y, np.ones(k) / k, mode="same")

    def hot_label(hot):
        r, c, kind = hot
        dr, dc = NPE_LINKS[kind]
        return f"row {r}, column {c} → row {r + dr}, column {c + dc}"

    vmax = 400.0
    fig = plt.figure(figsize=(18, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95])
    for i, (name, ends, mean, busiest, hot, us, avg, hot_t) in enumerate(data):
        ax = fig.add_subplot(gs[i])
        grid(
            ax,
            {rc: v for rc, v in busiest.items() if rc[1] not in DRAM_COLS},
            ORANGES,
            vmax,
            ends,
            f"{name}: busiest link into each router",
            fmt="{:.0f}",
            fs=6.5,
        )
        # links that lead INTO a DRAM-column router (NoC1's northbound column links) are drawn on that router
        for (r, c), v in busiest.items():
            if c in DRAM_COLS and v >= 5:
                ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color=ORANGES(min(v / vmax, 1.0)), lw=0))
                ax.text(
                    c,
                    r,
                    f"D\n{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=SURFACE if v / vmax > 0.45 else INK,
                )
        colorbar(fig, ax, ORANGES, vmax, "modelled link demand, % of the link's bandwidth")
    ax3 = fig.add_subplot(gs[2])
    for (name, _, mean, _, hot, us, avg, hot_t), color in zip(data, (BLUE, ORANGE)):
        ax3.plot(us, smooth(hot_t), color=color, lw=2, label=f"{name}, busiest link")
        ax3.plot(us, smooth(avg), color=color, lw=1.3, ls=(0, (4, 2)), label=f"{name}, average over all links")
    ax3.axhline(100, color=AXIS, lw=1, ls=(0, (1, 2)))
    ax3.set_xlabel("time since the matmul started, us")
    ax3.set_ylabel("modelled demand, % of the link's bandwidth")
    ax3.set_ylim(0, 620)
    ax3.legend(loc="upper right", fontsize=9, frameon=False)
    ax3.set_title("Busiest link (solid) and average over all links (dashed)", fontsize=11)
    (n0, _, m0, _, h0, *_), (n1, _, m1, _, h1, *_) = data
    over0 = sum(1 for v in m0.values() if v > 100)
    over1 = sum(1 for v in m1.values() if v > 100)
    desc = f"""
The congestion model's view (tt-npe) of the same weight-read matmul (gate_proj of a sliding-window layer,
64 cores, weights read from DRAM) in two runs from one session: the default, where each core's weight
reads travel on network 0, and a run with the two networks swapped, so the weight reads travel on network 1.
Network 0 routes along the row first, then down the column; network 1 routes up the column first, then
along the row. The model replays every recorded transfer in 32-cycle steps; a link's demand is the sum of
the bandwidth every in-flight transfer routed through it tries to use, as a percentage of the link's
bandwidth in the model. Both maps use the same colour scale.

- Left, default: for every router, the busiest link leading into it, averaged over the matmul ("D" squares
  with a number: links leading into a DRAM-column router). The hot
  links leave the DRAM columns ("D"; dark = endpoints that served reads) eastwards along their rows.
  Busiest: {hot_label(h0)}, {m0[h0]:.0f}% on average; {over0} links above 100%.
- Middle, swapped: the responses now leave each bank northwards up its DRAM column, and the four banks
  of a column share that column's vertical links. Busiest: {hot_label(h1)}, {m1[h1]:.0f}% on average;
  {over1} links above 100%.
- Right: over time, the busiest link of each run (solid) and the average over all links (dashed).

With the networks swapped, every DRAM-fed matmul read its weights 30-44% slower in total (43-78% longer
matmuls). This is a model of the
recorded transfers, not a measurement of the links.
"""
    save(fig, out, "fig16b_npe_link_demand_noc_swap", desc)


def fig16b_split(out):
    """fig16b as two figures, one per run (NoC0 default, NoC1 swapped), each a map plus that run's busiest link
    and average over time; both keep fig16b's colour scale and y range so they compare directly."""
    cells = CELLS.parent / "phase2_nocswap"
    runs = [
        ("NoC0 (default)", cells / "04_noc_B+logitsl1", "fig16b1_npe_link_demand_noc0_default"),
        ("NoC1 (swapped)", cells / "05_noc_B+logitsl1+nocswap", "fig16b2_npe_link_demand_noc1_swapped"),
    ]
    k = 32

    def smooth(y):
        return np.convolve(y, np.ones(k) / k, mode="same")

    def hot_label(hot):
        r, c, kind = hot
        dr, dc = NPE_LINKS[kind]
        return f"row {r}, column {c} → row {r + dr}, column {c + dc}"

    for name, cell, stem in runs:
        d = npeviz(cell, "gate_proj (sliding)", 0)
        _, ends = dram_reads(d)
        mean, busiest, hot, us, avg, hot_t = npe_link_means(d)
        over = sum(1 for v in mean.values() if v > 100)
        fig = plt.figure(figsize=(15, 6.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
        ax = fig.add_subplot(gs[0])
        npe_panel(fig, ax, busiest, ends, 400.0, f"{name}: busiest link into each router")
        colorbar(fig, ax, ORANGES, 400.0, "modelled link demand, % of the link's bandwidth")
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(us, smooth(hot_t), color=ORANGE, lw=2, label=f"busiest link ({hot_label(hot)})")
        ax2.plot(us, smooth(avg), color=BLUE, lw=2, label="average over all links")
        ax2.axhline(100, color=AXIS, lw=1, ls=(0, (1, 2)))
        ax2.set_xlabel("time since the matmul started, us")
        ax2.set_ylabel("modelled demand, % of the link's bandwidth")
        ax2.set_ylim(0, 620)
        ax2.legend(loc="upper right", fontsize=9, frameon=False)
        ax2.set_title("Busiest link and average over all links", fontsize=11)
        if "swapped" in stem:
            body = f"""The weight reads travel on network 1, which routes up the column first, then along the row. The responses
now leave each bank northwards up its DRAM column, and the four banks of a column share that column's
vertical links ("D" squares with a number: links leading into a DRAM-column router).
Busiest: {hot_label(hot)}, {mean[hot]:.0f}% on average over the matmul; {over} links above 100%.
With the networks swapped, every DRAM-fed matmul read its weights 30-44% slower in total (43-78% longer
matmuls) than with the default (fig16b1)."""
        else:
            body = f"""The weight reads travel on network 0 (the default), which routes along the row first, then down the column.
The hot links leave the DRAM columns ("D"; dark = endpoints that served reads) eastwards along their rows.
Busiest: {hot_label(hot)}, {mean[hot]:.0f}% on average over the matmul; {over} links above 100%.
fig16b2 is the same matmul with the networks swapped, from the same session and on the same colour scale."""
        desc = f"""
The congestion model's view (tt-npe) of one weight-read matmul (gate_proj of a sliding-window layer, 64 cores,
weights read from DRAM), {name.split(" ")[0]} run. The model replays every recorded transfer in 32-cycle steps; a
link's demand is the sum of the bandwidth every in-flight transfer routed through it tries to use, as a
percentage of the link's bandwidth in the model.

- Left: for every router, the busiest link leading into it, averaged over the matmul.
- Right: over time, the busiest link (orange) and the average over all links (blue).

{body}

This is a model of the recorded transfers, not a measurement of the links.
"""
        save(fig, out, stem, desc)


def fig16c(out):
    """fig16 after the access-point swap: gate_proj L0 on 64 cores, default endpoints (fig16's cell A0) vs bank 0 read
    through (0, 0) (TT_MM1D_IN1_BANK0_XY; cell npe_ep0/00_noc_B+ep0, 2026-10-28), one scale. The two cells come from
    different sessions; A0 also has the logits in L1, which does not touch gate_proj."""
    runs = [("Default endpoints", A0), ("Access-point swap", CELLS.parent / "npe_ep0" / "00_noc_B+ep0")]
    data = []
    for name, cell in runs:
        d = npeviz(cell, "gate_proj (sliding)", 0)
        _, ends = dram_reads(d)
        data.append((name, ends) + npe_link_means(d))
    k = 32

    def smooth(y):
        return np.convolve(y, np.ones(k) / k, mode="same")

    def hot_label(hot):
        r, c, kind = hot
        dr, dc = NPE_LINKS[kind]
        return f"row {r}, column {c} → row {(r + dr) % ROWS}, column {(c + dc) % COLS}"

    vmax = 200.0
    fig = plt.figure(figsize=(18, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95])
    for i, (name, ends, mean, busiest, hot, us, avg, hot_t) in enumerate(data):
        ax = fig.add_subplot(gs[i])
        over = sum(1 for v in mean.values() if v > 100)
        npe_panel(fig, ax, busiest, ends, vmax, f"{name}\nbusiest link {mean[hot]:.0f}%, {over} links above 100%")
        colorbar(fig, ax, ORANGES, vmax, "modelled link demand, % of the link's bandwidth")
    ax3 = fig.add_subplot(gs[2])
    for (name, _, mean, _, hot, us, avg, hot_t), color in zip(data, (BLUE, ORANGE)):
        ax3.plot(us, smooth(hot_t), color=color, lw=2, label=f"{name}, busiest link")
        ax3.plot(us, smooth(avg), color=color, lw=1.3, ls=(0, (4, 2)), label=f"{name}, average over all links")
    ax3.axhline(100, color=AXIS, lw=1, ls=(0, (1, 2)))
    ax3.set_xlabel("time since the matmul started, us")
    ax3.set_ylabel("modelled demand, % of the link's bandwidth")
    ax3.set_ylim(0, max(smooth(x[7]).max() for x in data) * 1.08)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9, frameon=False)
    ax3.set_title("Busiest link (solid) and average over all links (dashed)", fontsize=11)
    (n0, _, m0, _, h0, us0, *_), (n1, _, m1, _, h1, us1, *_) = data
    over0 = sum(1 for v in m0.values() if v > 100)
    over1 = sum(1 for v in m1.values() if v > 100)
    desc = f"""
The congestion model's view (tt-npe) of the same weight-read matmul as fig16 (gate_proj of a sliding-window
layer, 64 cores, weights read from DRAM) in two runs: fig16's default run, and a later run with
DRAM bank 0 read through its unused endpoint in the top-left corner instead of its default endpoint in the
bottom row (the access-point swap; same bank, same data, same bytes). By default, bank 0's responses and most
of bank 4's leave through the same eastbound link in the bottom row; after the swap, bank 0's responses enter
the network in the top row instead. The model replays every recorded transfer in 32-cycle steps; a link's
demand is the sum of the bandwidth every in-flight transfer routed through it tries to use, as a percentage of
the link's bandwidth in the model. Both maps use the same colour scale.

- Left, default: busiest link {hot_label(h0)}, {m0[h0]:.0f}% on average over the matmul; {over0} links above 100%.
- Middle, access-point swap: busiest link {hot_label(h1)}, {m1[h1]:.0f}% on average; {over1} links above 100%.
- Right: over time, the busiest link of each run (solid) and the average over all links (dashed).
  The matmul's recorded transfers span {us0[-1]:.1f} us by default and {us1[-1]:.1f} us after the swap.

This is a model of the recorded transfers, not a measurement of the links.
"""
    save(fig, out, "fig16c_npe_link_demand_ap_swap", desc)


def fig17(out):
    a = GRIDWIDE / "06_noc_B+logitsl1"
    wide = GRIDWIDE / "07_noc_B+logitsl1+gridwide"
    wide16 = GRIDWIDE / "08_noc_B+logitsl1+gridwide+blkwide2"
    g86, ends = dram_reads(npeviz(wide, "gate_proj (sliding)", 0))
    g86b, _ = dram_reads(npeviz(wide16, "gate_proj (sliding)", 0))
    q96, ends_q = dram_reads(npeviz(wide, "qkv_proj (global)"))
    q96b, _ = dram_reads(npeviz(wide16, "qkv_proj (global)"))
    g64, ends64 = dram_reads(npeviz(a, "gate_proj (sliding)", 0))
    panels = [
        (per_core_rate(g86, 0, 12), ends, "gate_proj, 86 cores: 0-12 us"),
        (per_core_rate_alive(g86, 26, 36), ends, "gate_proj, 86 cores: 26-36 us"),
        (per_core_rate(g86b), ends, "gate_proj, 86 cores, 16 blocks"),
        (per_core_rate(q96, 0, 15), ends_q, "global-layer qkv, 96 cores: 0-15 us"),
        (per_core_rate(q96b), ends_q, "global-layer qkv, 96 cores, 16 blocks"),
        (per_core_rate(g64, 0, 20), ends64, "reference: gate_proj, 64 cores: 0-20 us"),
    ]
    vmax = 12.0
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.6))
    for ax, (vals, eps, title) in zip(axs.flat, panels):
        grid(ax, vals, BLUES, vmax, eps, title)
    colorbar(fig, axs.ravel().tolist(), BLUES, vmax, "DRAM read rate per core, GB/s (issued weight bytes per second)")
    desc = """
The same per-core map as the previous figure, for the backbone matmuls spread over more cores: the
gate / up / qkv weights of the sliding-window layers on 86 cores instead of 64, and the global layer's
qkv on 96 instead of 48. Each square is one compute core, labelled with how fast it pulled its weights
from DRAM (GB/s); "D" squares are the DRAM columns, the dark ones the endpoints that served the reads.

- gate_proj on 86 cores, first 12 us (top left), before any core has finished: the extra cores in
  columns 13-15 read alongside the others, and together the cores pull 405 GB/s, against 380 on 64
  cores in their first 20 us (bottom right). The lower rows still get about twice the share of the
  upper rows.
- 26-36 us (top middle): by then only rows 2-5 are still reading. As on 64 cores, these upper-row cores
  speed up once the lower rows have finished, so their low rate is a share, not a cap.
- read in 16 blocks (top right): every core reads at about the same rate and they finish together; the
  end tail disappears.
- global-layer qkv on 96 cores (bottom left and middle): the same pattern (407 GB/s in total in its
  first 15 us), and the same fix.

On untraced runs this layout makes these matmuls about 15% faster, and about 18% with 16 blocks. With
both, they reach about 93% of the effective rate (weight bytes over matmul time) of the large
output-head matmul.

Rates are issued bytes per second from a per-transfer trace of one profiled run. They are effective
rates (bytes issued over time, including each core's waits), not a direct measurement of DRAM
bandwidth. Tracing slows the matmuls a little, so the absolute values are lower than untraced.
"""
    save(fig, out, "fig17_dram_read_rate_grid_wide", desc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--only", default="fig15,fig16,fig16b,fig16b_split,fig16c,fig17,fig18,fig19")
    args = p.parse_args()
    for name in args.only.split(","):
        globals()[name](args.out)


if __name__ == "__main__":
    main()
