#!/usr/bin/env python3
"""fig23_dram_tile_ep0_reload: why the DRISC lm_head reload and the access-point swap collide (gemma4-12b record §9.2e).

Bank 0's three DRAM tiles, each drawn as its two NoC ports (NIU0 / NIU1) and its RISC-V (DRISC), in four states:
default endpoints with the DRISC reload; the swap with the reload on both senders; the swap with (0, 0) left out
(the guard in lm_head_reload.py); and the proposed (0, 0) sender on NoC1 (not run). A port in bridge (NOC2AXI) mode
forwards reads to GDDR in hardware; a port in stream mode delivers every arriving packet to the RISC-V's own memory
(tt_metal/hw/inc/experimental/drisc_mode.h).

    python research_codes/mm_profiling/plot_drisc_ep0.py <figures dir>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

BLUE, ORANGE, GRAY, RED = "#2a78d6", "#eb6834", "#b8b7b1", "#c0392b"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BRIDGE_FC, STREAM_FC, FW_FC, IDLE_FC = "#dbe8f8", "#fbdccd", "#d9d8d4", "white"

TILES = [(0, 0), (0, 1), (0, 11)]
# per panel: title, footer, and per tile: (NoC0 port mode, NoC1 port mode, RISC-V state, port with reads, writes via)
# mode: "bridge" / "stream"; RISC-V: "fw" (system firmware), "reload", "reload2" (reload, double share), "idle";
# reads: None / "noc0" / "noc1"; writes via: None / "noc0" / "noc1"
PANELS = [
    (
        "Default endpoints + DRAM-core reload",
        "exact; bank 0 push 184 µs, hidden",
        {
            (0, 0): ("stream", "bridge", "reload", None, "noc0"),
            (0, 1): ("stream", "bridge", "reload", "noc1", "noc0"),
            (0, 11): ("bridge", "bridge", "fw", "noc0", None),
        },
    ),
    (
        "Access-point swap + reload on both senders",
        "wrong output: bank-0 reads return the RISC-V's memory",
        {
            (0, 0): ("stream", "bridge", "reload", "noc0", "noc0"),
            (0, 1): ("stream", "bridge", "reload", "noc1", "noc0"),
            (0, 11): ("bridge", "bridge", "fw", None, None),
        },
    ),
    (
        "Access-point swap, (0, 0) left out of the reload",
        "exact; (0, 1) pushes 2x the pages at the same pace: 344 µs, exposed",
        {
            (0, 0): ("bridge", "bridge", "idle", "noc0", None),
            (0, 1): ("stream", "bridge", "reload2", "noc1", "noc0"),
            (0, 11): ("bridge", "bridge", "fw", None, None),
        },
    ),
    (
        "Proposed: (0, 0) reload kernel on NoC1 (not run)",
        "both senders kept; (0, 0)'s NoC0 port stays a bridge",
        {
            (0, 0): ("bridge", "stream", "reload", "noc0", "noc1"),
            (0, 1): ("stream", "bridge", "reload", "noc1", "noc0"),
            (0, 11): ("bridge", "bridge", "fw", None, None),
        },
    ),
]
RISC_TEXT = {
    "fw": ("system\nfirmware", FW_FC, INK2),
    "reload": ("reload\nkernel", STREAM_FC, INK),
    "reload2": ("reload kernel\n2x pages", STREAM_FC, INK),
    "idle": ("idle", IDLE_FC, INK2),
}


def arrow(ax, a, b, color, ls="-", lw=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, linestyle=ls))


def panel(ax, title, footer, tiles):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.0)
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.add_patch(Rectangle((0.15, 0.9), 1.1, 9.4, facecolor="#efeee9", edgecolor=GRAY, lw=1))
    ax.text(0.7, 5.6, "bank 0 DRAM (GDDR)", rotation=90, ha="center", va="center", fontsize=9, color=INK)
    for i, t in enumerate(TILES):
        n0, n1, risc, reads, writes = tiles[t]
        y0 = 7.6 - i * 3.3  # tile bottom
        ax.add_patch(Rectangle((2.0, y0), 5.0, 2.6, facecolor="none", edgecolor=INK2, lw=1))
        ax.text(2.05, y0 + 2.75, f"tile {t}", fontsize=9, color=INK, va="bottom")
        for k, (mode, noc) in enumerate(((n0, "noc0"), (n1, "noc1"))):
            py = y0 + 1.35 - k * 1.2  # NoC0 port on top, NoC1 below
            hit = reads == noc
            ax.add_patch(
                Rectangle(
                    (2.15, py),
                    2.3,
                    1.1,
                    facecolor=BRIDGE_FC if mode == "bridge" else STREAM_FC,
                    edgecolor=INK if hit else GRAY,
                    lw=2.2 if hit else 0.8,
                )
            )
            label = f"NoC{noc[-1]} port: {mode}"
            if hit:
                label += f"\nNoC{noc[-1]} reads arrive"
            ax.text(3.3, py + 0.55, label, ha="center", va="center", fontsize=7.6, color=INK)
            if hit and mode == "bridge":
                arrow(ax, (2.15, py + 0.55), (1.27, py + 0.55), BLUE)
            if hit and mode == "stream":
                arrow(ax, (4.45, py + 0.55), (4.95, y0 + 1.3), RED, lw=2.2)
        txt, fc, tc = RISC_TEXT[risc]
        ax.add_patch(Rectangle((4.95, y0 + 0.15), 1.9, 2.3, facecolor=fc, edgecolor=GRAY, lw=0.8))
        ax.text(5.9, y0 + 1.3, "RISC-V\n" + txt, ha="center", va="center", fontsize=8, color=tc)
        if writes:
            arrow(ax, (6.85, y0 + 1.3), (8.2, y0 + 1.3), ORANGE, ls="--")
            ax.text(8.25, y0 + 1.3, f"head-pin\nwrites\n(NoC{writes[-1]})", fontsize=7.6, color=INK2, va="center")
    ax.text(0.15, 0.35, footer, fontsize=9, color=INK, va="center")


def main(out):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 11.0), facecolor=SURFACE)
    for ax, (title, footer, tiles) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        panel(ax, title, footer, tiles)
    handles = [
        Patch(facecolor=BRIDGE_FC, edgecolor=GRAY, label="port in bridge mode: reads forwarded to DRAM"),
        Patch(facecolor=STREAM_FC, edgecolor=GRAY, label="port in stream mode: arrivals land in the RISC-V's memory"),
        Patch(facecolor=FW_FC, edgecolor=GRAY, label="RISC-V running the system firmware (not usable by Metal)"),
        Patch(facecolor="white", edgecolor=INK, lw=2.2, label="port the matmul's reads are sent to"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08, wspace=0.06, hspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig23_dram_tile_ep0_reload.{ext}", dpi=200, facecolor=SURFACE)
    (out / "fig23_dram_tile_ep0_reload.txt").write_text(
        "DRAM bank 0 of a Blackhole P150 and its three DRAM tiles, (0, 0), (0, 1) and (0, 11). Each tile has two "
        "NoC ports, one per NoC, and one RISC-V core. A matmul's weight read is a packet sent to a port; in bridge "
        "mode (the hardware default) the port turns it into a DRAM read and sends the data back, with no RISC-V "
        "involved. The pinned lm_head is restored by a kernel on the RISC-Vs of (0, 0) and (0, 1) (and of two tiles "
        "of every other bank), which copies from DRAM and writes into the compute cores' L1. To send those writes, "
        "the kernel puts its own tile's NoC0 port into stream mode, where every arriving packet is delivered to the "
        "RISC-V's memory instead of DRAM. (0, 11)'s RISC-V runs the system firmware and is never used by the "
        "kernel; its NoC0 port normally serves the matmuls' NoC0 reads.\n"
        "- Top left, default: NoC0 reads go to (0, 11), NoC1 reads to (0, 1)'s NoC1 port; both are bridges, so the "
        "reload never meets a read. Exact, and the 184 us push hides behind other ops.\n"
        "- Top right, access-point swap: NoC0 reads of bank 0 go to (0, 0) to relieve the busy link next to (0, 11), "
        "but the reload has put (0, 0)'s NoC0 port in stream mode, so they read the RISC-V's memory: wrong output.\n"
        "- Bottom left, the fix used now: (0, 0) is left out of the reload, so its port stays a bridge. (0, 1) "
        "pushes all of bank 0's pages at the same pace and takes 344 us; the lm_head waits for it.\n"
        "- Bottom right, proposed, not run: (0, 0)'s kernel sends on NoC1, so only its NoC1 port, which no read "
        "uses, switches to stream mode.\n\n"
        "Setup: Gemma-4 12B (target) with its 12B-assistant drafter (4 layers, 423M parameters), speculative "
        "decoding with 3 draft tokens per iteration, bf16, one Tenstorrent Blackhole P150 chip.\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
