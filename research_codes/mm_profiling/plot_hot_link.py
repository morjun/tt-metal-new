#!/usr/bin/env python3
"""Figures for the row-11 hot link (gemma4-12b record §9.2e; lab meeting 2026-10-28).

fig20_dram_endpoints: the P150 NoC grid (physical NoC0 coordinates) with every DRAM endpoint by bank, marked as the
bank's NoC0 endpoint, its NoC1 endpoint or unused (tt_metal/soc_descriptors/blackhole_140_arch.yaml), the Tensix
compute grid, and the response link (0, 11) -> (1, 11) that banks 0 and 4 share.

fig21_hot_link_load: per drafter matmul layout, the static response load on the most loaded NoC0 link in units of
one bank's response bytes (H4 routes, noc_routes.py; columns assigned row-major, bank = column mod 8), split by the
bank it comes from, with the default endpoints and with bank 0 read through (0, 0).

    python research_codes/mm_profiling/plot_hot_link.py <figures dir>
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noc_routes import ENDPOINTS_NOC0, response_route  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b8b7b1"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BANK_ENDPOINTS = {  # blackhole_140_arch.yaml, list order per bank
    0: [(0, 0), (0, 1), (0, 11)],
    1: [(0, 2), (0, 10), (0, 3)],
    2: [(0, 9), (0, 4), (0, 8)],
    3: [(0, 5), (0, 7), (0, 6)],
    4: [(9, 0), (9, 1), (9, 11)],
    5: [(9, 2), (9, 10), (9, 3)],
    6: [(9, 9), (9, 4), (9, 8)],
    7: [(9, 5), (9, 7), (9, 6)],
}
NOC1 = {0: (0, 1), 1: (0, 10), 2: (0, 4), 3: (0, 7), 4: (9, 1), 5: (9, 10), 6: (9, 4), 7: (9, 7)}
COMPUTE_X = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]


def fig_endpoints(out):
    fig, ax = plt.subplots(figsize=(9.6, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y in range(12):
        for x in range(17):
            if x in COMPUTE_X and y >= 2:
                ax.add_patch(Rectangle((x, y), 0.94, 0.94, facecolor="#e9e8e4", edgecolor="none"))
    for b, eps in BANK_ENDPOINTS.items():
        for x, y in eps:
            if (x, y) == ENDPOINTS_NOC0[b]:
                fc, tc = BLUE, "white"
            elif (x, y) == NOC1[b]:
                fc, tc = ORANGE, "white"
            else:
                fc, tc = "white", INK
            ax.add_patch(
                Rectangle((x, y), 0.94, 0.94, facecolor=fc, edgecolor=GRAY if fc == "white" else "none", lw=1.2)
            )
            ax.text(x + 0.47, y + 0.5, f"B{b}", ha="center", va="center", fontsize=9, color=tc, weight="bold")
    ax.add_patch(Rectangle((-0.08, 10.92), 17.1, 1.1, fill=False, edgecolor=INK, lw=1.4, ls=(0, (4, 3))))
    ax.add_patch(
        FancyArrowPatch((0.94, 11.47), (2.0, 11.47), arrowstyle="-|>", mutation_scale=14, color=INK, lw=2.2, zorder=5)
    )
    ax.annotate(
        "shared response link (0, 11) → (1, 11)",
        xy=(1.5, 11.47),
        xytext=(2.6, 12.55),
        fontsize=9.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
    )
    ax.annotate(
        "unused bank-0 endpoint (0, 0):\nthe swap reads bank 0 here",
        xy=(0.94, 0.47),
        xytext=(2.6, -1.3),
        fontsize=9.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
    )
    ax.set_xlim(-0.3, 17.2)
    ax.set_ylim(13.3, -2.2)
    ax.set_xticks([x + 0.47 for x in range(17)], [str(x) for x in range(17)], fontsize=8.5, color=INK2)
    ax.set_yticks([y + 0.47 for y in range(12)], [str(y) for y in range(12)], fontsize=8.5, color=INK2)
    ax.set_xlabel("physical x (NoC0)", color=INK2)
    ax.set_ylabel("physical y (NoC0)", color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=BLUE),
        Rectangle((0, 0), 1, 1, facecolor=ORANGE),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRAY),
        Rectangle((0, 0), 1, 1, facecolor="#e9e8e4"),
    ]
    labels = ["bank's NoC0 endpoint", "bank's NoC1 endpoint", "unused endpoint", "Tensix compute core"]
    ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig20_dram_endpoints.{ext}", dpi=200, facecolor=SURFACE)
    (out / "fig20_dram_endpoints.txt").write_text(
        "Blackhole P150 NoC grid in physical NoC0 coordinates. The two DRAM columns (x 0 and x 9) hold three "
        "endpoints per bank (B0-B7). Each bank answers NoC0 reads from one endpoint (blue) and NoC1 reads from "
        "another (orange); the third is unused (white). Only row 11 holds two NoC0 endpoints, banks 0 and 4, so "
        "bank 0's responses and most of bank 4's leave through the same eastbound link (0, 11) -> (1, 11). "
        "The endpoint swap reads bank 0 through its unused endpoint (0, 0) instead.\n"
    )
    plt.close(fig)


def hot_contributions(Nt, pcn, gx, ncores, ep0):
    """{link: {bank: load}} in units of one bank's response bytes; columns row-major, bank = column mod 8."""
    per = defaultdict(float)
    for i in range(ncores):
        for c in range(i * pcn, min((i + 1) * pcn, Nt)):
            per[(i, c % 8)] += 1
    out = defaultdict(lambda: defaultdict(float))
    for (i, b), t in per.items():
        x, y = i % gx, i // gx
        core = (x + 1 if x < 6 else x + 5, y + 2)
        ep = (0, 0) if ep0 and b == 0 else ENDPOINTS_NOC0[b]
        for link in response_route(core, ep):
            out[link][b] += t / (Nt / 8)
    return out


def fig_hot_link(out):
    cfgs = [
        ("fat-K\n8 × 4, 32 cores", (32, 1, 8, 32)),
        ("thin-K\n8 × 8, 64 cores", (256, 4, 8, 64)),
        ("thin-K, wide\n86 cores", (256, 3, 11, 86)),
        ("lm_head\n110 cores", (8192, 75, 11, 110)),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rows = []
    for k, (name, cfg) in enumerate(cfgs):
        for j, ep0 in enumerate((False, True)):
            c = hot_contributions(*cfg, ep0)
            link = max(c, key=lambda lk: sum(c[lk].values()))
            b0, b4 = c[link].get(0, 0.0), c[link].get(4, 0.0)
            other = sum(c[link].values()) - b0 - b4
            yv = k * 2.6 + j * 1.0
            left = 0.0
            for val, col in ((b0, BLUE), (b4, ORANGE), (other, GRAY)):
                if val > 0:
                    ax.barh(yv, val - 0.02, left=left, height=0.78, color=col, edgecolor="none")
                    left += val
            ax.text(left + 0.04, yv, f"{left:.2f}", va="center", fontsize=9.5, color=INK)
            ax.text(
                -0.06,
                yv,
                "bank 0 via (0, 0)" if ep0 else "default endpoints",
                va="center",
                ha="right",
                fontsize=9,
                color=INK2,
            )
            rows.append((name.replace("\n", " "), ep0, left, b0, b4))
        ax.text(
            -0.36,
            k * 2.6 + 0.5,
            name,
            va="center",
            ha="right",
            fontsize=10,
            color=INK,
            weight="bold",
            transform=ax.get_yaxis_transform(),
        )
    ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(0, 2.35)
    ax.set_ylim(k * 2.6 + 1.6, -0.6)
    ax.set_yticks([])
    ax.set_xlabel("load on the most loaded NoC0 response link (in units of one bank's response bytes)", color=INK2)
    ax.tick_params(colors=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    handles = [Rectangle((0, 0), 1, 1, facecolor=col) for col in (BLUE, ORANGE)]
    ax.legend(handles, ["from bank 0", "from bank 4"], loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(left=0.40)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig21_hot_link_load.{ext}", dpi=200, facecolor=SURFACE)
    (out / "fig21_hot_link_load.txt").write_text(
        "Static response load on the most loaded NoC0 link for each drafter matmul layout, in units of one DRAM "
        "bank's response bytes (each bank holds 1/8 of a weight). With the default endpoints that link is "
        "(0, 11) -> (1, 11): it carries all of bank 0's responses plus the share of bank 4's that wraps west-to-east "
        "to the cores in columns x 1-6 (all of it for fat-K, whose bank-4 readers sit in x 5; 75% for the 8 x 8 "
        "thin-K grid; 56% for the 86-core thin-K grid; 55% for the lm_head). Reading bank 0 through (0, 0) leaves "
        "every link at most one bank's bytes. Routes: dimension order (east, then south) on the torus, validated "
        "against tt-npe; bytes per core from the matmul's column assignment.\n"
    )
    plt.close(fig)
    return rows


if __name__ == "__main__":
    out = Path(sys.argv[1])
    fig_endpoints(out)
    for r in fig_hot_link(out):
        print(r)
