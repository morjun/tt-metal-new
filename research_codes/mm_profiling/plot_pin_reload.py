#!/usr/bin/env python3
"""Standalone drafter, 3-step draft time: whole lm_head in DRAM vs 38,720 pinned cols vs 52,800 pinned
cols + DRAM reload (fig13).

Reads generated/lmhead_l1_profile/standalone_pin_reload/results.json, written by
run_pin_reload_standalone.py (arms A, D, R; arm B is not drawn). Style, palette and the public-figure
checks come from plot_contributions.py.

    python research_codes/mm_profiling/plot_pin_reload.py [--out DIR]
"""

import argparse
import json
import statistics
from pathlib import Path

from plot_contributions import BAR, BLUE, BLUE_L, DEFAULT_OUT, DRAFT, GRID, INK, VAL, plt, save, sg

DATA = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/standalone_pin_reload/results.json"
ARMS = [
    ("A", "Whole lm_head\nin DRAM", GRID),
    ("D", "38,720 columns\npinned in SRAM", BLUE_L),
    ("R", "52,800 columns\npinned in SRAM\n+ reload from DRAM", BLUE),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    rows = json.loads(DATA.read_text())
    runs = {a: [r["fused_ms_per_iter"] for r in rows if r["arm"] == a] for a, _, _ in ARMS}
    med = {a: statistics.median(v) for a, v in runs.items()}
    pct = {a: 100 * (med[a] / med["A"] - 1) for a in med}

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for x, (a, label, color) in enumerate(ARMS):
        ax.bar(x, med[a], 0.58, color=color, **BAR)
        ax.scatter([x] * len(runs[a]), runs[a], s=22, color=INK, alpha=0.55, zorder=3, linewidth=0)
        txt = f"{med[a]:.2f} ms" if a == "A" else f"{med[a]:.2f} ms\n{sg(pct[a], 1)}%"
        ax.text(x, max(runs[a]) + 0.15, txt, ha="center", va="bottom", **VAL)
    ax.set_xticks(range(len(ARMS)), [label for _, label, _ in ARMS])
    ax.set_ylim(0, max(med.values()) * 1.2)
    ax.set_ylabel("ms per 3-step draft (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"A larger pin with DRAM reload makes the draft {abs(pct['R']):.1f}% faster")
    save(
        fig,
        args.out,
        "fig13_lm_head_pin_reload_drafter",
        f"""
Drafter alone: time of one 3-step draft with the drafter's lm_head in DRAM, partly pinned in
on-chip SRAM, and pinned further with a reload from DRAM.

The lm_head (1024 x 262,144, 512 MiB in bf16) is split along its columns: a leading slice is
pinned in L1 width-sharded on all 110 cores, the rest stays in DRAM, and the two partial logits are
joined before the argmax. Bit-exact against the unsplit head.

- Whole lm_head in DRAM: {med['A']:.3f} ms per draft.
- 38,720 columns pinned (14.8% of the head): the most that fits below the highest on-chip
  scratch-buffer use of any other op (the global-attention layer's), so nothing ever overwrites
  it. {med['D']:.3f} ms ({sg(pct['D'], 2)}%).
- 52,800 columns pinned (20.1%): the slice now overlaps the global-attention layer's scratch
  buffers. The slice is released around that op (host bookkeeping only), and the overwritten part
  is copied back from a DRAM snapshot on the overwritten cores only, before each lm_head.
  {med['R']:.3f} ms ({sg(pct['R'], 2)}%), the copy-back included. Run alone, the drafter's
  attention overwrites 32 cores, so each draft step copies back 32 cores; in the full speculative
  loop, the target model's attention also overwrites all 110 once per iteration.

The DRAM tail's matmul block width is set per arm to its fastest measured value (8 and 4 output
tiles). Bars are medians of 3 counterbalanced rounds, one fresh process each (dots; each arm's
rounds within 0.004 ms). End to end, on the same prompt with identical output, the same three arms measure
99.225, 98.480 and 98.370 ms per decoding iteration (-0.71% and -0.86%).

{DRAFT}
""",
    )


if __name__ == "__main__":
    main()
