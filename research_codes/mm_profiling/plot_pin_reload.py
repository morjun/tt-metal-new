#!/usr/bin/env python3
"""Standalone drafter, 3-step draft time: whole lm_head in DRAM vs 38,720 pinned cols vs 59,840 pinned cols +
DRAM reload, plain (Tensix) and hidden (paced DRISC push) (fig13).

Reads generated/lmhead_l1_profile/standalone_fig13_hidden/results.json, written by
run_pin_reload_standalone.py --arms A,D,R59840_64_8_rope,R59840_64_8_rope_drisc60c2 (one paired campaign).
Style, palette and the public-figure checks come from plot_contributions.py.

    python research_codes/mm_profiling/plot_pin_reload.py [--out DIR]
"""

import argparse
import json
import statistics
from pathlib import Path

from plot_contributions import BAR, BLUE, BLUE_L, DEFAULT_OUT, DRAFT, GRID, INK, VAL, plt, save, sg

DATA = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/standalone_fig13_hidden/results.json"
BLUE_D = "#174f91"  # a darker step of the pin's blue: the reload hidden
ARMS = [
    ("A", "Whole lm_head\nin DRAM", GRID),
    ("D", "38,720 columns\npinned in SRAM", BLUE_L),
    ("R59840_64_8_rope", "59,840 columns\npinned in SRAM\n+ reload from DRAM", BLUE),
    ("R59840_64_8_rope_drisc60c2", "59,840 columns\npinned in SRAM\n+ reload from DRAM,\nhidden", BLUE_D),
]
TENSIX, HIDDEN = ARMS[2][0], ARMS[3][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    rows = json.loads(DATA.read_text())
    runs = {a: [r["fused_ms_per_iter"] for r in rows if r["arm"] == a] for a, _, _ in ARMS}
    med = {a: statistics.median(v) for a, v in runs.items()}
    pct = {a: 100 * (med[a] / med["A"] - 1) for a in med}

    fig, ax = plt.subplots(figsize=(11, 5.8))
    best = ARMS[-1][0]
    for x, (a, label, color) in enumerate(ARMS):
        ax.bar(x, med[a], 0.58, color=color, **BAR)
        ax.scatter([x] * len(runs[a]), runs[a], s=22, color=INK, alpha=0.55, zorder=3, linewidth=0)
        txt = f"{med[a]:.2f} ms" if a == "A" else f"{med[a]:.2f} ms\n{sg(pct[a], 1)}%"
        ax.text(x, max(runs[a]) + 0.15, txt, ha="center", va="bottom", **VAL)
    ax.set_xticks(range(len(ARMS)), [label for _, label, _ in ARMS])
    ax.set_ylim(0, max(med.values()) * 1.2)
    ax.set_ylabel("ms per 3-step draft (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"A larger pin with a hidden DRAM reload makes the draft {abs(pct[best]):.1f}% faster")
    spread = max(max(abs(x - med[a]) for x in v) for a, v in runs.items())
    save(
        fig,
        args.out,
        "fig13_lm_head_pin_reload_drafter",
        f"""
Drafter alone: time of one 3-step draft with the drafter's lm_head in DRAM, partly pinned in
on-chip SRAM, and pinned further with a reload from DRAM, plain and hidden.

The lm_head (1024 x 262,144, 512 MiB in bf16) is split along its columns: a leading slice is
pinned in L1 width-sharded on all 110 cores, the rest stays in DRAM, and the two partial logits are
joined before the argmax. Bit-exact against the unsplit head.

- Whole lm_head in DRAM: {med['A']:.3f} ms per draft.
- 38,720 columns pinned (14.8% of the head): the most that fits below the highest on-chip
  scratch-buffer use of any other op (the global-attention layer's), so nothing ever overwrites
  it. {med['D']:.3f} ms ({sg(pct['D'], 2)}%).
- 59,840 columns pinned (22.8%), reload from DRAM: the slice now overlaps the scratch buffers of
  the global-attention layer and its rotary embedding. The slice is released around those ops
  (host bookkeeping only), and the overwritten part, on the same 32 cores every step, is copied
  back from a DRAM snapshot by the compute cores before each lm_head. {med[TENSIX]:.3f} ms
  ({sg(pct[TENSIX], 2)}%), the copy-back included. From 63,360 columns the sliding-window
  attention layers overwrite all 110 cores before every lm_head, so a larger pin cannot gain.
- The same pin with the reload hidden: the DRAM controllers' own small cores push the overwritten
  part back while the attention output projection and the MLP of that layer run, at a capped rate
  (60 GB/s in small pieces) so those operations are not slowed. The compute cores only start the
  copy and check it is done before the lm_head. {med[HIDDEN]:.3f} ms ({sg(pct[HIDDEN], 2)}%), that
  is {1e3 * (med[TENSIX] - med[HIDDEN]):.0f} us per draft faster than the plain reload.

The 59,840 size is for the drafter alone: in the full speculative loop the target model's own
operations also reach into it, which was not measured. In every pinned arm the DRAM tail's matmul
block width is 8 output tiles, its fastest measured value. Bars are medians of 3 counterbalanced
rounds, one fresh process each, all arms in one paired campaign (dots; each arm's rounds within
{spread:.3f} ms of its median).

{DRAFT}
""",
    )


if __name__ == "__main__":
    main()
