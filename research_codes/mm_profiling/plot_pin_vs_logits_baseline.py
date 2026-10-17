#!/usr/bin/env python3
"""Standalone drafter, 3-step draft time: the best DRAM-head configuration (whole lm_head in DRAM, best block
width, logits kept in L1) vs the current best (59,840 pinned cols + hidden DRAM reload + logits in L1) (fig14).

Reads generated/lmhead_l1_profile/standalone_fig14/results.json, written by
run_pin_reload_standalone.py --arms B+logitsl1,R59840_64_8_rope_drisc60c2+logitsl1 (one paired campaign).
Style, palette and the public-figure checks come from plot_contributions.py.

    python research_codes/mm_profiling/plot_pin_vs_logits_baseline.py [--out DIR]
"""

import argparse
import json
import statistics
from pathlib import Path

from plot_contributions import BAR, DEFAULT_OUT, DRAFT, GRID, INK, VAL, plt, save, sg

DATA = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile/standalone_fig14/results.json"
BLUE_DD = "#0b2e5c"  # fig13's darkest step: the current best
BASE, BEST = "B+logitsl1", "R59840_64_8_rope_drisc60c2+logitsl1"
ARMS = [
    (BASE, "Whole lm_head in DRAM\n(best block width,\nlogits kept in SRAM)", GRID),
    (BEST, "59,840 columns pinned in SRAM\n+ hidden reload from DRAM\n(logits kept in SRAM)", BLUE_DD),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    rows = json.loads(DATA.read_text())
    runs = {a: [r["fused_ms_per_iter"] for r in rows if r["arm"] == a] for a, _, _ in ARMS}
    med = {a: statistics.median(v) for a, v in runs.items()}
    pct = 100 * (med[BEST] / med[BASE] - 1)
    paired = [1e3 * (b - a) for a, b in zip(runs[BASE], runs[BEST])]
    spread = max(max(abs(x - med[a]) for x in v) for a, v in runs.items())

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for x, (a, label, color) in enumerate(ARMS):
        ax.bar(x, med[a], 0.55, color=color, **BAR)
        ax.scatter([x] * len(runs[a]), runs[a], s=22, color=INK, alpha=0.55, zorder=3, linewidth=0)
        txt = f"{med[a]:.2f} ms" if a == BASE else f"{med[a]:.2f} ms\n{sg(pct, 1)}%"
        ax.text(x, max(runs[a]) + 0.15, txt, ha="center", va="bottom", **VAL)
    ax.set_xticks(range(len(ARMS)), [label for _, label, _ in ARMS])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, max(med.values()) * 1.2)
    ax.set_ylabel("ms per 3-step draft (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Against the best DRAM-head setup, the pin is {abs(pct):.1f}% faster")
    save(
        fig,
        args.out,
        "fig14_lm_head_pin_vs_dram_baseline",
        f"""
Drafter alone: time of one 3-step draft, the best configuration with the whole lm_head in DRAM against
the current best with part of it pinned in on-chip SRAM. Both arms carry every optimization that does
not depend on pinning, so the difference is the pin and its reload.

The lm_head is 1024 x 262,144 (512 MiB in bf16). In both arms its matmul writes the raw scores to on-chip
SRAM instead of DRAM, and only the single real row goes to DRAM before the argmax.

- Baseline, whole lm_head in DRAM: its matmul at its fastest measured block width (5 output tiles per
  pass). {med[BASE]:.3f} ms per draft.
- Current best: 59,840 columns (22.8% of the head) pinned in SRAM, width-sharded on all 110 cores; the
  part that the global-attention layer and its rotary embedding overwrite (32 cores) is copied back from
  a DRAM snapshot by the DRAM controllers' own small cores while the next operations run, at a capped
  rate so they are not slowed. The DRAM remainder runs at its fastest block width (8). {med[BEST]:.3f} ms
  ({sg(pct, 2)}%, {abs(statistics.median(paired)):.0f} us per draft).

Same drafted token and final hidden state in every run. Bars are medians of 3 counterbalanced rounds,
one fresh process each, both arms in one paired campaign (dots; each arm's rounds within {spread:.3f} ms of
its median). The 59,840 size is for the drafter alone: in the full speculative loop the target model's
own operations also reach into it, which was not measured.

{DRAFT}
""",
    )


if __name__ == "__main__":
    main()
