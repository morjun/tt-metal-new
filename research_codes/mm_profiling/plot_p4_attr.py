#!/usr/bin/env python3
"""fig24: attribution of H1 vs F0 (gemma4-12b record §9.4).

Left: per-step change of each part of the drafter (traced time-mode profile, analyze_p4_attr.py), µs per 3-step
draft, with the standalone step (median of 3 rounds) as a marker. Middle: why the 3 head tiles' DRAM-tail time
(+162) is twice the up L0 pin's kernel saving (−80.5): a bridge in µs per draft. Right: the DRAM tail's time per
call above F0's per-tile rate, against its width (every vc1 + ep0 time-mode profile with that tail).

    python research_codes/mm_profiling/plot_p4_attr.py <figure path stem>
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_p4_attr import cell  # noqa: E402

ROOT = Path("generated/lmhead_l1_profile")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STEPS = [
    ("F0 → b: slice A 17 → 14 tiles\nper core (330 tiles to DRAM)", "01", "00"),
    ("b → c: other tail config\n(same columns)", "02", "01"),
    ("c → d: up L0 on the\n86 pin cores", "03", "02"),
    ("d → e = H1: slice B,\n3 tiles on the 24 other cores", "04", "03"),
]
PARTS = [  # (label, keys, color); hue = part, purple family = head in L1
    ("DRAM tail", ["head DRAM tail"], "#5d6b7e"),
    ("head slice A (L1)", ["head slice A"], "#b44bb6"),
    ("head slice B (L1)", ["head slice B"], "#e6b8e7"),
    ("up L0 matmul", ["up_proj L0"], "#e69f00"),
    ("other backbone matmuls", ["other backbone matmuls"], "#0072b2"),
    ("untilize, concat, argmax, reload ops", ["untilize", "concat", "argmax", "reload generic ops"], "#009e73"),
    ("gaps and other ops", ["op-to-op gaps", "other ops", "post_projection"], "#b8b7b1"),
]
TAIL_CELLS = [  # vc1 + ep0 time-mode profiles; (cell dir, tail config)
    ("per_op_ep0/08_time_R59840_66_6_rope+logitsl1+gridwide+vc1+ep0", "66 / ow6"),
    ("per_op_ep0/09_time_R59840_66_6_rope+logitsl1+gridwide+vc1+ep0", "66 / ow6"),
    ("opprof_p4_attr/00_time_R59840_66_6_rope_drisc60c2+logitsl1+gridwide+vc1+ep0", "66 / ow6"),
    ("opprof_p4_attr/01_time_R49280_66_6_rope_drisc60c2+logitsl1+gridwide+vc1+ep0", "66 / ow6"),
    ("opprof_p4_attr/02_time_R49280_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0", "75 / ow5"),
    (
        "opprof_p4_attr/03_time_R49280_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinup_proj@0p3c+resB3+hybrid",
        "75 / ow5",
    ),
    (
        "opprof_p4_attr/04_time_R49280_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinup_proj@0p3c+sliceB3+hybrid",
        "75 / ow5",
    ),
    (
        "opprof_p4_s5/00_time_R49280_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinup_proj@0p3c+sliceB3+hybrid",
        "75 / ow5",
    ),
    (
        "opprof_p4_s5/01_time_R17600_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinwqkv@0p3c+pingate_proj@0p3c+pinup_proj@0p3c+pinup_proj@2p3c+sliceB12+hybrid",
        "75 / ow5",
    ),
    (
        "opprof_p4_s5/02_time_R3520_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinwqkv@0p4cy4+pingate_proj@0p4cy4+pinup_proj@0p4cy4+pinup_proj@2p4cy4+sliceB16top4+hybrid",
        "75 / ow5",
    ),
    ("per_op_ep0/03_time_B+logitsl1+gridwide+vc1+ep0", "75 / ow5"),
    ("per_op_ep0/14_time_B+logitsl1+gridwide+vc1+ep0", "75 / ow5"),
    ("ep0_time/01_time_B+logitsl1+gridwide+vc1+ep0", "75 / ow5"),
    ("ep0_time/02_time_B+logitsl1+gridwide+vc1+ep0", "75 / ow5"),
]


def tail_time(path):
    """(column tiles, median µs per call) of the DRAM head tail (weight wider than 100,000 columns, in DRAM)."""
    durs, tiles = [], None
    for r in csv.DictReader(open(path / "ops_perf.csv")):
        if not r["OP CODE"].startswith("Matmul") or not (r.get("METAL TRACE REPLAY SESSION ID") or "").strip():
            continue
        x = r["INPUT_1_X_PAD[LOGICAL]"]
        n = int(x.split("[")[1].rstrip("]")) if "[" in x else 0
        if n > 100000 and "DRAM" in r["INPUT_1_MEMORY"]:
            tiles = n // 32
            durs.append(float(r["DEVICE KERNEL DURATION [ns]"]) / 1e3)
    return tiles, statistics.median(durs)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9.5)


def main(stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    prof = {
        d.name[:2]: cell(d)[0] for d in sorted((ROOT / "opprof_p4_attr").iterdir()) if (d / "ops_perf.csv").exists()
    }
    runs = json.loads((ROOT / "standalone_p4_attr/results.json").read_text())
    med = {}
    for r in runs:
        med.setdefault(r["arm"], []).append(r["fused_ms_per_iter"])
    arms = [d.name.split("_", 2)[2] for d in sorted((ROOT / "opprof_p4_attr").iterdir()) if d.is_dir()]
    stand = {f"{i:02d}": statistics.median(med[a]) * 1e3 for i, a in enumerate(arms)}

    fig = plt.figure(figsize=(23, 7.6))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.2, 0.85, 0.85], wspace=0.62, left=0.105, right=0.985, top=0.9, bottom=0.24
    )

    # left: per-step part changes
    ax = fig.add_subplot(gs[0])
    style(ax)
    ys = range(len(STEPS))
    for i, (label, cur, prev) in enumerate(STEPS):
        pos = neg = 0.0
        for name, keys, color in PARTS:
            d = sum(prof[cur][k] - prof[prev][k] for k in keys)
            left = pos if d >= 0 else neg + d
            ax.barh(i, abs(d), left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=1.2)
            if abs(d) >= 25:
                ax.text(
                    left + abs(d) / 2,
                    i,
                    f"{d:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if color not in ("#e6b8e7", "#b8b7b1", "#e69f00") else INK,
                )
            if d >= 0:
                pos += d
            else:
                neg += d
        net = stand[cur] - stand[prev]
        ax.plot([0, net], [i + 0.4, i + 0.4], color=INK, lw=1)
        ax.plot(net, i + 0.4, marker="D", ms=7, color=INK, markeredgecolor=SURFACE, zorder=5)
        ax.text(max(net, 0) + 6, i + 0.4, f"standalone {net:+.0f}", fontsize=9, color=INK, va="center")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(list(ys), [s[0] for s in STEPS], fontsize=10, color=INK)
    ax.set_ylim(len(STEPS) - 0.4, -0.5)
    ax.set_xlim(-130, 260)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("change per 3-token draft, µs (profiled op time)", color=INK2, fontsize=10)
    ax.set_title("Where each step's time goes", loc="left", fontsize=12.5, fontweight="bold", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PARTS]
    handles.append(Line2D([], [], marker="D", ls="", color=INK, ms=7))
    ax.legend(
        handles,
        [p[0] for p in PARTS] + ["measured step, standalone (median of 3 rounds)"],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.13),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    # middle: bridge from the pin's saving to the head tiles' tail time
    a, b, c, d = prof["00"], prof["01"], prof["02"], prof["03"]
    up_dram, up_l1 = c["up_proj L0"], d["up_proj L0"]
    tail_add = b["head DRAM tail"] - a["head DRAM tail"]
    f0_tiles = 202304 // 32
    per_up = up_dram / 3 / 256  # µs per column tile per call, up L0 from DRAM
    per_avg = a["head DRAM tail"] / 3 / f0_tiles  # F0 tail average
    per_marg = tail_add / 3 / 330  # tiles 15-17 x 110 cores
    rows = [
        ("up L0: direct value (DRAM-fed − L1-fed)", up_dram - up_l1, "total", "#e69f00"),
        ("+ up L0's L1-fed matmul time", up_l1, "step", "#e69f00"),
        ("= up L0's DRAM-fed time (256 tiles)", up_dram, "total", "#e69f00"),
        ("+ 74 more tiles moved (330 vs 256)", (330 - 256) * per_up * 3, "step", "#5d6b7e"),
        ("+ tail's average per tile is below up L0's", (per_avg - per_up) * 330 * 3, "step", "#5d6b7e"),
        ("+ step rate near F0 is below the average", (per_marg - per_avg) * 330 * 3, "step", "#5d6b7e"),
        ("= DRAM-tail time of the 330 head tiles", tail_add, "total", "#5d6b7e"),
    ]
    ax = fig.add_subplot(gs[1])
    style(ax)
    run = 0.0
    for i, (label, v, kind, color) in enumerate(rows):
        if kind == "total":
            ax.barh(i, v, color=color, height=0.6, edgecolor=SURFACE)
            run = v
            ax.text(v + 3, i, f"{v:.1f}", va="center", fontsize=9.5, color=INK, fontweight="bold")
        else:
            left = run if v >= 0 else run + v
            ax.barh(i, abs(v), left=left, color=color, alpha=0.45, height=0.6, edgecolor=color, linewidth=1)
            ax.text(max(run, run + v) + 3, i, f"{v:+.1f}", va="center", fontsize=9.5, color=INK)
            run += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0, 185)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("µs per 3-token draft (profiled op time)", color=INK2, fontsize=10)
    ax.set_title(
        "Why the 330 head tiles cost +162 but up L0 saves 80.5", loc="left", fontsize=12.5, fontweight="bold", color=INK
    )

    # right: tail time above F0's per-tile rate, vs width
    ax = fig.add_subplot(gs[2])
    style(ax)
    marks = {"66 / ow6": ("o", "#0072b2"), "75 / ow5": ("s", "#e69f00")}
    pts = {}
    for path, cfg in TAIL_CELLS:
        t, us = tail_time(ROOT / path)
        pts.setdefault((t, cfg), []).append(us)
    for (t, cfg), v in sorted(pts.items()):
        m, col = marks[cfg]
        ex = statistics.median(v) - per_avg * t
        ax.plot(t, ex, marker=m, ls="", ms=9, color=col, markeredgecolor=SURFACE, alpha=0.95)
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("DRAM-tail width, tiles (64 KiB = 32 columns each)", color=INK2, fontsize=10)
    ax.set_ylabel("µs per call above F0's per-tile rate", color=INK2, fontsize=10)
    ax.set_title("The tail's cost per tile rises with width", loc="left", fontsize=12.5, fontweight="bold", color=INK)
    for (t, cfg), label in {
        (6322, "66 / ow6"): "F0 (96 cores)",
        (6652, "66 / ow6"): "b (101 cores)",
        (8192, "75 / ow5"): "no L1 head (110 cores)",
    }.items():
        if (t, cfg) in pts:
            right = t > 8000
            ax.annotate(
                label,
                (t, statistics.median(pts[(t, cfg)]) - per_avg * t),
                xytext=(-8 if right else 8, -14 if right else 4),
                textcoords="offset points",
                ha="right" if right else "left",
                fontsize=8.5,
                color=INK2,
            )
    ax.legend(
        [Line2D([], [], marker=marks[k][0], ls="", color=marks[k][1], ms=8) for k in marks],
        [f"tail config {k} (per-core tiles / out_block_w)" for k in marks],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
        fontsize=9,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(
        f"up L0 DRAM {up_dram:.1f}, L1 {up_l1:.1f}; tail +{tail_add:.1f}; µs/tile/call up {per_up:.4f} avg {per_avg:.4f} marg {per_marg:.4f}"
    )
    print(
        f"GB/s: up {65536 / per_up / 1e3:.0f}, tail avg {65536 / per_avg / 1e3:.0f}, tail marg {65536 / per_marg / 1e3:.0f}"
    )
    for label, v, kind, _ in rows:
        print(f"  {label:50s} {v:7.1f}")
    for (t, cfg), v in sorted(pts.items()):
        print(
            f"  tail {t} {cfg}: {statistics.median(v):.1f} µs/call (n={len(v)}), above F0 rate {statistics.median(v) - per_avg * t:+.1f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
