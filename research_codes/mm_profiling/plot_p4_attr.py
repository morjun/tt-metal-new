#!/usr/bin/env python3
"""fig24: attribution of H1 vs F0 (gemma4-12b record §9.4).

Left: per-step change of each part of the drafter (traced time-mode profile, analyze_p4_attr.py), µs per 3-step
draft, with the standalone step (median of 3 rounds) as a marker. Middle: for the 330 head tiles (a vs b) and the up
L0 pin (c vs d), the DRAM read window the move to L1 removes (NIU samples, tail_slow_niu cells; run
item6/analyze_niu_instances.py on them first) beside the op time it saves (profiled D); record §9.2b, "Read windows
and residuals". Right: the DRAM tail's time per
call above F0's per-tile rate, against its width (every vc1 + ep0 time-mode profile with that tail).

    python research_codes/mm_profiling/plot_p4_attr.py <figures dir>

Writes three separate figures: fig24a_h1_step_parts (left), fig24b_head_vs_pin_saving (middle),
fig24c_tail_above_rate (right).
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


def main(outdir):
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

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(f"{outdir}/{name}.{ext}", dpi=150)
        plt.close(fig)

    # left: per-step part changes
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.9, bottom=0.27)
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

    save(fig, "fig24a_h1_step_parts")

    # middle: DRAM read window each weight removes (NIU, tail_slow_niu cells of arms a / b / c) against the op time
    # its move to L1 saves (profiled D); the gap is the residual P (record §9.2b, "Read windows and residuals")
    a, b, c, d = prof["00"], prof["01"], prof["02"], prof["03"]
    up_dram, up_l1 = c["up_proj L0"], d["up_proj L0"]
    tail_add = b["head DRAM tail"] - a["head DRAM tail"]
    f0_tiles = 202304 // 32
    per_avg = a["head DRAM tail"] / 3 / f0_tiles  # F0 tail average, µs per tile per call (right panel)
    head_vis = a["head slice A"] - b["head slice A"]  # slice A's per-tile time for the 3 tiles per core
    head_d = tail_add - head_vis
    up_d = up_dram - up_l1
    niu = {
        k: json.loads(next((ROOT / "tail_slow_niu").glob(f"{k}_*/niu_instances.json")).read_text())
        for k in ("00", "01", "02")
    }  # written by item6/analyze_niu_instances.py
    win = lambda k, op: 3 * niu[k][op]["window_us"]["mean"]  # noqa: E731  µs per draft
    r_head = win("01", "lm_head DRAM tail") - win("00", "lm_head DRAM tail")
    r_up = win("02", "up_proj (sliding) L0")
    rows = [  # (label, read window removed, op time saved)
        ("330 head tiles\n(3 per core on 110 cores)", r_head, head_d),
        ("up_proj layer 0\n(256 tiles, 3 per core on 86 cores)", r_up, up_d),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.27, right=0.96, top=0.87, bottom=0.2)
    style(ax)
    h = 0.34
    for i, (label, r, dd) in enumerate(rows):
        ax.barh(i - h / 2, r, height=h, color="#5d6b7e", edgecolor=SURFACE)
        ax.barh(i + h / 2, dd, height=h, color="#e69f00", edgecolor=SURFACE)
        ax.text(r + 2, i - h / 2, f"{r:.1f}", va="center", fontsize=10, color=INK)
        ax.text(dd + 2, i + h / 2, f"{dd:.1f}", va="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(
            max(r, dd) + 22,
            i,
            f"difference\n{r - dd:.1f}",
            va="center",
            fontsize=9.5,
            color=INK2,
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0, 215)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("µs per 3-token draft", color=INK2, fontsize=10)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#5d6b7e"),
            plt.Rectangle((0, 0), 1, 1, color="#e69f00"),
        ],
        labels=[
            "DRAM read time removed (hardware read counters: reader start to last weight byte)",
            "matmul time saved (time with the weight in DRAM − time with it in SRAM)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.4, -0.11),
        frameon=False,
        fontsize=9,
        ncol=1,
    )
    ax.set_title(
        "DRAM read time removed vs matmul time saved",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )

    save(fig, "fig24b_head_vs_pin_saving")

    # right: tail time above F0's per-tile rate, vs width
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.9, bottom=0.22)
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
    ax.set_title(
        "Tail time above F0's per-tile rate, by width", loc="left", fontsize=12.5, fontweight="bold", color=INK
    )
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
    save(fig, "fig24c_tail_above_rate")
    print(f"up L0 DRAM {up_dram:.1f}, L1 {up_l1:.1f}; tail +{tail_add:.1f}; slice A per-tile {head_vis:.1f}")
    for label, r, dd in rows:
        print(f"  {label.split(chr(10))[0]:16s} read window {r:6.1f}  saved {dd:6.1f}  difference {r - dd:5.1f}")
    for (t, cfg), v in sorted(pts.items()):
        print(
            f"  tail {t} {cfg}: {statistics.median(v):.1f} µs/call (n={len(v)}), above F0 rate {statistics.median(v) - per_avg * t:+.1f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
