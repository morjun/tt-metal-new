#!/usr/bin/env python3
"""11x10 Tensix SRAM maps of the Phase 4 layouts, one per arm (gemma4-12b record §9.4; fig25).

Same square layout as ``lm_head_reload_tensix_grid.py`` (each core = one absolute L1 address bar, address 0 at the
bottom, 1,572,864 at the top), but for the persistent weights only, while every weight is intact (at the lm_head).
Every buffer is read from the arm's standalone log:
  - head slice A: ``[lm_head-split] pinning <cols> ... per_core_N=<n>`` and ``head at S=<addr>``, on all 110 cores;
  - head slice B: ``slice B: cols [a, b) per-core on [x-y - x-y] (<n> cores, <t> tiles/core) at <addr>``;
  - slice-B reservation (empty): ``slice-B reservation: <t> tiles/core per-core on <n> cores ... at <addr>``;
  - per-core pins: ``[placement] per-core pin [1, 1, K, N] at <addr> on <n> cores``, named by the weight-cache load
    just before it (row-major prefix of <n> cores from row 0, or from the arm's ``y<row>`` origin);
  - LOCKSTEP pins (no per-core line): the ``largest tensors pinned to L1`` list in allocation order, top-down from
    the top of L1 on the row-major prefix, the same byte range held empty on every other core; the lowest one is
    checked against the reload's ``extra buffer ... lockstep at [<addr>]`` line.
The hatched outline marks [S, reload end) on the global SDPA's 32 cores: overwritten by its CBs every drafter step
and copied back from DRAM by the reload. Program CBs and small L1 activations (logits) come and go per op and are
not drawn. Per core: weight bytes / allocatable bytes (above the fixed reservation).

    python research_codes/mm_profiling/plot_p4_sram_maps.py <out dir> [<figure path stem>]
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

GX, GY = 11, 10
TOP, BASE = 1572864, 111488  # L1 top and the lowest allocatable address (standalone_reload_59840_capture report)
TILE = 65536  # one 32-column head / pin tile at K = 1024, bf16
SDPA = [(x, y) for y in range(4) for x in range(8)]
ROOT = Path("generated/lmhead_l1_profile")
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
KIND = {  # hue = weight type, lightness = variant
    "head slice A": "#b44bb6",
    "head slice B": "#e6b8e7",
    "qkv L0": "#0072b2",
    "gate L0": "#009e73",
    "up L0": "#e69f00",
    "up L2": "#a34a00",
}
RESERVED = "#999999"

# (id, label, log): the attribution arms a-e (standalone_p4_attr) and the S5 layouts (standalone_p4_s5)
ARMS = [
    ("a", "a. F0: head 17 tiles", "standalone_p4_attr/00_R59840_66_6_rope_drisc60c2+logitsl1+gridwide+vc1+ep0.log"),
    ("b", "b. head 14 tiles", "standalone_p4_attr/01_R49280_66_6_rope_drisc60c2+logitsl1+gridwide+vc1+ep0.log"),
    (
        "c",
        "c. head 14 tiles, other tail",
        "standalone_p4_attr/02_R49280_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0.log",
    ),
    (
        "d",
        "d. + up L0 pin, slice-B room empty",
        "standalone_p4_attr/03_R49280_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinup_proj@0p3c+resB3+hybrid.log",
    ),
    (
        "e",
        "e. H1: + slice B",
        "standalone_p4_attr/04_R49280_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinup_proj@0p3c+sliceB3+hybrid.log",
    ),
    (
        "L4",
        "L4: 4 pins, LOCKSTEP",
        "standalone_p4_s5/01_R17600_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinwqkv@0p3+pingate_proj@0p3+pinup_proj@0p3+pinup_proj@2p3.log",
    ),
    (
        "H4",
        "H4: 4 pins + slice B, per-core",
        "standalone_p4_s5/03_R17600_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinwqkv@0p3c+pingate_proj@0p3c+pinup_proj@0p3c+pinup_proj@2p3c+sliceB12+hybrid.log",
    ),
    (
        "R4",
        "R4: 4 pins on rows 4-9, slice B on rows 0-3",
        "standalone_p4_s5/04_R3520_75_5_rope_drisc60c2+logitsl1+gridwide+vc1+ep0+pinwqkv@0p4cy4+pingate_proj@0p4cy4+pinup_proj@0p4cy4+pinup_proj@2p4cy4+sliceB16top4+hybrid.log",
    ),
]


def weight_name(path):
    m = re.search(r"layer_(\d+)/(?:self_attn|mlp)/(wqkv|gate_proj|up_proj)", path)
    return {"wqkv": "qkv", "gate_proj": "gate", "up_proj": "up"}[m.group(2)] + f" L{m.group(1)}"


def prefix(n, row=0):
    return [(x, y) for y in range(row, GY) for x in range(GX)][:n]


def rect(s):
    x0, y0, x1, y1 = map(int, re.match(r"\[(\d+)-(\d+) - (\d+)-(\d+)\]", s).groups())
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def layout(log, arm):
    """Buffers of one arm: dicts kind, lo, hi, cores, pad (per-core bytes that are padding), empty."""
    text = log.read_text(errors="replace").splitlines()
    bufs, last_load, lockstep_names = [], None, []
    head = slice_a = reload_end = lockstep_low = None
    row = int(m.group(1)) if (m := re.search(r"p\dc?y(\d+)", arm)) else 0
    for i, line in enumerate(text):
        if "Loaded cache for" in line:
            last_load = line
        elif m := re.search(r"\[placement\] per-core pin \[1, 1, (\d+), (\d+)\] at (\d+) on (\d+) cores", line):
            k, n, addr, cores = map(int, m.groups())
            tiles = n // 32
            pcn = -(-tiles // cores)
            per = pcn * k // 32 * 2048
            on = prefix(cores, row)
            bufs.append({"kind": weight_name(last_load), "lo": addr, "hi": addr + per, "cores": on, "pad": {}})
            short = tiles - pcn * (cores - 1)
            if short < pcn:
                bufs[-1]["pad"][on[-1]] = (pcn - short) * k // 32 * 2048
        elif "largest tensors pinned to L1" in line and not lockstep_names:
            for nxt in text[i + 1 :]:
                if not nxt.startswith("    "):
                    break
                lockstep_names.append(weight_name(nxt))
        elif m := re.search(r"pinning (\d+)/262144 cols .*per_core_N=(\d+)", line):
            slice_a = int(m.group(2)) * TILE
        elif m := re.search(r"head at S=(\d+) on 110 cores; reloads \[S, (\d+)\)", line):
            head, reload_end = int(m.group(1)), int(m.group(2))
        elif m := re.search(
            r"slice B: cols \[\d+, \d+\) per-core on (\[[^\]]+\]) \((\d+) cores, (\d+) tiles/core\) at (\d+)", line
        ):
            on, t, addr = rect(m.group(1)), int(m.group(3)), int(m.group(4))
            assert len(on) == int(m.group(2))
            bufs.append({"kind": "head slice B", "lo": addr, "hi": addr + t * TILE, "cores": on, "pad": {}})
        elif m := re.search(
            r"slice-B reservation: (\d+) tiles/core per-core on (\d+) cores outside the (\d+)-core prefix, at (\d+)",
            line,
        ):
            t, n, p, addr = map(int, m.groups())
            on = prefix(GX * GY)[p:]
            assert len(on) == n
            bufs.append({"kind": "empty", "lo": addr, "hi": addr + t * TILE, "cores": on, "pad": {}})
        elif m := re.search(r"extra buffer \S+ \[1, 1, 1024, 8192\]: lockstep at \[(\d+)\]", line):
            lockstep_low = int(m.group(1))
    if lockstep_names and not any(b["kind"] in lockstep_names for b in bufs):  # LOCKSTEP pins
        pcn = int(re.search(r"pin\w+@\dp(\d)\+", arm + "+").group(1))
        cores = -(-256 // pcn)
        per = pcn * TILE
        on = prefix(cores)
        for j, name in enumerate(lockstep_names):
            hi = TOP - j * per
            bufs.append(
                {"kind": name, "lo": hi - per, "hi": hi, "cores": on, "pad": {on[-1]: (pcn * cores - 256) * TILE}}
            )
            bufs.append({"kind": "empty", "lo": hi - per, "hi": hi, "cores": prefix(GX * GY)[cores:], "pad": {}})
        if lockstep_low is not None and lockstep_low != TOP - len(lockstep_names) * per:
            raise ValueError(
                f"{log.name}: lowest LOCKSTEP pin at {lockstep_low}, derived {TOP - len(lockstep_names) * per}"
            )
    bufs.append({"kind": "head slice A", "lo": head, "hi": head + slice_a, "cores": prefix(GX * GY), "pad": {}})
    for c in prefix(GX * GY):  # no two buffers overlap on a core
        spans = sorted((b["lo"], b["hi"]) for b in bufs if c in b["cores"])
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if b0 < a1:
                raise ValueError(f"{log.name}: overlap on {c}: {a0}-{a1} and {b0}-{b1}")
    return bufs, head, reload_end


def draw(ax, bufs, head, reload_end, labels=True, size=0.88):
    used = {}
    for x, y in prefix(GX * GY):

        def stripe(lo, hi, **style):
            if hi > lo:
                ax.add_patch(
                    Rectangle((x - size / 2, y + size / 2 - size * hi / TOP), size, size * (hi - lo) / TOP, **style)
                )

        ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, facecolor="white", edgecolor="none"))
        stripe(0, BASE, facecolor=RESERVED, edgecolor="none")
        used[(x, y)] = 0
        for b in bufs:
            if (x, y) not in b["cores"]:
                continue
            if b["kind"] == "empty":
                stripe(b["lo"], b["hi"], facecolor="white", edgecolor="#8a8a8a", hatch="////", linewidth=0)
                continue
            pad = b["pad"].get((x, y), 0)
            stripe(b["lo"] + pad, b["hi"], facecolor=KIND[b["kind"]], edgecolor="none")
            stripe(b["lo"], b["lo"] + pad, facecolor="white", edgecolor=KIND[b["kind"]], hatch="////", linewidth=0)
            used[(x, y)] += b["hi"] - b["lo"] - pad
        if (x, y) in SDPA:
            stripe(head, reload_end, facecolor="none", edgecolor=INK, hatch="\\\\\\\\", linewidth=0.6)
        ax.add_patch(
            Rectangle((x - size / 2, y - size / 2), size, size, fill=False, edgecolor="#303030", linewidth=0.5)
        )
        if labels:
            ax.text(
                x,
                y + size / 2 - size * (BASE + 40000) / TOP - 0.06,
                f"{100 * used[(x, y)] / (TOP - BASE):.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8},
            )
    ax.set_xlim(-0.55, GX - 0.45)
    ax.set_ylim(GY - 0.45, -0.55)
    ax.set_aspect("equal")
    ax.set_xticks(range(GX))
    ax.set_yticks(range(GY))
    ax.tick_params(length=0, labelsize=8, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    return used


def legend_handles(kinds):
    h = [Patch(facecolor=RESERVED, label="fixed reservation")]
    h += [Patch(facecolor=KIND[k], label=k) for k in KIND if k in kinds]
    h += [
        Patch(facecolor="white", edgecolor="#8a8a8a", hatch="////", label="held empty (slice-B room / LOCKSTEP)"),
        Patch(facecolor="white", edgecolor="#e69f00", hatch="////", label="pin padding (86th core)"),
        Patch(
            facecolor="none",
            edgecolor=INK,
            hatch="\\\\\\\\",
            label="overwritten by the global SDPA, reloaded from DRAM",
        ),
        Patch(facecolor="white", edgecolor="#777777", label="free (program CBs, activations)"),
    ]
    return h


def main(out, stem=None):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    arms, manifest = [], []
    for key, label, log in ARMS:
        arm = Path(log).stem.split("_", 1)[1]
        bufs, head, reload_end = layout(ROOT / log, arm)
        arms.append((key, label, bufs, head, reload_end))
        fig, ax = plt.subplots(figsize=(12, 11.8))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.12)
        used = draw(ax, bufs, head, reload_end)
        ax.set_title(label, fontsize=14, fontweight="bold", color=INK, loc="left")
        ax.set_xlabel("Tensix x", color=INK2)
        ax.set_ylabel("Tensix y", color=INK2)
        kinds = {b["kind"] for b in bufs}
        fig.legend(handles=legend_handles(kinds), loc="lower center", ncol=3, fontsize=9, frameon=False)
        for ext in ("png", "svg"):
            fig.savefig(out / f"{key}.{ext}", dpi=150)
        plt.close(fig)
        manifest.append(
            {
                "arm": key,
                "label": label,
                "runner_arm": arm,
                "log": log,
                "head_S": head,
                "reload_end": reload_end,
                "buffers": [
                    {
                        "kind": b["kind"],
                        "lo": b["lo"],
                        "hi": b["hi"],
                        "cores": len(b["cores"]),
                        "padding": sum(b["pad"].values()),
                    }
                    for b in bufs
                ],
                "weight_bytes_per_core": {f"{x},{y}": u for (x, y), u in used.items()},
                "weight_bytes_total": sum(used.values()),
            }
        )
        print(
            f"{key:3s} {label:45s} weights {sum(used.values()) / 1e6:7.2f} MB, per core {min(used.values()):,}-{max(used.values()):,} B"
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if stem:
        fig, axes = plt.subplots(2, 4, figsize=(20, 11.2))
        fig.subplots_adjust(left=0.02, right=0.995, top=0.96, bottom=0.11, wspace=0.06, hspace=0.12)
        for ax, (key, label, bufs, head, reload_end) in zip(axes.flat, arms):
            draw(ax, bufs, head, reload_end, labels=False)
            ax.set_title(label, fontsize=11.5, fontweight="bold", color=INK, loc="left")
        kinds = set(KIND)
        fig.legend(handles=legend_handles(kinds), loc="lower center", ncol=6, fontsize=9.5, frameon=False)
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
