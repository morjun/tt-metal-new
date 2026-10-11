#!/usr/bin/env python3
"""11x10 Tensix SRAM maps for the DRAM-reload scheme (gemma4-12b MEASUREMENT_RECORD §5.5.14 Gate 5).

Same square layout as ``lm_head_l1_tensix_grid.py`` (each core = one absolute L1 address bar, low address
at the bottom), drawn from a capture taken with GEMMA4_LMHEAD_L1_RELOAD=1. The pinned head is followed
through the iteration, per core:

- **purple**: head bytes that hold the weights, whether the head tensor is allocated or freed inside a
  bracket (freeing is host bookkeeping only; L1 is untouched);
- **green**: head range covered by this op's program-local CBs (the op overwrites it now; it is reloaded
  from DRAM before the next drafter lm_head);
- **light green**: head range overwritten by an earlier op and not yet reloaded (stale until the reload);
- **hatched**: at a reload op, the range it copies back from the DRAM snapshot on its cores.

Utilization = union(program-local CBs, physical L1 tensor pages, intact head bytes) / allocatable L1. Intact
head bytes count whether the tensor is allocated or freed-but-preserved. Stale head bytes (light green) hold no
valid data and are excluded, as LOCKSTEP-only reservations are in lm_head_l1_tensix_grid.py; a reload op's
copied-back range is its output and counts. The manifest records the stale bytes per core separately.

The stale state at the start of the captured iteration is taken from its end (steady state: the previous
iteration's target window overwrote the same cores). Drafter step 0 only: head_and_other, layer_00, layer_03 (the global
layer) and the global-SDPA CB ceiling (plus any other bracketed op, e.g. the rotary embedding under
GEMMA4_LMHEAD_L1_RELOAD_ROPE). Works on the demo capture and on the standalone drafter capture
(test_lm_head_l1_address_capture, 72 matmuls).

    python research_codes/mm_profiling/lm_head_reload_tensix_grid.py generated/lmhead_l1_profile/demo_reload_52800
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from lm_head_l1_analyze import COLORS, call_identity, cores, intervals_union
from lm_head_l1_tensix_grid import GRID_X, GRID_Y, write_index

GRID = [(x, y) for y in range(GRID_Y) for x in range(GRID_X)]
OVERWRITE, STALE = "#2ca02c", "#a1d99b"
PAGE = 2048


def tensor_from_graph(p):
    return {
        "address": int(p["address"]),
        "size": int(p["max_size_per_bank"]),
        "num_cores": int(p.get("num_cores", 0)),
        "physical": json.loads(p["physical_ranges"]) if p.get("physical_ranges") else None,
    }


def walk(graph, before, head, head_size, ceiling, stale0):
    """Outermost device ops with the L1 state during each: live tensors, head allocated?, stale ranges."""
    base = before["base"]
    live = {}
    for block in before["blocks"]:
        if block.get("allocated") == "yes":
            address = base + int(block["address"])
            live[address] = {
                "address": address,
                "size": int(block["size"]),
                "num_cores": 110 if address == head else 0,
                "physical": before.get("physical_pages", {}).get(str(address)),
            }
    stale = dict(stale0)  # core -> end of the overwritten head range [head, end)
    reload_end = head + -(-(ceiling - head) // PAGE) * PAGE
    stack, current, ops, matmuls = [], None, [], 0
    for index, node in enumerate(graph):
        kind, p = node["node_type"], node.get("params", {})
        if kind == "function_start":
            name = p.get("name", "")
            stack.append(name)
            if current is None and name.endswith("DeviceOperation"):
                current = {"op": name, "start": index, "depth": len(stack), "matmul": matmuls, "cbs": []}
                if name == "GenericOpDeviceOperation":
                    args = " ".join(node.get("arguments", []))
                    if "dram_reload_pages.cpp" not in args:
                        raise ValueError(f"graph node {index}: a generic op that is not the reload")
                    current["reload_cores"] = sorted(cores(args))
        elif kind == "function_end":
            if stack:
                stack.pop()
            if current is not None and len(stack) < current["depth"]:
                head_live = head in live
                current["head_allocated"] = head_live
                current["tensors"] = [t for a, t in live.items() if a != head]
                current["stale_before"] = dict(stale)
                if not head_live:  # bracketed op: its CBs overwrite the freed head range
                    for cb in current["cbs"]:
                        if cb["address"] + cb["size"] > head:
                            for c in cb["cores"]:
                                stale[c] = max(stale.get(c, head), cb["address"] + cb["size"])
                elif any(cb["address"] + cb["size"] > head for cb in current["cbs"]):
                    raise ValueError(f"graph node {index}: CBs reach the allocated head")
                if "reload_cores" in current:
                    current["reload_range"] = (head, reload_end)
                    for c in current["reload_cores"]:
                        if stale.get(c, head) > reload_end:
                            raise ValueError(f"core {c}: stale range beyond the reload range")
                        stale.pop(c, None)
                current["end"] = index
                matmuls += current["op"] == "MatmulDeviceOperation"
                ops.append(current)
                current = None
        elif kind == "circular_buffer_allocate" and current is not None and p.get("globally_allocated") == "0":
            current["cbs"].append(
                {"address": int(p["address"]), "size": int(p["size"]), "cores": sorted(cores(p.get("core_range_set")))}
            )
        elif kind == "buffer_allocate" and p.get("type") == "L1":
            t = tensor_from_graph(p)
            if t["address"] == head and t["size"] != head_size:
                raise ValueError(f"graph node {index}: a different buffer at the head address")
            live[t["address"]] = t
        elif kind == "buffer_deallocate" and p.get("type") == "L1":
            live.pop(int(p["address"]), None)
    return ops, matmuls, stale


def subtract(span, cuts):
    lo, hi = span
    out = []
    for a, b in intervals_union([(max(lo, a), min(hi, b)) for a, b in cuts if b > lo and a < hi]):
        if a > lo:
            out.append((lo, a))
        lo = max(lo, b)
    if lo < hi:
        out.append((lo, hi))
    return out


def core_layers(op, core, head, top):
    x, y = core
    cbs = [(cb["address"], cb["address"] + cb["size"]) for cb in op["cbs"] if core in map(tuple, cb["cores"])]
    tensors = []
    for t in op["tensors"]:
        if t["physical"] is not None:
            phys = [tuple(r) for r in t["physical"].get(f"{x},{y}", [])]
        else:
            phys = [(t["address"], t["address"] + t["size"])] if t["num_cores"] == 110 else []
        tensors += phys
    h = (head, top)
    now = intervals_union([(max(a, head), b) for a, b in cbs if b > head])
    stale_end = op["stale_before"].get(core)
    stale = subtract((head, stale_end), now) if stale_end else []
    intact = subtract(h, now + stale)
    reload = [op["reload_range"]] if core in map(tuple, op.get("reload_cores", [])) else []
    used = intervals_union(cbs + tensors + intact + reload)
    return {
        "cbs": [(a, min(b, head)) for a, b in cbs if a < head],
        "tensors": tensors,
        "now": now,
        "stale": stale,
        "intact": intact,
        "reload": reload,
        "used_bytes": sum(b - a for a, b in used),
        "stale_bytes": sum(b - a for a, b in subtract((head, stale_end), now + reload)) if stale_end else 0,
    }


def render(op, title, base, top, head, cols, destination, pdf):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 12.3))
    fig.subplots_adjust(left=0.065, right=0.93, top=0.87, bottom=0.1)
    hh = 0.88

    def stripe(x, y, lo, hi, **style):
        if hi > lo:
            ax.add_patch(Rectangle((x - hh / 2, y + hh / 2 - hh * hi / top), hh, hh * (hi - lo) / top, **style))

    summary = {}
    for x, y in GRID:
        L = core_layers(op, (x, y), head, top)
        ax.add_patch(Rectangle((x - hh / 2, y - hh / 2), hh, hh, facecolor="white", edgecolor="none"))
        stripe(x, y, 0, base, facecolor=COLORS["reserved"], edgecolor="none")
        for lo, hi in L["tensors"]:
            stripe(x, y, lo, hi, facecolor=COLORS["tensor"], edgecolor="none")
        for lo, hi in L["intact"]:
            stripe(x, y, lo, hi, facecolor=COLORS["head"], edgecolor="none")
        for lo, hi in L["stale"]:
            stripe(x, y, lo, hi, facecolor=STALE, edgecolor="none")
        for lo, hi in L["cbs"]:
            stripe(x, y, lo, hi, facecolor=COLORS["cb"], edgecolor="none")
        for lo, hi in L["now"]:
            stripe(x, y, lo, hi, facecolor=OVERWRITE, edgecolor="none")
        for lo, hi in L["reload"]:
            stripe(x, y, lo, hi, facecolor="none", edgecolor="#1b5e20", hatch="xxxx", linewidth=0.3)
        ax.add_patch(Rectangle((x - hh / 2, y - hh / 2), hh, hh, fill=False, edgecolor="#303030", linewidth=0.5))
        pct = 100 * L["used_bytes"] / (top - base)
        ax.text(
            x,
            y + 0.24,  # below the head band, so the label never hides the overlap it describes
            f"{pct:.0f}%",
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.83, "pad": 0.9},
        )
        summary[f"{x},{y}"] = {
            "used_bytes": L["used_bytes"],
            "stale_bytes_excluded": L["stale_bytes"],
            "used_percent": round(pct, 3),
            "head_overwritten_now": L["now"],
            "head_stale": L["stale"],
            "reloaded": L["reload"],
        }
    ax.set_xlim(-0.55, 10.55)
    ax.set_ylim(9.55, -0.55)
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID_X))
    ax.set_yticks(range(GRID_Y))
    ax.set_xlabel("Tensix x")
    ax.set_ylabel("Tensix y")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    state = "allocated" if op["head_allocated"] else "FREED for this op (bracket; L1 untouched)"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.928,
        f"Pinned head {cols:,} cols at S = {head:,}, {state}.\n"
        f"Each square: bottom = address 0, top = {top:,} B. "
        f"Utilization = CB + tensor data + intact head / {top - base:,} allocatable B; stale head bytes excluded.",
        ha="center",
        fontsize=9,
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["reserved"], label="fixed reservation"),
            Patch(facecolor=COLORS["cb"], label="program CB (bottom-up)"),
            Patch(facecolor=COLORS["tensor"], label="L1 tensor (top-down)"),
            Patch(facecolor=COLORS["head"], label="pinned head, weights intact (allocated or freed)"),
            Patch(facecolor=OVERWRITE, label="head range under this op's CBs (overwritten now)"),
            Patch(facecolor=STALE, label="head range overwritten earlier, awaiting reload (not counted)"),
            Patch(facecolor="white", edgecolor="#1b5e20", hatch="xxxx", label="copied back from DRAM by this reload"),
            Patch(facecolor="white", edgecolor="#777777", label="free"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    fig.savefig(destination.with_suffix(".svg"))
    fig.savefig(destination.with_suffix(".png"), dpi=150)
    pdf.savefig(fig)
    plt.close(fig)
    return summary


def select(ops, total):
    """Drafter step 0: matmuls of head_and_other / layer 0, the layer-0 SDPA, the reload, the concat,
    and the step's bracketed global SDPA (the CB ceiling)."""
    end = next(o["start"] for o in ops if o["op"] == "MatmulDeviceOperation" and o["matmul"] == 24)
    chosen = []
    for op in ops:
        if op["start"] >= end:  # step 1 starts at its pre_projection (matmul 24)
            break
        name = op["op"]
        if name == "MatmulDeviceOperation":
            ident = call_identity(op["matmul"], total)
            if ident["model"] != "draft" or ident["draft_step"] != 0:
                continue
            if ident["layer"] in (0, 3):  # layer 3 is the global layer: its ops after the SDPA see the stale 8x4
                chosen.append((f'layer_{ident["layer"]:02d}', ident["role"], op))
            elif ident["layer"] is None:
                chosen.append(("head_and_other", ident["role"], op))
        elif name == "GenericOpDeviceOperation":
            chosen.append(("head_and_other", f'dram_reload_{len(op["reload_cores"])}cores', op))
        elif name == "RotaryEmbeddingDeviceOperation" and not op["head_allocated"]:
            chosen.append(("cb_ceiling", f"global_rope_layer{(op['matmul'] - 1) // 5:02d}", op))
        elif name == "ConcatDeviceOperation" and not op["head_allocated"]:
            chosen.append(("head_and_other", "split_logits_concat", op))
        elif name == "SdpaDecodeDeviceOperation":
            pos = op["matmul"]
            layer = (pos - 1) // 5
            if not op["head_allocated"]:
                chosen.append(("cb_ceiling", f"global_sdpa_layer{layer:02d}", op))
            elif layer == 0:
                chosen.append(("layer_00", "sliding_sdpa", op))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir", type=Path)
    ap.add_argument("--ceiling", type=int, default=801024, help="GEMMA4_LMHEAD_L1_RELOAD_CEIL of the capture")
    args = ap.parse_args()
    root = args.capture_dir
    report = json.loads((root / "analysis/report.json").read_text())
    if report["errors"] or report["physical_mapping_unresolved_records"]:
        raise SystemExit("run lm_head_l1_analyze.py first; its report has errors")
    graph = json.loads((root / "iteration.graph.json").read_text())
    before = json.loads((root / "before_iteration.l1.json").read_text())
    base, top, head, head_size = report["base"], report["top"], report["head_addr"], report["head_shard_bytes"]
    _, _, stale_end = walk(graph, before, head, head_size, args.ceiling, {})  # steady state at entry
    ops, total, stale_after = walk(graph, before, head, head_size, args.ceiling, stale_end)
    if stale_after != stale_end:
        raise ValueError("the stale state at the end of the iteration is not periodic")
    out = root / "analysis/tensix_grid_reload"
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    with PdfPages(out / "representatives.pdf") as pdf:
        for group, role, op in select(ops, total):
            title = f'Drafter · step 0 · {group.replace("_", " ")} · {role} · graph op {op["start"]:05d}'
            stem = out / "draft" / group / f'{role}_{op["start"]:05d}'
            summary = render(op, title, base, top, head, report["cols"], stem, pdf)
            entries.append(
                {
                    "index": op["start"],
                    "model": "draft",
                    "group": group,
                    "role": role,
                    "op": op["op"],
                    "head_allocated": op["head_allocated"],
                    "reload_cores": len(op.get("reload_cores", [])),
                    "svg": stem.with_suffix(".svg").relative_to(out).as_posix(),
                    "png": stem.with_suffix(".png").relative_to(out).as_posix(),
                    "cores": summary,
                }
            )
    manifest = {
        "capture": str(root),
        "base": base,
        "top": top,
        "head_addr": head,
        "reload_ceiling": args.ceiling,
        "utilization": "union(program-local CBs, physical L1 tensor pages, intact head bytes, a reload's copied-back range) / allocatable L1; stale head bytes excluded (stale_bytes_excluded per core)",
        "stale_at_entry_cores": len(stale_end),
        "representative_plots": len(entries),
        "entries": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_index(out, entries, root.name + " (DRAM reload)")
    for e in entries:
        u = [c["used_percent"] for c in e["cores"].values()]
        print(
            f'{e["group"]:15s} {e["role"]:28s} op {e["index"]:05d} head_alloc={e["head_allocated"]!s:5s} '
            f"util {min(u):.0f}-{max(u):.0f}%"
        )


if __name__ == "__main__":
    main()
