#!/usr/bin/env python3
"""Standalone drafter, K=3 chained steps in one trace: DRAM head vs 38,720 pin vs 52,800 pin + DRAM reload
(gemma4-12b MEASUREMENT_RECORD §5.5.14, three-way comparison).

Instrument: test_assistant_standalone_l1.py::test_fused_draft_k_steps (§5.5.7 step level), tuned matmuls,
1x1, GEMMA4_DRAFT_K_SWEEP=3, a fresh process per cell. Arms:
  A  no pin, drafter head auto out_block_w (75)                     (§5.5.13 arm A, drafter side)
  B  no pin, drafter head out_block_w 5                             (§5.5.13 arm B, drafter side)
  D  38,720 pinned cols, tail out_block_w 8                         (§5.5.13 arm D)
  R  52,800 pinned cols + GEMMA4_LMHEAD_L1_RELOAD, tail per_core_N 60 / out_block_w 4   (Gate 5 arm R)

The drafter runs alone here, so no target SDPA overwrites the head: R reloads 32 cores before every
lm_head (the drafter SDPA's 8x4), never the 110 of the demo's first step.

Gates (abort): exit code; the knob log lines (tail / whole-head width, reload registration at S=589,824,
reloads on 32 cores only).

    python research_codes/mm_profiling/run_pin_reload_standalone.py --rounds 3 [--arms A,D,R,R56320_60_4]
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from run_lmhead_e2e import REPO, setup_overlay
from run_lmhead_obw_e2e import env_for

ARMS = {
    "A": ("A", {}),
    "B": ("B", {}),
    "D": ("D", {}),
    "R": (
        "D",
        {
            "GEMMA4_LMHEAD_L1_COLS": "52800",
            "GEMMA4_LMHEAD_L1_RELOAD": "1",
            "GEMMA4_LMHEAD_DRAFT_OBW": "4",
            "GEMMA4_LMHEAD_DRAFT_PCN": "60",
        },
    ),
}
WIDTH = {"A": set(), "B": {("262144", "75", "5")}, "D": {("223424", "64", "8")}, "R": {("209344", "60", "4")}}


# "+blk<group><cap>" arm suffix (record §9.2): per-shape in0_block_w cap (GEMMA4_MM_BLOCK_CAP_SHAPES) on the drafter
# matmuls of one group; each shape gets the largest divisor of its Kt that is <= cap
BLK_GROUPS = {
    "fatk": [(4096, 1024), (8192, 1024), (7680, 1024)],  # o_proj sliding, down_proj / o_proj global, pre_projection
    "wide": [(1024, 8192), (1024, 9216), (1024, 3840)],  # qkv sliding / gate / up, qkv global, post_projection
}


# "+gridwide" (record §9.2): (K, N) -> (per_core_N, grid x, grid y) on the 11 x 10 grid, i.e. the widest legal
# DRAM-fed grid: ceil(Nt / per_core_N) cores = 86, 96, 60
GRID_WIDE = {(1024, 8192): (3, 11, 8), (1024, 9216): (3, 11, 9), (1024, 3840): (2, 11, 6)}


def blk_expected(arm):
    """{(K, N): in0_block_w} the arm's +blk suffixes should produce."""
    out = {}
    for group, cap in re.findall(r"\+blk(fatk|wide)(\d+)", arm):
        for k, n in BLK_GROUPS[group]:
            out[(k, n)] = max(d for d in range(1, int(cap) + 1) if (k // 32) % d == 0)
    return out


def arm_spec(arm):
    """A fixed arm, or R<cols>_<tail per_core_N>_<tail out_block_w>[_rope][_drisc[<GB/s>][c<pages>]] (pin + reload
    at another size; _rope also brackets the global layer's rotary embedding, GEMMA4_LMHEAD_L1_RELOAD_ROPE; _drisc
    pushes the reload from the DRAM cores during the window, GEMMA4_LMHEAD_L1_RELOAD_DRISC, optionally paced to
    <GB/s> (…_DRISC_GBPS) with <pages>-page DMA chunks (…_DRISC_CHUNK)). A "+logitsl1" suffix on any arm also
    writes the drafter lm_head logits to L1 (GEMMA4_LMHEAD_LOGITS_L1). Trailing "+pin<weight>[@<layers>][p<per_core_N>][r]"
    suffixes (record §9.2) also pin those drafter weights L1 WIDTH_SHARDED (GEMMA4_L1_PIN; p = own grid with that
    per_core_N instead of the tuner's, r = gather_in0 ring), e.g. "B+logitsl1+pindown_proj@0r". A "+fatk_dram_ring"
    suffix runs the fat-K matmuls (o_proj, down_proj; the ring guard refuses pre_projection) as the gather_in0 ring
    with their weights left in DRAM (GEMMA4_GATHER_DRAM_WEIGHT, record §9.2). "+blkfatk<cap>" / "+blkwide<cap>" cap
    in0_block_w for one BLK_GROUPS group (GEMMA4_MM_BLOCK_CAP_SHAPES). "+nocswap" swaps the 1D-mcast matmuls' in0 / in1
    NoCs (TT_MM1D_MCAST_IN0_SWAP_NOCS; needs mcast1d_swap_nocs.patch built in)."""
    if "+pin" in arm:
        base, *pins = arm.split("+pin")
        base, extra, width = arm_spec(base)
        spec, ring = [], False
        for pin in pins:
            m = re.fullmatch(r"([a-z_]+)(?:@([\d-]+))?(?:p(\d+))?(r)?", pin)
            if not m:
                raise SystemExit(f"unknown pin {pin!r} in {arm}")
            name, layers, pcn, r = m.groups()
            spec.append(name + (f"@{layers}" if layers else "") + (f":{pcn}" if pcn else ""))
            ring |= bool(r)
        extra = {
            **extra,
            "GEMMA4_STANDALONE_PLACEMENT": "l1_sharded",
            "GEMMA4_L1_PIN": ";".join(spec),
            "GEMMA4_L1_WEIGHT_BUDGET_MB": "4096",
        }
        if ring:
            extra.update({"GEMMA4_GATHER_IN0": "1", "GEMMA4_GATHER_DEBUG": "1"})
        return base, extra, width
    m = re.search(r"\+blk(fatk|wide)(\d+)$", arm)
    if m:
        base, extra, width = arm_spec(arm[: m.start()])
        group, cap = m.groups()
        spec = [x for x in extra.get("GEMMA4_MM_BLOCK_CAP_SHAPES", "").split(";") if x]
        spec += [f"{k}x{n}={cap}" for k, n in BLK_GROUPS[group]]
        return base, {**extra, "GEMMA4_MM_BLOCK_CAP_SHAPES": ";".join(spec)}, width
    if arm.endswith("+dramshard"):
        # fat-K weights DRAM WIDTH_SHARDED, run as the DRAM-sharded matmul at the tuner's in0_block_w (record §9.2a;
        # bit-identical in isolation, test_dram_sharded_fatk.py)
        base, extra, width = arm_spec(arm[: -len("+dramshard")])
        return base, {**extra, "GEMMA4_DRAM_SHARD": "pre_projection,o_proj,down_proj"}, width
    m = re.search(r"\+sl(ow\d+|osw\d+|iw\d+|fid\w+|ring)$", arm)
    if m:
        # lm_head L1 slice config (record §9.2b C4): ow / osw class C; iw / fid / ring class A / B, mechanism only
        base, extra, width = arm_spec(arm[: m.start()])
        v = m.group(1)
        key = {"ow": "GEMMA4_LMHEAD_L1_OBW", "osw": "GEMMA4_LMHEAD_L1_OSW", "iw": "GEMMA4_LMHEAD_L1_IW"}
        if v == "ring":
            return base, {**extra, "GEMMA4_LMHEAD_L1_GATHER": "1"}, width
        if v.startswith("fid"):
            return base, {**extra, "GEMMA4_LMHEAD_L1_FIDELITY": v[3:]}, width
        k = "osw" if v.startswith("osw") else v[:2]
        return base, {**extra, key[k]: v[len(k) :]}, width
    m = re.search(r"\+headiw(\d+)$", arm)
    if m:
        # in0_block_w of the explicit-config lm_head calls (GEMMA4_LMHEAD_DRAFT_IW; record §9.2a C1). Reblocks the K
        # sum: class A, mechanism only
        base, extra, width = arm_spec(arm[: m.start()])
        return base, {**extra, "GEMMA4_LMHEAD_DRAFT_IW": m.group(1)}, width
    m = re.search(r"\+niusample(\d*)$", arm)
    if m:
        # sample the 1D-mcast in1 reader's NIU read-word counter per block, and with a number every that many
        # wall-clock ticks (TT_MM1D_IN1_NIU_SAMPLE -> MM_IN1_NIU_SAMPLE in the in1 reader, needs build_codex_current;
        # record §9.2 item 6b)
        base, extra, width = arm_spec(arm[: m.start()])
        return base, {**extra, "TT_MM1D_IN1_NIU_SAMPLE": m.group(1) or "1"}, width
    m = re.search(r"\+vc([0-3])$", arm)
    if m:
        # per-core static VC of every 1D-mcast matmul's in1 read requests (TT_MM1D_IN1_VC_MODE; record §5.5.12 part 9,
        # §9.2): 1 = logical y & 3, 2 = logical x & 3, 3 = (x + y) & 3
        base, extra, width = arm_spec(arm[: m.start()])
        return base, {**extra, "TT_MM1D_IN1_VC_MODE": m.group(1)}, width
    if arm.endswith("+gridwide"):
        # DRAM-fed qkv / gate / up on 86 cores, global qkv on 96, post_projection on 60 (GEMMA4_MM_GRID_SHAPES)
        base, extra, width = arm_spec(arm[: -len("+gridwide")])
        spec = ";".join(f"{k}x{n}={pcn}" for (k, n), (pcn, _, _) in GRID_WIDE.items())
        return base, {**extra, "GEMMA4_MM_GRID_SHAPES": spec}, width
    if arm.endswith("+nocswap"):
        # every 1D-mcast matmul's in1 reader + writer on NoC1 and its in0 kernels on NoC0
        # (mcast1d_swap_nocs.patch; record §5.5.12 part 8, §9.2)
        base, extra, width = arm_spec(arm[: -len("+nocswap")])
        return base, {**extra, "TT_MM1D_MCAST_IN0_SWAP_NOCS": "1"}, width
    if arm.endswith("+fatk_dram_ring"):
        base, extra, width = arm_spec(arm[: -len("+fatk_dram_ring")])
        return (
            base,
            {**extra, "GEMMA4_GATHER_IN0": "1", "GEMMA4_GATHER_DRAM_WEIGHT": "1", "GEMMA4_GATHER_DEBUG": "1"},
            width,
        )
    if arm.endswith("+logitsl1"):
        base, extra, width = arm_spec(arm[: -len("+logitsl1")])
        return base, {**extra, "GEMMA4_LMHEAD_LOGITS_L1": "1"}, width
    if arm in ARMS:
        return ARMS[arm][0], ARMS[arm][1], WIDTH[arm]
    m = re.fullmatch(r"H(\d+)_(\d+)", arm)
    if m:
        # whole lm_head in DRAM (no pin) on an explicit per_core_N / out_block_w (record §9.2 whole-head re-sweep)
        pcn, obw = m.groups()
        return "B", {"GEMMA4_LMHEAD_DRAFT_OBW": obw, "GEMMA4_LMHEAD_DRAFT_PCN": pcn}, {("262144", pcn, obw)}
    m = re.fullmatch(r"P(\d+)_(\d+)_(\d+)", arm)
    if m:
        # head pin of <cols> columns WITHOUT a reload (fits the 772,032 B/bank no-reload budget, record §9.1), DRAM
        # tail on its own per_core_N / out_block_w (record §9.2 additivity test)
        cols, pcn, obw = m.groups()
        extra = {"GEMMA4_LMHEAD_L1_COLS": cols, "GEMMA4_LMHEAD_DRAFT_OBW": obw, "GEMMA4_LMHEAD_DRAFT_PCN": pcn}
        return "D", extra, {(str(262144 - int(cols)), pcn, obw)}
    m = re.fullmatch(r"R(\d+)_(\d+)_(\d+)(_rope)?(_drisc(\d+)?(?:c(\d+))?)?", arm)
    if not m:
        raise SystemExit(f"unknown arm {arm}")
    cols, pcn, obw, rope, drisc, gbps, chunk = m.groups()
    extra = {
        "GEMMA4_LMHEAD_L1_COLS": cols,
        "GEMMA4_LMHEAD_L1_RELOAD": "1",
        "GEMMA4_LMHEAD_DRAFT_OBW": obw,
        "GEMMA4_LMHEAD_DRAFT_PCN": pcn,
    }
    if rope:
        extra["GEMMA4_LMHEAD_L1_RELOAD_ROPE"] = "1"
    if drisc:
        extra["GEMMA4_LMHEAD_L1_RELOAD_DRISC"] = "1"
    if gbps:
        extra["GEMMA4_LMHEAD_L1_RELOAD_DRISC_GBPS"] = gbps
    if chunk:
        extra["GEMMA4_LMHEAD_L1_RELOAD_DRISC_CHUNK"] = chunk
    return "D", extra, {(str(262144 - int(cols)), pcn, obw)}


def parse(log):
    # the K-sweep row "  3  <fused ms/iter> <ms/step> <tok/s/u> <control ms/iter> <ms/step> <speedup>x"
    # (control columns are "nan" when the control is skipped: GEMMA4_LMHEAD_LOGITS_L1)
    n = r"(?:[\d.]+|nan)"
    fused = re.search(rf"(?:^|- )\s*3\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+{n}\s+{n}\s+{n}x", log, re.M)
    part = {
        name: float(m.group(1))
        for name, pat in (
            ("backbone", r"backbone \(4 layers, no head\)\s+([\d.]+)"),
            ("head", r"CME head forward\s+([\d.]+)"),
            ("argmax", r"argmax_token_id \(\+2nd gather\)\s+([\d.]+)"),
            ("total", r"TOTAL\s+([\d.]+)"),
        )
        if (m := re.search(pat, log))
    }
    out = re.search(r"\[fused-k\] K=3 output: token=\[(\d+)\] hidden_sha256=(\w+)", log)
    pinned = re.search(r"\[placement:[^\]]*\] mode=l1_sharded L1=([\d.]+) MB/device over (\d+) tensors", log)
    return {
        "fused_ms_per_iter": float(fused.group(1)) if fused else None,
        **part,
        "token": int(out.group(1)) if out else None,
        "hidden_sha": out.group(2) if out else None,
        "pinned_mb": float(pinned.group(1)) if pinned else 0.0,
        "pinned_tensors": int(pinned.group(2)) if pinned else 0,
        "ring_takes": log.count("[gather] TAKE"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--arms", default="A,B,D,R", help="comma-separated; R<cols>_<pcn>_<obw> for other pin sizes")
    p.add_argument("--output", type=Path, default=REPO / "generated/lmhead_l1_profile/standalone_pin_reload")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_overlay()
    arms = args.arms.split(",")
    order = []
    for r in range(args.rounds):
        order += arms if r % 2 == 0 else arms[::-1]
    results = []
    for index, arm in enumerate(order):
        name = f"{index:02d}_{arm}"
        base, extra, expected_width = arm_spec(arm)
        env = env_for(base)
        env.update(extra)
        env["GEMMA4_DRAFT_K_SWEEP"] = "3"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py",
            "-k",
            "test_fused_draft_k_steps",
        ]
        print("running", name, flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.output / f"{name}.log").write_text(proc.stdout)
        reg = re.findall(r"\[lm-head-reload\] head at S=(\d+)", proc.stdout)
        reloads = sorted({int(n) for n in re.findall(r"\[lm-head-reload\] reload on (\d+) dirty cores", proc.stdout)})
        drisc_reloads = sorted(
            {int(n) for n in re.findall(r"\[lm-head-reload\] DRISC reload on (\d+) dirty cores", proc.stdout)}
        )
        push = re.search(r"DRISC last push: slowest DRISC \d+ ticks \(([\d.]+) us\)", proc.stdout)
        width = set(re.findall(r"GEMMA4_LMHEAD_DRAFT_OBW: N=(\d+) per_core_N=(\d+) out_block_w=(\d+)", proc.stdout))
        row = {
            "index": index,
            "arm": arm,
            "exit_code": proc.returncode,
            "log": f"{name}.log",
            "reload_S": reg,
            "reload_core_counts": reloads,
            "drisc_reload_core_counts": drisc_reloads,
            "drisc_last_push_us": float(push.group(1)) if push else None,
            "width": sorted(width),
            **parse(proc.stdout),
        }
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(row, flush=True)
        if proc.returncode != 0 or row["total"] is None or width != expected_width:
            raise SystemExit(f"cell {name} failed its exit/width gate: width={width}")
        ring = (arm.endswith("r") and "+pin" in arm) or "+fatk_dram_ring" in arm
        if ("+pin" in arm) != (row["pinned_tensors"] > 0) or ring != (row["ring_takes"] > 0):
            raise SystemExit(
                f"cell {name}: pin/ring gate failed: {row['pinned_tensors']} pinned, {row['ring_takes']} ring"
            )
        if ("+logitsl1" in arm.split("+pin")[0]) != ("[lm-head-logits] GEMMA4_LMHEAD_LOGITS_L1" in proc.stdout):
            raise SystemExit(f"cell {name}: logits-in-L1 knob gate failed")
        want = ([], [32]) if "_drisc" in arm else ([32], [])  # (Tensix reloads, DRISC reloads)
        if (
            arm.startswith("R")
            and "+pin" not in arm
            and (not reg or (reloads, drisc_reloads) != want or (arm == "R" and reg[:1] != ["589824"]))
        ):
            raise SystemExit(f"cell {name}: reload gate failed: S={reg} reloads={reloads} drisc={drisc_reloads}")
    for key in ("fused_ms_per_iter", "backbone", "head", "argmax", "total"):
        med = {a: statistics.median([r[key] for r in results if r["arm"] == a]) for a in arms}
        print(f"median {key}:", med, flush=True)


if __name__ == "__main__":
    main()
