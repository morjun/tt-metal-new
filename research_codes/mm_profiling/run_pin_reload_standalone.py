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

    python research_codes/mm_profiling/run_pin_reload_standalone.py --rounds 3
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


def parse(log):
    # the K-sweep row "  3  <fused ms/iter> <ms/step> <tok/s/u> <control ms/iter> <ms/step> <speedup>x"
    fused = re.search(r"(?:^|- )\s*3\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+x", log, re.M)
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
    return {"fused_ms_per_iter": float(fused.group(1)) if fused else None, **part}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--arms", default="ABDR")
    p.add_argument("--output", type=Path, default=REPO / "generated/lmhead_l1_profile/standalone_pin_reload")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_overlay()
    arms = list(args.arms)
    order = []
    for r in range(args.rounds):
        order += arms if r % 2 == 0 else arms[::-1]
    results = []
    for index, arm in enumerate(order):
        name = f"{index:02d}_{arm}"
        base, extra = ARMS[arm]
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
        width = set(re.findall(r"GEMMA4_LMHEAD_DRAFT_OBW: N=(\d+) per_core_N=(\d+) out_block_w=(\d+)", proc.stdout))
        row = {
            "index": index,
            "arm": arm,
            "exit_code": proc.returncode,
            "log": f"{name}.log",
            "reload_S": reg,
            "reload_core_counts": reloads,
            "width": sorted(width),
            **parse(proc.stdout),
        }
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(row, flush=True)
        if proc.returncode != 0 or row["total"] is None or width != WIDTH[arm]:
            raise SystemExit(f"cell {name} failed its exit/width gate: width={width}")
        if arm == "R" and (reg[:1] != ["589824"] or reloads != [32]):
            raise SystemExit(f"cell {name}: reload gate failed: S={reg} reloads={reloads}")
    for key in ("fused_ms_per_iter", "backbone", "head", "argmax", "total"):
        med = {a: statistics.median([r[key] for r in results if r["arm"] == a]) for a in arms}
        print(f"median {key}:", med, flush=True)


if __name__ == "__main__":
    main()
