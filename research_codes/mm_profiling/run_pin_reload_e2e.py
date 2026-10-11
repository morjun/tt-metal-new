#!/usr/bin/env python3
"""In-demo prototype of the pinned-head DRAM-reload scheme (gemma4-12b MEASUREMENT_RECORD §5.5.14, Gate 5).

Arms (tuned matmuls, fused traced route, 500 tokens, fresh process per cell; protocol of
run_lmhead_obw_e2e.py):
  A  no pin, lm_head auto out_block_w                                      (§5.5.13 arm A)
  B  no pin, drafter head out_block_w 5, target 3                          (§5.5.13 arm B)
  D  38,720 pinned cols, tail out_block_w 8, target 3                        (§5.5.13 arm D)
  R  52,800 pinned cols + GEMMA4_LMHEAD_L1_RELOAD, tail per_core_N 60 / out_block_w 4, target 3
     (the head is freed around each global SDPA and the split-logits concat, and reloaded from
     DRAM on the dirty cores before each drafter lm_head; tt/lm_head_reload.py)

Gates (abort): exit code, 500 tokens, fused-traced route; R must log the reload registration at
S=589,824 and both reload core counts; the tail must log per_core_N 60 / out_block_w 4 in R.
Every cell must produce the same text, acceptance and iteration count.

    python research_codes/mm_profiling/run_pin_reload_e2e.py --rounds 6 [--arms R --rounds 1] [--arms ABDR]
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from run_lmhead_e2e import REPO, parse, setup_overlay
from run_lmhead_obw_e2e import ARMS as ARMS_OBW
from run_lmhead_obw_e2e import env_for

ARMS = {
    "A": {},
    "B": {},
    "D": {},
    "R": {
        "GEMMA4_LMHEAD_L1_COLS": "52800",
        "GEMMA4_LMHEAD_L1_RELOAD": "1",
        "GEMMA4_LMHEAD_DRAFT_OBW": "4",
        "GEMMA4_LMHEAD_DRAFT_PCN": "60",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--arms", default="DR")
    p.add_argument("--output", type=Path, default=REPO / "generated/lmhead_l1_profile/e2e_500_pin_reload")
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
        env = env_for(arm if arm in ARMS_OBW else "D")
        env.update(ARMS[arm])
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "models/demos/gemma4/demo/text_demo_v2.py",
            "-k",
            "test_demo_spec_decode",
            "--tt-arch",
            "blackhole",
        ]
        print("running", name, flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.output / f"{name}.log").write_text(proc.stdout)
        metrics = parse(proc.stdout)
        reg = re.findall(r"\[lm-head-reload\] head at S=(\d+)", proc.stdout)
        reloads = sorted({int(n) for n in re.findall(r"\[lm-head-reload\] reload on (\d+) dirty cores", proc.stdout)})
        tail = sorted(
            set(re.findall(r"GEMMA4_LMHEAD_DRAFT_OBW: N=(\d+) per_core_N=(\d+) out_block_w=(\d+)", proc.stdout))
        )
        row = {
            "index": index,
            "arm": arm,
            "exit_code": proc.returncode,
            "log": f"{name}.log",
            "reload_S": reg,
            "reload_core_counts": reloads,
            "tail": tail,
            **metrics,
        }
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            {
                k: row[k]
                for k in (
                    "index",
                    "arm",
                    "exit_code",
                    "ms_per_iteration",
                    "accepted_per_3",
                    "iterations",
                    "reload_S",
                    "reload_core_counts",
                    "tail",
                )
            },
            flush=True,
        )
        if proc.returncode != 0 or metrics["generated_tokens"] != 500 or metrics["route"] != "fused-traced":
            raise SystemExit(f"cell {name} failed its route/token gate")
        if arm == "R" and (reg[:1] != ["589824"] or reloads != [32, 110] or {t[1:] for t in tail} != {("60", "4")}):
            raise SystemExit(f"cell {name}: reload/tail gate failed: S={reg} reloads={reloads} tail={tail}")
    print(
        "identical text / acceptance / iterations across all cells:",
        len({r["text_sha256"] for r in results}) == 1,
        len({r["accepted_per_3"] for r in results}) == 1,
        len({r["iterations"] for r in results}) == 1,
        flush=True,
    )
    med = {a: statistics.median([r["ms_per_iteration"] for r in results if r["arm"] == a]) for a in arms}
    print("median ms/iteration:", med, flush=True)
    n = len(arms)
    rounds = [{x["arm"]: x["ms_per_iteration"] for x in results[r * n : (r + 1) * n]} for r in range(len(order) // n)]
    for a, b in [(x, y) for i, y in enumerate(arms) for x in arms[i + 1 :]]:
        paired = [round((cell[a] - cell[b]) * 1000, 1) for cell in rounds]
        print(f"{a} - {b} per round {paired} us, median {statistics.median(paired):+.1f} us/iter", flush=True)


if __name__ == "__main__":
    main()
