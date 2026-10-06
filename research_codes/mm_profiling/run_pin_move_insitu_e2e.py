#!/usr/bin/env python3
"""In-situ cost of moving pinned lm_head bytes around the global-layer SDPAs (12B, record §5.5.14).

Every arm is §5.5.13 arm D (38,720 pinned cols, tail out_block_w 8, target 3). The dummy arms add,
around each of the 11 global SDPAs per iteration, the L1 round trip that a pin needing that many
overlap bytes per SDPA core would cost (models/demos/gemma4/tt/pin_move_dummy.py, junk data):

  0  no moves                                  (baseline = arm D)
  S  14,592 B/core   (overlap of a 42,240-col pin)
  M  80,128 B/core   (45,760)
  L  145,664 B/core  (49,280)

Protocol: as run_lmhead_obw_e2e.py. Fused traced route, 500 tokens, a fresh process per cell,
order 0SML / LMS0 alternating. Gates (abort): exit code, 500 tokens, the fused-traced route, and the
[pin-move-dummy] lines: none in arm 0; in the dummy arms, both the drafter (batch 1) and the target
(batch > 1) cases at the arm's byte count.

The moves must not change the output, so the text, acceptance and iteration count must be
identical in every cell. This is checked and reported.

Standalone prediction (schedule A, record §5.5.14, Gate 2):
- S: 8 x 7.1 + 3 x 6.9 = 77.5 µs/iter
- M: 160.7 µs/iter
- L: 268.1 µs/iter

    python research_codes/mm_profiling/run_pin_move_insitu_e2e.py --rounds 6 [--arms 0M --rounds 1]
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from run_lmhead_e2e import REPO, parse, setup_overlay
from run_lmhead_obw_e2e import env_for

ARMS = {"0": 0, "S": 14_592, "M": 80_128, "L": 145_664}
PREDICTED_US = {"S": 77.5, "M": 160.7, "L": 268.1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--arms", default="0SML")
    p.add_argument("--output", type=Path, default=REPO / "generated/lmhead_l1_profile/e2e_500_pin_move_insitu")
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
        env = env_for("D")
        if ARMS[arm]:
            env["GEMMA4_PIN_MOVE_DUMMY_B"] = str(ARMS[arm])
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
        moves = re.findall(r"\[pin-move-dummy\] (out|back) batch=(\d+) \S+ cores (\d+) B/core", proc.stdout)
        row = {
            "index": index,
            "arm": arm,
            "dummy_B_per_core": ARMS[arm],
            "exit_code": proc.returncode,
            "log": f"{name}.log",
            "move_lines": len(moves),
            "move_batches": sorted({int(b) for _, b, _ in moves}),
            **metrics,
        }
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            {k: row[k] for k in ("index", "arm", "ms_per_iteration", "accepted_per_3", "iterations", "move_lines")},
            flush=True,
        )
        if proc.returncode != 0 or metrics["generated_tokens"] != 500 or metrics["route"] != "fused-traced":
            raise SystemExit(f"cell {name} failed its route/token gate")
        if ARMS[arm] == 0:
            ok = not moves
        else:
            ok = {int(b) for _, _, b in moves} == {ARMS[arm]} and 1 in row["move_batches"]
            ok = ok and any(b > 1 for b in row["move_batches"])
        if not ok:
            raise SystemExit(f"cell {name}: [pin-move-dummy] lines do not match arm {arm} ({len(moves)} lines)")
    print(
        "identical text / acceptance / iterations across all cells:",
        len({r["text_sha256"] for r in results}) == 1,
        len({r["accepted_per_3"] for r in results}) == 1,
        len({r["iterations"] for r in results}) == 1,
        flush=True,
    )
    med = {a: statistics.median([r["ms_per_iteration"] for r in results if r["arm"] == a]) for a in arms}
    print("median ms/iteration:", med, flush=True)
    if "0" in arms:
        rounds = len(order) // len(arms)
        for a in arms:
            if a == "0":
                continue
            paired = []
            for r in range(rounds):
                cell = {x["arm"]: x["ms_per_iteration"] for x in results[r * len(arms) : (r + 1) * len(arms)]}
                paired.append(round((cell[a] - cell["0"]) * 1000, 1))
            print(
                f"{a} - 0: per round {paired} us, median {statistics.median(paired):+.1f} us/iter "
                f"(standalone prediction +{PREDICTED_US[a]} us)",
                flush=True,
            )


if __name__ == "__main__":
    main()
