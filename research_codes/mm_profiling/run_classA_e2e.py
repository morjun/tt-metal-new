#!/usr/bin/env python3
"""End-to-end check of class-A drafter configurations (gemma4-12b record §9.2b, "best allowing non-bit-exact output").

Each arm is a standalone arm spec (run_pin_reload_standalone.arm_spec), applied to the fused traced spec-decode demo
(text_demo_v2.py::test_demo_spec_decode, 500 tokens, the protocol of run_lmhead_obw_e2e.py) as env_for(base) plus the
arm's knobs, as run_traced_opprof_arms.py does. Class-A drafter knobs change the drafted tokens and so acceptance, so
the measure is ms per iteration, which folds acceptance in: (acc + 1) / steady. The target verifies every token, so
every cell must produce the same text; a text change means the target path changed and stops the run.

    python research_codes/mm_profiling/run_classA_e2e.py --arms "<arm>,<arm>" --rounds 3 --output <dir>
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from run_lmhead_e2e import REPO, parse, setup_overlay
from run_lmhead_obw_e2e import env_for
from run_pin_reload_standalone import arm_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_overlay()
    arms = args.arms.split(",")
    order = []
    for r in range(args.rounds):
        order += arms if r % 2 == 0 else arms[::-1]
    results = []
    for index, arm in enumerate(order):
        base, extra, _ = arm_spec(arm)
        env = env_for(base)
        env.update(extra)
        name = f"{index:02d}_{arm}"
        cmd = [sys.executable, "-m", "pytest", "-q", "-s", "models/demos/gemma4/demo/text_demo_v2.py"]
        cmd += ["-k", "test_demo_spec_decode", "--tt-arch", "blackhole"]
        print("running", name, flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.output / f"{name}.log").write_text(proc.stdout)
        row = {"index": index, "arm": arm, "exit_code": proc.returncode, **parse(proc.stdout)}
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print({k: row.get(k) for k in ("index", "arm", "ms_per_iteration", "accepted_per_3", "iterations")}, flush=True)
        if proc.returncode != 0 or row["generated_tokens"] != 500 or row["route"] != "fused-traced":
            raise SystemExit(f"cell {name} failed its route/token gate")
        if len({r["text_sha256"] for r in results}) != 1:
            raise SystemExit(f"cell {name}: generated text differs from earlier cells (target path changed)")
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm]
        ms = [r["ms_per_iteration"] for r in rows]
        print(
            f"{arm}: ms/iter {statistics.median(ms):.3f} (rounds {ms}), accepted/3 "
            f"{sorted({r['accepted_per_3'] for r in rows})}, iterations {sorted({r['iterations'] for r in rows})}",
            flush=True,
        )


if __name__ == "__main__":
    main()
