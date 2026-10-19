#!/usr/bin/env python3
"""L1 address captures of the standalone drafter's K=3 fused trace, one per arm (gemma4-12b record §9.1,
space-time budget map).

Runs test_assistant_standalone_l1.py::test_lm_head_l1_address_capture under one standalone arm's env (arm names
as run_pin_reload_standalone.py) with GEMMA4_LMHEAD_PROFILE_OUT set, so the capture brackets the real trace
capture. Analyse each output with analyze_spacetime_map.py.

    python research_codes/mm_profiling/run_spacetime_capture.py \\
        --arms B+logitsl1,R59840_64_8_rope_drisc60c2+logitsl1 [--output generated/lmhead_l1_profile/spacetime]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from run_lmhead_e2e import REPO, setup_overlay
from run_lmhead_obw_e2e import env_for
from run_pin_reload_standalone import arm_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="B+logitsl1,R59840_64_8_rope_drisc60c2+logitsl1")
    p.add_argument("--output", type=Path, default=REPO / "generated/lmhead_l1_profile/spacetime")
    args = p.parse_args()
    setup_overlay()
    failed = []
    for arm in args.arms.split(","):
        out = args.output / arm
        out.mkdir(parents=True, exist_ok=False)
        base, extra, _ = arm_spec(arm)
        env = env_for(base)
        env.update(extra)
        env["GEMMA4_LMHEAD_PROFILE_OUT"] = str(out)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py",
            "-k",
            "test_lm_head_l1_address_capture",
        ]
        print("capturing", arm, flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / "run.log").write_text(proc.stdout)
        if proc.returncode != 0:
            failed.append(arm)
            print(f"  FAILED (exit {proc.returncode}), see {out / 'run.log'}", flush=True)
    if failed:
        raise SystemExit(f"captures failed: {failed}")


if __name__ == "__main__":
    main()
