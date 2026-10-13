#!/usr/bin/env python3
"""Traced per-op device profiles of the standalone drafter, per arm and profiler mode (gemma4-12b record §8,
resource profile).

Each cell runs test_drisc_traced_opprof.py (fused K=3 body captured, OPPROF_REPLAYS replays, device profiler read
after each) under Tracy with one standalone arm's env (arm names as run_pin_reload_standalone.py), then the legacy
post-processor on that cell's .logs, which gives PER CORE MAX, per-RISC durations, tensor shapes / memory and the
counter or NoC columns. Modes:
  time         5 replays; plain device profile
  fpu          3 replays; --profiler-capture-perf-counters=fpu
  unpack_pack  3 replays; --profiler-capture-perf-counters=unpack,pack
  instrn       3 replays; --profiler-capture-perf-counters=instrn (thread stall rates)
  noc          1 replay;  --collect-noc-traces, analysed by tt-npe (NPE_ROOT, default ~/codes/tt-npe;
               tt-npe 3ce4534, the last before it needs a soc_descriptor.yaml this tree does not write)

    python research_codes/mm_profiling/run_traced_opprof_arms.py --cells time:A,time:B59,time:B59,time:A \\
        [--output generated/lmhead_l1_profile/resource_profile]

B59 is shorthand for R59840_64_8_rope_drisc60c2. Writes <output>/<index>_<mode>_<arm>/ with run.log,
cpp_device_perf_report.csv (time mode), ops_perf.csv (legacy) and npe_viz/ (noc mode).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from run_lmhead_e2e import REPO, setup_overlay
from run_lmhead_obw_e2e import env_for
from run_pin_reload_standalone import arm_spec

ALIAS = {"B59": "R59840_64_8_rope_drisc60c2"}
MODES = {
    "time": (5, []),
    "fpu": (3, ["--profiler-capture-perf-counters=fpu"]),
    "unpack_pack": (3, ["--profiler-capture-perf-counters=unpack,pack"]),
    "instrn": (3, ["--profiler-capture-perf-counters=instrn"]),
    "noc": (1, ["--collect-noc-traces"]),
}
LOGS = REPO / "generated/profiler/.logs"
REPORTS = REPO / "generated/profiler/reports"
NPE_ROOT = Path(os.getenv("NPE_ROOT", Path.home() / "codes/tt-npe"))


def npe_env(env):
    inst = NPE_ROOT / "install"
    env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{inst / 'lib'}:{inst / 'bin'}"
    env["PATH"] = f"{env['PATH']}:{inst / 'bin'}"
    env["TT_NPE_DEVICE_MODEL_CONFIG_DIR"] = str(inst / "share/tt_npe/device/models")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True, help="comma-separated <mode>:<arm>[:<replays>]")
    p.add_argument("--output", type=Path, default=REPO / "generated/lmhead_l1_profile/resource_profile")
    p.add_argument("--start-index", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_overlay()
    failed = 0
    for index, cell in enumerate(args.cells.split(","), args.start_index):
        mode, arm, *rep = cell.split(":")  # optional :<replays> overrides the mode's default
        arm = ALIAS.get(arm, arm)
        replays, flags = MODES[mode]
        replays = int(rep[0]) if rep else replays
        out = args.output / f"{index:02d}_{mode}_{arm}"
        out.mkdir(exist_ok=True)
        base, extra, _ = arm_spec(arm)
        env = env_for(base)
        env.update(extra)
        env["OPPROF_REPLAYS"] = str(replays)
        if mode != "time":
            # NoC event and counter markers overflow the default profiler DRAM buffer within one iteration
            # (tt-npe's advice for NoC traces; counters fill BRISC's buffer)
            env["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = os.getenv("OPPROF_PROGRAMS", "10000")
        if mode == "noc":
            npe_env(env)
        cmd = [sys.executable, "-m", "tracy", "-r", "-p", "-v", *flags, "-m", "pytest", "-s", "-q"]
        cmd.append("research_codes/mm_profiling/test_drisc_traced_opprof.py")
        print("running", out.name, flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = proc.stdout
        for name in ("cpp_device_perf_report.csv", "tracy_ops_data.csv"):
            if (LOGS / name).exists() and mode == "time":
                shutil.copy(LOGS / name, out / name)
        # legacy post-processor on this cell's .logs: PER CORE MAX, per-RISC, shapes, counter / NoC columns
        rep = REPORTS / f"{args.output.name}_{out.name}"
        if rep.exists():
            shutil.rmtree(rep)
        post = [sys.executable, "tools/tracy/process_ops_logs.py", "--force-legacy-device-logs"]
        post += ["-n", rep.name] + (["--analyze-noc-traces"] if mode == "noc" else [])
        pp = subprocess.run(post, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log += "\n==== legacy post-processing ====\n" + pp.stdout
        (out / "run.log").write_text(log)
        csvs = sorted(rep.rglob("ops_perf_results*.csv"))
        if csvs:
            shutil.copy(csvs[-1], out / "ops_perf.csv")
        npe = sorted(rep.rglob("npe_viz"))
        if npe:
            shutil.copytree(npe[-1], out / "npe_viz", dirs_exist_ok=True)
        dropped = len(re.findall(r"markers were dropped", log))
        ok = proc.returncode == 0 and pp.returncode == 0 and bool(csvs)
        failed += not ok
        print(
            f"{out.name}: exit {proc.returncode}/{pp.returncode}, legacy csv {bool(csvs)}, "
            f"dropped-marker warnings {dropped}, npe_viz {bool(npe)}",
            flush=True,
        )
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
