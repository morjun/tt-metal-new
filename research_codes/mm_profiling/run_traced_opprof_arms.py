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
import time
from pathlib import Path

from run_lmhead_e2e import REPO, setup_overlay
from run_lmhead_obw_e2e import env_for
from run_pin_reload_standalone import GRID_WIDE, arm_spec, blk_expected

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
        elif "+niusample" in arm or "+timeline" in arm:
            # timed NIU samples (item 6b) overflow the default buffer the same way
            env["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = os.getenv("OPPROF_PROGRAMS", "40000")
        if mode == "noc":
            npe_env(env)
        cmd = [sys.executable, "-m", "tracy", "-r", "-p", "-v", *flags, "-m", "pytest", "-s", "-q"]
        cmd.append("research_codes/mm_profiling/test_drisc_traced_opprof.py")
        print("running", out.name, flush=True)
        cell_start = time.time()
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = proc.stdout
        if "+niusample" in arm and (LOGS / "profile_log_device.csv").exists():
            # keep only the sampler's rows (device-log header + IN1_NIU_RD_WORDS); the raw log is not kept
            with open(LOGS / "profile_log_device.csv") as src, open(out / "niu_samples.csv", "w") as dst:
                for i, line in enumerate(src):
                    if i < 2 or "IN1_NIU_RD_WORDS" in line:
                        dst.write(line)
        if "+timeline" in arm and (LOGS / "profile_log_device.csv").exists():
            # keep only the timeline markers (record §9.2b M3) and the device-log header
            with open(LOGS / "profile_log_device.csv") as src, open(out / "timeline_samples.csv", "w") as dst:
                for i, line in enumerate(src):
                    if i < 2 or ",MM_TL," in line:
                        dst.write(line)
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
        if npe and mode == "noc":
            shutil.copytree(npe[-1], out / "npe_viz", dirs_exist_ok=True)
        # Every report written during this cell carries an npe_viz copy: this cell's named report (copied into the cell
        # above for noc cells), and the timestamped report the tracy run itself writes, which outside noc mode holds a
        # stale copy of the last noc run's generated/profiler/npe_viz (1.3-6 GB each). The cell keeps the only copy.
        for d in REPORTS.glob("*/npe_viz"):
            if d.stat().st_mtime >= cell_start - 1:
                shutil.rmtree(d)
        npe = npe if mode == "noc" else []
        dropped = len(re.findall(r"markers were dropped", log))
        ok = proc.returncode == 0 and pp.returncode == 0 and bool(csvs)
        if ("+pin" in arm) != ("largest tensors pinned to L1" in proc.stdout):
            raise SystemExit(f"{out.name}: pin gate failed (arm pins weights but none were pinned, or vice versa)")
        for (k, n), blk in blk_expected(arm).items():
            tuned = [line for line in proc.stdout.splitlines() if f"M=1 K={k} N={n} -> TUNED" in line]
            if not tuned or any(f"in0_block_w={blk} " not in line for line in tuned):
                raise SystemExit(f"{out.name}: block-cap gate failed ({k}x{n} did not get in0_block_w={blk})")
        for (k, n), (pcn, gx, gy) in GRID_WIDE.items() if "+gridwide" in arm else ():
            want = f"M=1 K={k} N={n} -> TUNED grid={gx}x{gy} "
            tuned = [line for line in proc.stdout.splitlines() if want in line]
            if not tuned or any(f"per_core_N={pcn}" not in line for line in tuned):
                raise SystemExit(f"{out.name}: grid gate failed ({k}x{n} not on {gx}x{gy} with per_core_N={pcn})")
        if "+fatk_dram_ring" in arm and not all(f"[gather] TAKE {k}x1024" in proc.stdout for k in (4096, 8192)):
            raise SystemExit(f"{out.name}: fat-K ring gate failed (o_proj / down_proj did not take the ring)")
        failed += not ok
        print(
            f"{out.name}: exit {proc.returncode}/{pp.returncode}, legacy csv {bool(csvs)}, "
            f"dropped-marker warnings {dropped}, npe_viz {bool(npe)}",
            flush=True,
        )
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
